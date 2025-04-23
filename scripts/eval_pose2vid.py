import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List

import av
import numpy as np
import torch
import torchvision
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection

from configs.prompts.test_cases import TestCasesDict
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from src.cameractrl.pose_adaptor import CameraPoseEncoder
from src.dataset.dance_image_h_v_camera import Camera, ray_condition
import numpy as np
from src.utils.util import get_fps, read_frames, save_videos_grid

def get_relative_pose(cam_params):
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    source_cam_c2w = abs_c2ws[0]
    cam_to_origin = 0
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -cam_to_origin],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses
    
def load_cameras(pose_file, img_size):
    with open(pose_file, 'r') as f:
        poses = f.readlines()
    poses = [pose.strip().split(' ') for pose in poses[:]]
    cam_params = [[float(x) for x in pose] for pose in poses]
    cam_params = [Camera(cam_param, pose_file, img_size) for cam_param in cam_params]
    return cam_params

def camera_file_to_embedding(video_length, pose_path, ref_img_idx, tgt_img_idx, img_size, is_same_video = True):
    camera_file = pose_path.replace("/dwpose/", "/camera/").replace("/pose_videos/", "/camera/").replace(".mp4", ".txt").replace(".png", ".txt")
    #import ipdb; ipdb.set_trace()
    if not os.path.exists(camera_file):
        print(f"Camera file not found: {camera_file}, using static camera")
        static_cam_params = [0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, -0.0]
        cam_params = [Camera(static_cam_params, "test", img_size)] * video_length
    else:
        print(f"Camera file found: {camera_file}")
        cam_params = load_cameras(camera_file, img_size)
    #assert len(cam_params) == video_length, f"{len(cam_params) = } != {video_length = }"
    if is_same_video:
        cam_params = [cam_params[ref_img_idx]] + [cam_params[idx] for idx in tgt_img_idx]
    else:
        cam_params = [cam_params[tgt_img_idx[0]]] + [cam_params[idx] for idx in tgt_img_idx]
    intrinsics = np.asarray([[cam_param.fx * img_size[0],
                                cam_param.fy * img_size[1],
                                cam_param.cx * img_size[0],
                                cam_param.cy * img_size[1]]
                                for cam_param in cam_params[1:]], dtype=np.float32)

    intrinsics = torch.as_tensor(intrinsics)[None]                  # [1, 1, 4]
    c2w_poses = get_relative_pose(cam_params)[1:]              # [frame, 4, 4]
    c2w = torch.as_tensor(c2w_poses)[None]                          # [1, frame, 4, 4]
    flip_flag = None
    plucker_embedding = ray_condition(intrinsics, c2w, img_size[1], img_size[0], device='cpu',
                                        flip_flag=flip_flag)[0].permute(0, 3, 1, 2).contiguous()  # [frame, 6, H, W]
    return plucker_embedding.unsqueeze_(0)

def load_ids_from_file(input_file):
    video_ids_set = set()
    with open(input_file, 'r') as f:
        for line in f:
            video_id = line.strip()
            video_ids_set.add(video_id)
    return video_ids_set

#pexels_test_set = load_ids_from_file("/mnt/afs/wangzhenzhi/code/animate-with-camera/output/champ/pexels-test-h-ids.txt") 
#pexels_test_set.update(load_ids_from_file("/mnt/afs/wangzhenzhi/code/animate-with-camera/output/champ/pexels-test-v-ids.txt"))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=320)
    parser.add_argument("-L", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--fps", type=int)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    W = args.W
    H = args.H
    L = args.L
    seed = args.seed
    cfg = args.cfg
    steps = args.steps
    fps = 12
    config = args.config

    config = OmegaConf.load(config)

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device="cuda")

    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device="cuda")

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device="cuda"
    )
    camera_pose_encoder_kwargs = OmegaConf.to_container(infer_config.pose_encoder_kwargs)
    camera_pose_encoder = CameraPoseEncoder(**camera_pose_encoder_kwargs)
    keys_to_remove = []  #("encoder_down_attention_blocks.0.0.attention_blocks.0.pos_encoder.pe", "encoder_down_attention_blocks.0.1.attention_blocks.0.pos_encoder.pe")
    state_dict = torch.load(os.path.join(config.camera_pose_encoder_path), map_location="cpu")
    for key in keys_to_remove:
        if key in state_dict:
            del state_dict[key]
    camera_pose_encoder.load_state_dict(
        state_dict,
        strict=False,
    )

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device="cuda")

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    generator = torch.manual_seed(seed)

    width, height = W, H

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )
    pose_guider.load_state_dict(
        torch.load(config.pose_guider_path, map_location="cpu"),
    )


    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        camera_pose_encoder = camera_pose_encoder,
        scheduler=scheduler,
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)

    for repeat_idx in range(args.repeat):
        for ref_image_root in config['test_cases']:
            pose_root = config["test_cases"][ref_image_root][0]
            date_str = datetime.now().strftime("%Y%m%d")
            time_str = datetime.now().strftime("%H%M")
            save_dir_name = f"{time_str}--seed_{seed}-{W}x{H}"
            config_name = (Path(args.config).stem).split(".")[0]
            test_set_name = (ref_image_root).split('/')[-2]
            save_dir = Path(f"output/{config_name}/{test_set_name}_{date_str}_{save_dir_name}")
            save_dir.mkdir(exist_ok=True, parents=True)

            all_vids = os.listdir(ref_image_root)
            all_vids.sort()
            #all_vids = [f for f in all_vids if f.split('.')[-1] in ['png', 'jpg', 'mp4'] and f.split('.')[0] in pexels_test_set]
            length = len(all_vids)
            all_ref_images = [os.path.join(ref_image_root, f) for f in all_vids]#[int(length/2):]

            for ind, ref_image_path in enumerate(all_ref_images):
                print(f"Processing {ind+1}/{len(all_ref_images)}: {ref_image_path}")
                vid = ref_image_path.split('/')[-1]
                pose_video_path = os.path.join(pose_root, vid)
                if not os.path.exists(pose_video_path):
                    print(f"Skipping pose file {pose_video_path} as it does not exist.")
                    continue
                if not os.path.exists(ref_image_path):
                    print(f"Skipping ref_img file {ref_image_path} as it does not exist.")
                    continue
                ref_name = Path(ref_image_path).stem
                pose_name = Path(pose_video_path).stem.replace("_kps", "")

                pose_list = []
                pose_tensor_list = []
                pose_images = read_frames(pose_video_path)
                src_fps = get_fps(pose_video_path)
                
                # read width and height from first image
                img_width, img_height = pose_images[0].size
                larger_size = max(width, height)
                smaller_size = min(width, height)
                if img_width > img_height:
                    width = larger_size
                    height = smaller_size
                else:
                    width = smaller_size
                    height = larger_size
                video_length = len(pose_images)
                print(f"{vid}: pose video has {len(pose_images)} frames, with {src_fps} fps")
                stride = 3
                seq_len = L
                while len(pose_images) - seq_len * stride < 0:
                    stride -= 1
                if stride == 0:
                    print(f"{pose_name} has {len(pose_images)} frames, which is less than {seq_len * stride} frames needed for {L} length video.")
                    continue
                random_start = 0
                tgt_idx = np.arange(random_start, random_start + seq_len * stride, stride)

                image_transform = transforms.Compose(
                    [transforms.Resize((height, width)), transforms.ToTensor()])
                
                for idx in tgt_idx:
                    pose_image_pil = pose_images[idx]
                    pose_tensor_list.append(image_transform(pose_image_pil))
                    pose_list.append(pose_image_pil)
                
                ref_images = read_frames(ref_image_path)
                ref_random_idx = tgt_idx[len(tgt_idx) // 2]
                ref_image = ref_images[ref_random_idx]
                ref_image_tensor= image_transform(ref_image).unsqueeze(0).unsqueeze(2)

                ref_image_tensor = repeat(
                    ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=seq_len
                )

                pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
                pose_tensor = pose_tensor.transpose(0, 1)
                pose_tensor = pose_tensor.unsqueeze(0)

                camera_embedding = camera_file_to_embedding(video_length, pose_video_path, ref_random_idx, tgt_idx, (width, height), is_same_video =True) # (bs, f, c, H, W)
                #print(f"camera_embedding shape: {camera_embedding.shape}")
                camera_embedding = camera_embedding.transpose(1, 2)  # (bs, c, f, H, W)
                #print(f"camera_embedding shape: {camera_embedding.shape}")

                video = pipe(
                    ref_image,
                    pose_list,
                    camera_embedding,
                    width,
                    height,
                    seq_len,
                    steps,
                    cfg,
                    generator=generator,
                ).videos

                # save output video
                save_videos_grid(
                    video,
                    f"{save_dir}/output_{ref_name}.mp4",
                    n_rows=1,
                    fps=src_fps if fps is None else fps,
                )



if __name__ == "__main__":
    main()

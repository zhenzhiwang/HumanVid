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

#from configs.prompts.test_cases import TestCasesDict
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2img import Pose2ImagePipeline
from src.cameractrl.pose_adaptor import CameraPoseEncoder
from src.utils.util import get_fps, read_frames, save_image_grid
from metrics.utils.loss_utils import ssim
from metrics.lpipsPyTorch import lpips
import json
from tqdm import tqdm
from metrics.utils.image_utils import psnr
from src.dataset.dance_image_h_v_camera import Camera, ray_condition
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=896)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--eval_distance", type=int, default=30)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    return args

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


def camera_file_to_embedding(video_path, video_length, ref_img_idx, tgt_img_idx, img_size):
    camera_file = video_path.replace("/videos/", "/camera/").replace("/ref_images/", "/camera/").replace(".mp4", ".txt")
    if not os.path.exists(camera_file):
        print(f"Camera file not found: {camera_file}, using static camera")
        static_cam_params = [0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0]
        cam_params = [Camera(static_cam_params, "test", img_size)] * video_length
    else:
        print(f"Camera file found: {camera_file}")
        cam_params = load_cameras(camera_file, img_size)
    #assert len(cam_params) == video_length, f"{len(cam_params) = } != {video_length = }"
    cam_params = [cam_params[ref_img_idx], cam_params[tgt_img_idx]]
    intrinsics = np.asarray([[cam_param.fx * img_size[0],
                                cam_param.fy * img_size[1],
                                cam_param.cx * img_size[0],
                                cam_param.cy * img_size[1]]
                                for cam_param in cam_params[1:]], dtype=np.float32)

    intrinsics = torch.as_tensor(intrinsics)[None]                  # [1, 1, 4]
    c2w_poses = get_relative_pose(cam_params)[1:]              # [1, 4, 4]
    c2w = torch.as_tensor(c2w_poses)[None]                          # [1, 1, 4, 4]
    flip_flag = torch.zeros(1, dtype=torch.bool, device=c2w.device)
    plucker_embedding = ray_condition(intrinsics, c2w, img_size[1], img_size[0], device='cpu',
                                        flip_flag=flip_flag)[0].permute(0, 3, 1, 2).contiguous()[0]  # [6, H, W]
    return plucker_embedding.unsqueeze_(0)


def main():
    args = parse_args()

    W = args.W
    H = args.H
    seed = args.seed
    cfg = args.cfg
    steps = args.steps
    config_path = args.config
    # Load the configuration
    config = OmegaConf.load(config_path)

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
        "",
        subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
        },
    ).to(dtype=weight_dtype, device="cuda")

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device="cuda"
    )

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device="cuda")

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    generator = torch.manual_seed(seed)

    width, height = W, H
    camera_pose_encoder_kwargs = OmegaConf.to_container(infer_config.pose_encoder_kwargs)
    camera_pose_encoder = CameraPoseEncoder(**camera_pose_encoder_kwargs)
    keys_to_remove = ("encoder_down_attention_blocks.0.0.attention_blocks.0.pos_encoder.pe", "encoder_down_attention_blocks.0.1.attention_blocks.0.pos_encoder.pe")
    state_dict = torch.load(os.path.join(config.camera_pose_encoder_path), map_location="cpu")
    for key in keys_to_remove:
        if key in state_dict:
            del state_dict[key]
    camera_pose_encoder.load_state_dict(
        state_dict,
        strict=False,
    )
    
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
        strict=False,
    )
    pose_guider.load_state_dict(
        torch.load(config.pose_guider_path, map_location="cpu"),
    )
    pipe = Pose2ImagePipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        camera_pose_encoder=camera_pose_encoder,
        scheduler=scheduler,
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)
    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    save_dir_name = f"{time_str}--seed_{seed}-{width}x{height}"

    config_name = (Path(args.config).stem).split(".")[0]
    eval_dir = Path(f"output/{config_name}/{date_str}/")
    save_dir = Path(f"output/{config_name}/{date_str}/{save_dir_name}")
    save_dir.mkdir(exist_ok=True, parents=True)
    
    
    ssims = []
    psnrs = []
    lpipss = []
    l1_loss =[]
    for repeat_ind in range(args.repeat):
        for ref_image_root in config['test_cases']:
            pose_root = config["test_cases"][ref_image_root][0]

            all_vids = os.listdir(ref_image_root)
            all_vids.sort()
            all_ref_videos = [os.path.join(ref_image_root, f) for f in all_vids]

            for ind, ref_video_path in enumerate(all_ref_videos):
                vid = ref_video_path.split('/')[-1].split('.')[0]
                pose_video_path = os.path.join(pose_root, vid + '.mp4')

                all_ref_images = read_frames(ref_video_path)
                img_width, img_height = all_ref_images[0].size
                larger_size = max(width, height)
                smaller_size = min(width, height)
                if img_width > img_height:
                    width = larger_size
                    height = smaller_size
                else:
                    width = smaller_size
                    height = larger_size

                image_transform = transforms.Compose(
                    [transforms.Resize((height, width)), transforms.ToTensor()])
                #assert len(pose_images) == len(ref_images), f"{len(pose_images) = } != {len(ref_images) = }"
                ref_random_idx = np.random.randint(0, len(all_ref_images))
                ref_image = all_ref_images[ref_random_idx]
                ref_image_tensor= image_transform(ref_image)

                all_pose_images = read_frames(pose_video_path)
                #tgt_random_idx is args.eval_distance away from ref_random_idx
                valid_indices = list(range(0, max(0, ref_random_idx - args.eval_distance))) + \
                            list(range(min(len(all_pose_images), ref_random_idx + args.eval_distance), len(all_pose_images)))

                if not valid_indices:
                    tgt_random_idx  = np.random.randint(0, len(all_pose_images))
                else:
                    tgt_random_idx = np.random.choice(valid_indices)
                
                pose_image = all_pose_images[tgt_random_idx]
                pose_tensor_list = [image_transform(pose_image)]
                pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (1, c, h, w)

                print(f'video #{ind}: {vid} in {len(all_ref_videos)}')
                ref_name = str(ref_random_idx)
                pose_name = str(tgt_random_idx)
               
                video_length = len(all_ref_images)
                if 'no_camera' in config.keys() and config.no_camera:
                    camera_embedding = camera_file_to_embedding('', video_length, ref_random_idx, tgt_random_idx, (width, height))
                else:
                    camera_embedding = camera_file_to_embedding(ref_video_path, video_length, ref_random_idx, tgt_random_idx, (width, height))
                

                image = pipe(
                    ref_image,
                    pose_image,
                    camera_embedding,
                    width,
                    height,
                    steps,
                    cfg,
                    generator=generator,
                ).images

                image = image.squeeze(2)

                if config.data_type == 'frames':
                    gt_img = Image.open(all_ref_images[tgt_random_idx]).convert('RGB')
                else:
                    gt_img = all_ref_images[tgt_random_idx]
                gt_img = torch.stack([image_transform(gt_img)], dim=0)
                ssims.append(ssim(image, gt_img))
                psnrs.append(psnr(image, gt_img))
                lpipss.append(lpips(image, gt_img, net_type='vgg'))
                l1_loss.append(torch.nn.functional.l1_loss(image, gt_img).item())
                image = torch.cat([ref_image_tensor.unsqueeze(0), pose_tensor, image, gt_img], dim=0)
                save_image_grid(
                        image,
                        f"{save_dir}/{repeat_ind}_{vid}_{ref_name}_{pose_name}_{height}x{width}_{int(cfg)}_{time_str}.jpg",
                        n_rows=4,
                )
        ssim_result = torch.tensor(ssims).mean().item()
        psnr_result = torch.tensor(psnrs).mean().item()
        lpips_result = torch.tensor(lpipss).mean().item()
        l1_result = torch.tensor(l1_loss).mean().item()
        print(f"repeat id = {repeat_ind}, SSIM = {ssim_result}, PSNR = {psnr_result}, LPIPS = {lpips_result}, L1 = {l1_result}")

    ssim_result = torch.tensor(ssims).mean()
    psnr_result = torch.tensor(psnrs).mean()
    lpips_result = torch.tensor(lpipss).mean()
    l1_result = torch.tensor(l1_loss).mean().item()
    result_dict = {"SSIM": ssim_result.item(), "PSNR": psnr_result.item(), "LPIPS": lpips_result.item(), \
                   'test_set_name': ref_image_root, 'L1': l1_result, "method_name": config.camera_pose_encoder_path.split('/')[-2]}
    print("SSIM : {:>12.7f}".format(ssim_result, ".5"))
    print("PSNR : {:>12.7f}".format(psnr_result, ".5"))
    print("LPIPS: {:>12.7f}".format(lpips_result, ".5"))
    print("L1: {:>12.7f}".format(l1_result, ".5"))
    with open(os.path.join(eval_dir, f"{config.camera_pose_encoder_path.split('/')[-2]}_{save_dir_name}.json"), 'w') as fp:
        json.dump(result_dict, fp, indent=True)

if __name__ == "__main__":
    main()

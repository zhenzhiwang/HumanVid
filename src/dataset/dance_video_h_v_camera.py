import json
import random
from typing import List

import numpy as np
import torch
import torchvision.transforms as transforms
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
import os
import torchvision.transforms.functional as TF
from src.dataset.dance_image_h_v_camera import Camera, ray_condition, OrientationBatchSampler, OrientationSampler  #
from src.dataset.visualization_utils import CameraPoseVisualizer, visualize_camera_pose, to_image, pca_visualize  # 

class RandomResizeCrop(object):
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, img, scale, random_state=None):
        if random_state is not None:
            torch.set_rng_state(random_state)

        w, h = img.size
        aspect_ratio = w / h
        if aspect_ratio < 1:
            new_w = int(self.img_size[0] * scale)
            new_h = int(new_w / aspect_ratio)
            if new_h < self.img_size[1]:
                new_h = self.img_size[1]
                new_w = int(new_h * aspect_ratio)
                assert new_w >= self.img_size[0], f"new_w={new_w} < img_size_w={self.img_size[0]}"
        else:
            new_h = int(self.img_size[1] * scale)
            new_w = int(new_h * aspect_ratio)
            if new_w < self.img_size[0]:
                new_w = self.img_size[0]
                new_h = int(new_w / aspect_ratio)
                assert new_h >= self.img_size[1], f"new_h={new_h} < img_size_h={self.img_size[1]}"
        new_size = (new_h, new_w)
        img = TF.resize(img, new_size, interpolation=transforms.InterpolationMode.BILINEAR)

        # Center crop to target size
        w, h = img.size
        left = (w - self.img_size[0]) // 2
        top = (h - self.img_size[1]) // 2
        img = TF.crop(img, top, left, self.img_size[1], self.img_size[0])
        return img
    
class HumanDanceCameraVideoDataset(Dataset):
    def __init__(
        self,
        sample_rate,
        n_sample_frames,
        train_size_large,
        train_size_small,
        img_scale=(1.0, 1.1),
        drop_ratio=0.1,
        horizontal_meta =["./data/pexels-human_horizontal_meta.json"],
        vertical_meta = ["./data/pexels-human_vertical_meta.json"],
        visualize_dataset = False
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames
        self.train_size_large = train_size_large
        self.train_size_small = train_size_small
        self.img_scale = img_scale

        self.horizontal_meta = []
        self.vertical_meta = []
        if self.horizontal_meta is not None:
            for data_meta_path in horizontal_meta:
                if data_meta_path == "":
                    continue
                self.horizontal_meta.extend(json.load(open(data_meta_path, "r")))

        if self.vertical_meta is not None:
            for data_meta_path in vertical_meta:
                if data_meta_path == "":
                    continue
                self.vertical_meta.extend(json.load(open(data_meta_path, "r")))

        self.clip_image_processor = CLIPImageProcessor()

        self.drop_ratio = drop_ratio
        self.zero_t_first_frame = True
        self.visualize_dataset = visualize_dataset
        self.broken_txt_path = "./data/broken_kps_videos.txt"

    def get_relative_pose(self, cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        source_cam_c2w = abs_c2ws[0]
        if self.zero_t_first_frame:
            cam_to_origin = 0
        else:
            cam_to_origin = np.linalg.norm(source_cam_c2w[:3, 3])
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
    
    def load_cameras(self, pose_file, img_size):
        with open(pose_file, 'r') as f:
            poses = f.readlines()
        poses = [pose.strip().split(' ') for pose in poses[:]]
        cam_params = [[float(x) for x in pose] for pose in poses]
        cam_params = [Camera(cam_param, pose_file, img_size) for cam_param in cam_params]
        return cam_params
    
    def augmentation(self, images, random_resize_crop, random_scale, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(random_resize_crop(img, random_scale, random_state=state)) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(random_resize_crop(images, random_scale, random_state=state))  # (c, h, w)
        return ret_tensor

    def get_new_index(self, index):
        if index < len(self.horizontal_meta):
            return random.randint(0, len(self.horizontal_meta))
        else:
            return random.randint(len(self.horizontal_meta), len(self.horizontal_meta) + len(self.vertical_meta))
        
    def __getitem__(self, index):
        if index < len(self.horizontal_meta):
            video_meta = self.horizontal_meta[index]
        else:
            video_meta = self.vertical_meta[index - len(self.horizontal_meta)]
        video_path = video_meta["video_path"]
        kps_path = video_meta["kps_path"]
        start_frame = video_meta["start_frame"]
        end_frame = video_meta["end_frame"]


        if not os.path.exists(video_path):
            with open(self.broken_txt_path, "a") as f:
                f.write(f"{video_path}\n")
            return self.__getitem__(self.get_new_index(index))
        
        if not os.path.exists(kps_path):
            with open(self.broken_txt_path, "a") as f:
                f.write(f"{kps_path}\n")
            return self.__getitem__(self.get_new_index(index))
        
        try:
            video_reader = VideoReader(video_path)
        except Exception as e:  # Replace 'Exception' with a more specific exception if possible
            with open(self.broken_txt_path, "a") as f:
                f.write(f"{video_path}\n")
                os.remove(video_path)
            return self.__getitem__(self.get_new_index(index))

        try:
            kps_reader = VideoReader(kps_path)
        except Exception as e:  # Replace 'Exception' with a more specific exception if possible
            with open(self.broken_txt_path, "a") as f:
                f.write(f"{kps_path}\n")
                os.remove(kps_path)
            return self.__getitem__(self.get_new_index(index))

        if len(video_reader) != len(kps_reader):
            # write the broken video path to a file
            with open(self.broken_txt_path, "a") as f:
                f.write(f"{kps_path}\n")
                os.remove(kps_path)
            #print(f"{len(video_reader) = } != {len(kps_reader) = } in {video_path}")
            return self.__getitem__(self.get_new_index(index))

        segment_length = end_frame - start_frame
        video_length = len(video_reader)
        
        max_sample_rate = (segment_length - 1) // (self.n_sample_frames - 1)
        sample_rate = np.random.randint(2, min(max_sample_rate, self.sample_rate) + 1) if max_sample_rate > 1 else 1
        clip_length = (self.n_sample_frames - 1) * sample_rate + 1
        start_idx = random.randint(0, segment_length - clip_length) + start_frame
        end_idx = start_idx + clip_length - 1
        batch_index = np.linspace(start_idx, end_idx, self.n_sample_frames, dtype=int).tolist()

        # read frames and kps
        vid_pil_image_list = []
        pose_pil_image_list = []
        for index in batch_index:
            img = video_reader[index]
            vid_pil_image_list.append(Image.fromarray(img.asnumpy()))
            img = kps_reader[index]
            pose_pil_image_list.append(Image.fromarray(img.asnumpy()))
        
        # sample ref_img_idx
        all_indices = list(range(start_frame, end_frame))
        excluded_indices = set(range(start_idx, end_idx + 1))
        valid_indices = [idx for idx in all_indices if idx not in excluded_indices]
        if valid_indices:
            random_index = random.choice(valid_indices)
        else:
            random_index = random.randint(start_frame, end_frame -1)  # This case occurs only if the exclusion covers all indices, which is unlikely
        ref_img_idx = random_index
        ref_img = Image.fromarray(video_reader[ref_img_idx].asnumpy())

        # Determine image size based on video orientation
        if ref_img.width > ref_img.height:
            img_size = (self.train_size_large, self.train_size_small)
        else:
            img_size = (self.train_size_small, self.train_size_large)

        # transform
        random_scale = random.uniform(*self.img_scale)
        random_resize_crop = RandomResizeCrop(img_size)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        cond_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        state = torch.get_rng_state()
        pixel_values_vid = self.augmentation(
            vid_pil_image_list, random_resize_crop, random_scale, transform, state
        )
        pixel_values_pose = self.augmentation(
            pose_pil_image_list, random_resize_crop, random_scale, cond_transform, state
        )
        pixel_values_ref_img = self.augmentation(ref_img, random_resize_crop, random_scale, transform, state)
        clip_ref_img = self.clip_image_processor(
            images=ref_img, return_tensors="pt"
        ).pixel_values[0]

        # load camera parameters
        camera_file = video_path.replace("/videos/", "/camera/").replace("/mp4/", "/camera/").replace(".mp4", ".txt")
        if not os.path.exists(camera_file):
            with open("./data/broken_kps_videos.txt", "a") as f:
                f.write(f"{camera_file} not exists\n")
            return self.__getitem__(self.get_new_index(index))
        try:
            cam_params = self.load_cameras(camera_file, img_size)
        except Exception as e:  # Replace 'Exception' with a more specific exception if possible
            with open("./data/broken_kps_videos.txt", "a") as f:
                f.write(f"{camera_file} cannot open\n")
                os.remove(camera_file)
            return self.__getitem__(self.get_new_index(index))
        
        if len(cam_params) != video_length:
            with open("./data/broken_kps_videos.txt", "a") as f:
                f.write(f"{video_path} length != camera\n")
                os.remove(video_path)
                os.remove(camera_file)
            return self.__getitem__(self.get_new_index(index))
        
        assert len(cam_params) == video_length, f"len(cam_params) = {len(cam_params)} != video_length = {video_length}"
        cam_params = [cam_params[ref_img_idx]] + [cam_params[tgt_idx]for tgt_idx in batch_index]
        intrinsics = np.asarray([[cam_param.fx * img_size[0],
                                  cam_param.fy * img_size[1],
                                  cam_param.cx * img_size[0],
                                  cam_param.cy * img_size[1]]
                                 for cam_param in cam_params[1:]], dtype=np.float32)

        intrinsics = torch.as_tensor(intrinsics)[None]                  # [1, 1, 4]
        c2w_poses = self.get_relative_pose(cam_params)[1:]              # [frame, 4, 4]
        c2w = torch.as_tensor(c2w_poses)[None]                          # [1, frame, 4, 4]
        flip_flag = torch.zeros(1, dtype=torch.bool, device=c2w.device)
        plucker_embedding = ray_condition(intrinsics, c2w, img_size[1], img_size[0], device='cpu',
                                          flip_flag=flip_flag)[0].permute(0, 3, 1, 2).contiguous()  # [frame, 6, H, W]

        if self.visualize_dataset:
            import cv2
            dir_path = "tmp"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            fps = 12
            vid_name = video_path.split("/")[-1]
            output_path = f"{dir_path}/video_visual_{vid_name}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            size = (pixel_values_ref_img.shape[2] * 4, pixel_values_ref_img.shape[1])
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, size)
            video_length = pixel_values_vid.shape[0]
            # Concatenate images in width dimension
            for i in range(video_length):
                img_visual = np.concatenate([to_image(pixel_values_ref_img, True), 
                                            to_image(pixel_values_vid[i], True), 
                                            to_image(pixel_values_pose[i], False),
                                            pca_visualize(plucker_embedding[i])], axis=1)
                """to_image(plucker_embedding[i, :3,:,:], False),
                to_image(plucker_embedding[i, 3:,:,:], False)]"""
                video_writer.write(img_visual)
            video_writer.release()
            visualizer = CameraPoseVisualizer([-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5])
            visualize_camera_pose(visualizer, c2w_poses, dir_path, vid_name)
        
        sample = dict(
            video_dir=video_path,
            pixel_values_vid=pixel_values_vid,
            pixel_values_pose=pixel_values_pose,
            pixel_values_ref_img=pixel_values_ref_img,
            clip_ref_img=clip_ref_img,
            plucker_embedding=plucker_embedding,
        )

        return sample

    def __len__(self):
        return len(self.horizontal_meta) + len(self.vertical_meta)
    
if __name__ == "__main__":
    dataset = HumanDanceCameraVideoDataset(
        train_size_large=896,
        train_size_small=512,
        img_scale=(1.0, 1.1),
        drop_ratio=0.1,
        horizontal_meta=["./data/json_files/pexels_horizontal_10.0s.json" ],
        vertical_meta=["./data/json_files/pexels-tiktok-ubc_vertical_10.0s.json"],
        n_sample_frames=24,
        sample_rate=6,
        visualize_dataset = True,
    )

    sampler = OrientationSampler(
        horizontal_indices=list(range(len(dataset.horizontal_meta))),
        vertical_indices=list(range(len(dataset.horizontal_meta), len(dataset))),
    )

    batch_sampler = OrientationBatchSampler(sampler, batch_size=2)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
    )

    for i, batch in enumerate(dataloader):
        print(i, batch["pixel_values_vid"].shape, batch["pixel_values_pose"].shape, batch["pixel_values_ref_img"].shape, batch["clip_ref_img"].shape, batch["plucker_embedding"].shape)
        if i > 50:
            break

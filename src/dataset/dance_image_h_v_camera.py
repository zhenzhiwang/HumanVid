import json
import random
import torch
import torchvision.transforms as transforms
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
import os
import numpy as np
import random
import torchvision.transforms.functional as TF
from packaging import version as pver
import torch.distributed as dist
from src.dataset.visualization_utils import CameraPoseVisualizer, visualize_camera_pose, to_image, pca_visualize  # 

class Camera(object):
    def __init__(self, entry, pose_file_name, image_scale=(1920, 1080)):
        assert len(entry) == 10 or len(entry) == 11, f"length of entry should be 11 (extrinsic + fx fy + scale) or 10 (+ fx fy), got {len(entry)}"
        if image_scale[0] > image_scale[1]:
            self.fx = entry[8]
            self.fy = self.fx * (image_scale[0] / image_scale[1])
            self.cx = 0.5
            self.cy = 0.5
        else:
            self.fy = entry[9]
            self.fx = self.fy * (image_scale[1] / image_scale[0])
            self.cx = 0.5
            self.cy = 0.5
        # Read the timestamp and pose information from the entry
        self.timestamp = entry[0]
        tx, ty, tz = entry[1:4]
        qx, qy, qz, qw = entry[4:8]
        scale = entry[10] if len(entry) == 11 else 1.0
        # normalize quaternion
        norm = np.linalg.norm([qx, qy, qz, qw])
        if np.abs(norm - 1) > 1e-3:
            with open("./data/broken_kps_videos.txt", "a") as f:
                f.write(f"{pose_file_name}'s quaternion is not well normalized! \n")
        qx, qy, qz, qw = [x / norm for x in [qx, qy, qz, qw]]

        
        # Convert quaternion to rotation matrix
        rotation_matrix = self.quaternion_to_rotation_matrix(qx, qy, qz, qw)
        
        # Create the translation vector
        translation_vector = np.array([tx, ty, tz])
        
        if "bedlam" in pose_file_name or "blender" in pose_file_name or "ue_rendered" in pose_file_name:
            # Create the world-to-camera transformation matrix
            self.w2c_mat = np.eye(4)
            self.w2c_mat[:3, :3] = rotation_matrix
            self.w2c_mat[:3, 3] = translation_vector
            
            # Invert the matrix to get the camera-to-world transformation matrix
            self.c2w_mat = np.linalg.inv(self.w2c_mat)
        elif "pexels" in pose_file_name or "inference" in pose_file_name or "ubc" in pose_file_name or "tiktok" in pose_file_name or "webvid" in pose_file_name or "test" in pose_file_name:
            # Create the camera-to-world transformation matrix
            self.c2w_mat = np.eye(4)
            self.c2w_mat[:3, :3] = rotation_matrix
            self.c2w_mat[:3, 3] = translation_vector * scale
            
            # Invert the matrix to get the world-to-camera transformation matrix
            self.w2c_mat = np.linalg.inv(self.c2w_mat)
        else:
            raise ValueError(f"Unknown camera pose dataset name: {pose_file_name}")
        
    @staticmethod
    def quaternion_to_rotation_matrix(qx, qy, qz, qw):
        # Convert a quaternion to a rotation matrix
        # Using the formula from https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw,         1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw,         2*qy*qz + 2*qx*qw,     1 - 2*qx**2 - 2*qy**2]
        ])
        return R


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def ray_condition(K, c2w, H, W, device, flip_flag=None):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B, V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype)
        )
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    fx, fy, cx, cy = K.chunk(4, dim=-1)     # B,V, 1

    zs = torch.ones_like(i)                 # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)              # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)             # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)        # B, V, HW, 3
    rays_o = c2w[..., :3, 3]                                        # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)                   # B, V, HW, 3
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)                          # B, V, HW, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)             # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker

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

class HumanDanceCameraDataset(Dataset):
    def __init__(
        self,
        train_size_large,
        train_size_small,
        img_scale=(1.0, 1.1),
        drop_ratio=0.1,
        horizontal_meta=["./data/pexels-human_horizontal_meta.json"],
        vertical_meta=["./data/pexels-human_vertical_meta.json"],
        sample_margin=30,
        visualize_dataset = False,
    ):
        super().__init__()

        self.train_size_large = train_size_large
        self.train_size_small = train_size_small
        self.img_scale = img_scale
        self.sample_margin = sample_margin
        self.visualize_dataset = visualize_dataset

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
    
    def get_new_index(self, index):
        if index < len(self.horizontal_meta):
            return random.randint(0, len(self.horizontal_meta))
        else:
            return random.randint(len(self.horizontal_meta), len(self.horizontal_meta) + len(self.vertical_meta))
        
    def __getitem__(self, index):
        if index < len(self.horizontal_meta):
            video_meta = self.horizontal_meta[index]
        else:
            # process out of range index
            if index >= len(self.horizontal_meta) + len(self.vertical_meta) or index < 0:
                with open("./data/broken_kps_videos.txt", "a") as f:
                    f.write(f"index {index} out of range\n")
                index = self.get_new_index(index)
            video_meta = self.vertical_meta[index - len(self.horizontal_meta)]
        
        video_path = video_meta["video_path"]
        kps_path = video_meta["kps_path"]
        start_frame = video_meta["start_frame"]
        end_frame = video_meta["end_frame"]
        
        if not os.path.exists(video_path):
            with open("./data/broken_kps_videos.txt", "a") as f:
                f.write(f"{video_path} not exists\n")
            return self.__getitem__(self.get_new_index(index))
        
        if not os.path.exists(kps_path):
            with open("./data/broken_kps_videos.txt", "a") as f:
                f.write(f"{kps_path} not exists\n")
            return self.__getitem__(self.get_new_index(index))
        
        try:
            video_reader = VideoReader(video_path)
        except Exception as e:  # Replace 'Exception' with a more specific exception if possible
            with open("./data/broken_kps_videos.txt", "a") as f:
                f.write(f"{video_path} broken\n")
                os.remove(video_path)
            return self.__getitem__(self.get_new_index(index))

        try:
            kps_reader = VideoReader(kps_path)
        except Exception as e:  # Replace 'Exception' with a more specific exception if possible
            with open("./data/broken_kps_videos.txt", "a") as f:
                f.write(f"{kps_path} broken\n")
                os.remove(kps_path)
            return self.__getitem__(self.get_new_index(index))

        if len(video_reader) != len(kps_reader):
            # write the broken video path to a file
            with open("./data/broken_kps_videos.txt", "a") as f:
                f.write(f"{kps_path}  len != video\n")
                os.remove(kps_path)
            #print(f"{len(video_reader) = } != {len(kps_reader) = } in {video_path}")
            return self.__getitem__(self.get_new_index(index))
        
        segment_length = end_frame - start_frame
        video_length = len(video_reader)
        ref_img_idx = random.randint(start_frame, end_frame - 1)
        margin = min(self.sample_margin, segment_length)

        start_exclude = max(start_frame, ref_img_idx - margin)
        end_exclude = min(end_frame - 1, ref_img_idx + margin)
        valid_indices = [i for i in range(start_frame, end_frame) if not (start_exclude <= i <= end_exclude)]
        
        if valid_indices:
            tgt_img_idx = random.choice(valid_indices)
        else:
            tgt_img_idx = random.randint(start_frame, end_frame - 1)

        ref_img = video_reader[ref_img_idx]
        ref_img_pil = Image.fromarray(ref_img.asnumpy())
        tgt_img = video_reader[tgt_img_idx]
        tgt_img_pil = Image.fromarray(tgt_img.asnumpy())

        tgt_pose = kps_reader[tgt_img_idx]
        tgt_pose_pil = Image.fromarray(tgt_pose.asnumpy())

        # Determine image size based on video orientation
        if tgt_img_pil.width > tgt_img_pil.height:
            img_size = (self.train_size_large, self.train_size_small)
        else:
            img_size = (self.train_size_small, self.train_size_large)


        random_scale = random.uniform(*self.img_scale)
        random_resize_crop = RandomResizeCrop(img_size)

        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5], [0.5]),])
        cond_transform = transforms.Compose([transforms.ToTensor(),])

        state = torch.get_rng_state()
        tgt_img = transform(random_resize_crop(tgt_img_pil, random_scale, random_state=state))
        ref_img_vae = transform(random_resize_crop(ref_img_pil, random_scale, random_state=state))
        tgt_pose_img = cond_transform(random_resize_crop(tgt_pose_pil, random_scale, random_state=state))


        clip_image = self.clip_image_processor(images=ref_img_pil, return_tensors="pt").pixel_values[0]

        # load camera parameters
        camera_file = video_path.replace("/videos/", "/camera/").replace("/mp4/", "/camera/").replace(".mp4", ".txt")
        if not os.path.exists(camera_file):
            with open("./data/broken_kps_videos.txt", "a") as f:
                f.write(f"{camera_file} not exists\n")
            return self.__getitem__(self.get_new_index(index))
        try:
            cam_params = self.load_cameras(camera_file, img_size)
        except Exception as e:  # Replace 'Exception' with a more specific exception if possible
            print(e)
            with open("./data/broken_kps_videos.txt", "a") as f:
                f.write(f"{camera_file} error in load_cameras\n")
                #os.remove(camera_file)
            return self.__getitem__(self.get_new_index(index))
        
        if len(cam_params) != video_length and "blender" in video_path:
            with open("./data/broken_kps_videos.txt", "a") as f:
                f.write(f"{video_path} length != camera\n")
                os.remove(video_path)
                os.remove(camera_file)
            return self.__getitem__(self.get_new_index(index))
        #assert len(cam_params) == video_length, f"{camera_file} :{len(cam_params) = } != {video_length = }"
        cam_params = [cam_params[ref_img_idx], cam_params[tgt_img_idx]]
        intrinsics = np.asarray([[cam_param.fx * img_size[0],
                                  cam_param.fy * img_size[1],
                                  cam_param.cx * img_size[0],
                                  cam_param.cy * img_size[1]]
                                 for cam_param in cam_params[1:]], dtype=np.float32)

        intrinsics = torch.as_tensor(intrinsics)[None]                  # [1, 1, 4]
        c2w_poses = self.get_relative_pose(cam_params)[1:]              # [1, 4, 4]
        c2w = torch.as_tensor(c2w_poses)[None]                          # [1, 1, 4, 4]
        flip_flag = torch.zeros(1, dtype=torch.bool, device=c2w.device)
        plucker_embedding = ray_condition(intrinsics, c2w, img_size[1], img_size[0], device='cpu',
                                          flip_flag=flip_flag)[0].permute(0, 3, 1, 2).contiguous()[0]  # [6, H, W]
        #print(f"plucker_embedding shape = {plucker_embedding.shape}")
        if self.visualize_dataset:
            dir_path = "tmp"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            
            # Concatenate images in width dimension
            img_visual = np.concatenate([to_image(ref_img_vae, True), 
                                        to_image(tgt_img, True), 
                                        to_image(tgt_pose_img, False),
                                        pca_visualize(plucker_embedding)], axis=1)
            
            # Create and save image
            img = Image.fromarray(img_visual)
            img.save(f"{dir_path}/img_visual_{video_path.split('/')[-1]}.png")
            visualizer = CameraPoseVisualizer([-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5])
            visualize_camera_pose(visualizer, c2w_poses, dir_path, video_path.split("/")[-1])

        sample = dict(
            video_dir=video_path,
            tgt_img=tgt_img,
            tgt_pose=tgt_pose_img,
            ref_img=ref_img_vae,
            clip_images=clip_image,
            plucker_embedding=plucker_embedding,
        )
        return sample

    def __len__(self):
        return len(self.horizontal_meta) + len(self.vertical_meta)

class OrientationSampler(torch.utils.data.Sampler):
    def __init__(self, horizontal_indices, vertical_indices):
        self.horizontal_indices = horizontal_indices
        self.vertical_indices = vertical_indices

    def __iter__(self):
        indices = self.horizontal_indices + self.vertical_indices
        return iter(indices)

    def __len__(self):
        return len(self.horizontal_indices) + len(self.vertical_indices)
    
class OrientationBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, sampler, batch_size, rank=1, seed=42):
        self.sampler = sampler
        self.batch_size = batch_size
        self.horizontal_indices = self.sampler.horizontal_indices
        self.vertical_indices = self.sampler.vertical_indices
        self.epoch = 0
        self.seed = seed
        self.rank = rank

    def __iter__(self):
        random.seed(self.seed + self.epoch + self.rank)
        indices = self.horizontal_indices + self.vertical_indices
        horizontal_indices = self.horizontal_indices.copy()
        vertical_indices = self.vertical_indices.copy()
        random.shuffle(horizontal_indices)
        random.shuffle(vertical_indices)

        while len(indices) > self.batch_size:
            # Randomly choose between horizontal and vertical indices for the current batch
            if random.random() < 0.5 and len(horizontal_indices) >= self.batch_size:
                batch_indices = horizontal_indices[:self.batch_size]
                horizontal_indices = horizontal_indices[self.batch_size:]
            elif len(vertical_indices) >= self.batch_size:
                batch_indices = vertical_indices[:self.batch_size]
                vertical_indices = vertical_indices[self.batch_size:]
            else:
                # Not enough samples left for a full batch, move to next epoch
                horizontal_indices = []
                vertical_indices = []
                break

            yield batch_indices
            indices = horizontal_indices + vertical_indices

        # Reset horizontal_indices and vertical_indices for the next epoch
        self.horizontal_indices = self.sampler.horizontal_indices
        self.vertical_indices = self.sampler.vertical_indices

    def __len__(self):
        return (len(self.sampler.horizontal_indices) + len(self.sampler.vertical_indices)) // self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch

class DistributedOrientationBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, sampler, batch_size, num_replicas=None, rank=None, seed=42):
        self.sampler = sampler
        self.batch_size = batch_size
        self.horizontal_indices = self.sampler.horizontal_indices
        self.vertical_indices = self.sampler.vertical_indices
        
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        indices = self.horizontal_indices + self.vertical_indices
        n_samples = len(indices)
        
        # Ensure all processes have the same number of samples
        n_samples_per_replica = n_samples // self.num_replicas
        total_size = n_samples_per_replica * self.num_replicas
        
        # Get indices for this replica
        indices = indices[:total_size]  # Trim to ensure equal division
        indices = indices[self.rank:total_size:self.num_replicas]
        
        # Shuffle indices
        indices = [indices[i] for i in torch.randperm(len(indices), generator=g)]
        
        # Separate horizontal and vertical indices
        horizontal_indices = [i for i in indices if i in self.horizontal_indices]
        vertical_indices = [i for i in indices if i in self.vertical_indices]
        
        # Generate batches
        while len(horizontal_indices) + len(vertical_indices) >= self.batch_size:
            if len(horizontal_indices) >= self.batch_size and (len(vertical_indices) < self.batch_size or random.random() < 0.5):
                batch = horizontal_indices[:self.batch_size]
                horizontal_indices = horizontal_indices[self.batch_size:]
            else:
                batch = vertical_indices[:self.batch_size]
                vertical_indices = vertical_indices[self.batch_size:]
            yield batch

    def __len__(self):
        return len(self.sampler) // self.batch_size // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch  

if __name__ == "__main__":
    dataset = HumanDanceCameraDataset(
        train_size_large=896,
        train_size_small=512,
        img_scale=(1.0, 1.1),
        drop_ratio=0.1,
        horizontal_meta=["./data/json_files/pexels_horizontal_10.0s.json" ],  # ,   "./data/pexels-human_horizontal_meta.json"
        vertical_meta=["./data/json_files/pexels-tiktok-ubc_vertical_10.0s.json"],  # ,   "./data/pexels-human_vertical_meta.json"
        sample_margin=30,
        visualize_dataset = True,
    )
    print(len(dataset))
    print(len(dataset.horizontal_meta))
    print(len(dataset.vertical_meta))

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
    from tqdm import tqdm
    for i, batch in tqdm(enumerate(dataloader)):
        #print(i, batch["img"].shape, batch["tgt_pose"].shape, batch["ref_img"].shape, batch["clip_images"].shape, batch["plucker_embedding"].shape)
        if i > 5:
            break
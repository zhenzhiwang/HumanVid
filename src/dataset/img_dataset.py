import torch
import numpy as np

class Camera(object):
    def __init__(self, entry, pose_file_name, image_scale=(1920, 1080)):
        assert len(entry) == 8 or len(entry) == 11, f"length of entry should be 11 (extrinsic + fx fy + scale) or 8 (extrinsic), got {len(entry)}"
        if image_scale[0] > image_scale[1]:
            self.fx = 0.688 if len(entry) == 8 else entry[8]
            self.fy = self.fx * (image_scale[0] / image_scale[1])
            self.cx = 0.5
            self.cy = 0.5
        else:
            self.fy = 0.688 if len(entry) == 8 else entry[9]
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
        
        if "synthetic" in pose_file_name:
            # Create the world-to-camera transformation matrix
            self.w2c_mat = np.eye(4)
            self.w2c_mat[:3, :3] = rotation_matrix
            self.w2c_mat[:3, 3] = translation_vector
            
            # Invert the matrix to get the camera-to-world transformation matrix
            self.c2w_mat = np.linalg.inv(self.w2c_mat)

        elif "pexels" in pose_file_name:
            # Create the camera-to-world transformation matrix
            self.c2w_mat = np.eye(4)
            self.c2w_mat[:3, :3] = rotation_matrix
            self.c2w_mat[:3, 3] = translation_vector
            
            # Invert the matrix to get the world-to-camera transformation matrix
            self.w2c_mat = np.linalg.inv(self.c2w_mat)
        else:
            raise ValueError(f"Unknown camera pose file name: {pose_file_name}")
        
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


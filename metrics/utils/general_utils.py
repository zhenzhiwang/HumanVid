#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def quaternion_multiply(A, B):
    """
    Apply quaternion A to a batch of quaternions B.
    A should be of shape (4,) and B should be of shape (N, 4).
    """
    # Ensure A and B have the correct shapes
    # assert A.shape == (4,)
    # assert B.dim() == 2 and B.shape[1] == 4

    # Expand A to match the batch size of B
    # A = A.expand(B.shape[0], 4)

    # Unpack the components of the quaternions
    w1, x1, y1, z1 = A[:, 0], A[:, 1], A[:, 2], A[:, 3]
    w2, x2, y2, z2 = B[:, 0], B[:, 1], B[:, 2], B[:, 3]

    # Perform the quaternion multiplication for each pair
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    # Stack the components together to form new quaternions
    quat = torch.stack([w, x, y, z], dim=-1)
    
    return quat

def build_qvec(r):
    Rxx=r[:, 0, 0]
    Ryx=r[:, 0, 1]
    Rzx=r[:, 0, 2]
    Rxy=r[:, 1, 0]
    Ryy=r[:, 1, 1]
    Rzy=r[:, 1, 2]
    Rxz=r[:, 2, 0]
    Ryz=r[:, 2, 1]
    Rzz=r[:, 2, 2]

    K = torch.zeros((r.size(0), 4, 4), device='cuda')
    # print(K.shape)

    K[:, 0, 0] = Rxx - Ryy - Rzz
    K[:, 1, 0] = Ryx + Rxy
    K[:, 1, 1] = Ryy - Rxx - Rzz
    K[:, 2, 0] = Rzx + Rxz
    K[:, 2, 1] = Rzy + Ryz
    K[:, 2, 2] = Rzz - Rxx - Ryy
    K[:, 3, 0] = Ryz - Rzy
    K[:, 3, 1] = Rzx - Rxz
    K[:, 3, 2] = Rxy - Ryx
    K[:, 3, 3] = Rxx + Ryy + Rzz
    K = K / 3.0
    
    eigvals, eigvecs = torch.linalg.eigh(K)

    assert(torch.all(torch.argmax(eigvals, dim=1)==3))
    qvec = eigvecs[:, [3, 0, 1, 2], 3]
    qvec[qvec[:,0]<0]*=-1

    return qvec

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

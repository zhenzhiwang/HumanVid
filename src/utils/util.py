import importlib
import os
import os.path as osp
import shutil
import sys
from pathlib import Path

import av
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict

def save_checkpoint(model, save_dir, prefix, ckpt_num, total_limit=None, logger=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0]))

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            if logger is not None:
                logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    state_dict = model.state_dict()
    if prefix == "motion_module":
        mm_state_dict = OrderedDict()
        for key in state_dict:
            if "motion_module" in key:
                mm_state_dict[key] = state_dict[key]
        torch.save(mm_state_dict, save_path)
    else:
        torch.save(state_dict, save_path)


def seed_everything(seed):
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


def import_filename(filename):
    spec = importlib.util.spec_from_file_location("mymodule", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def delete_additional_ckpt(base_path, num_keep):
    dirs = []
    for d in os.listdir(base_path):
        if d.startswith("checkpoint-"):
            dirs.append(d)
    num_tot = len(dirs)
    if num_tot <= num_keep:
        return
    # ensure ckpt is sorted and delete the ealier!
    del_dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))[: num_tot - num_keep]
    for d in del_dirs:
        path_to_dir = osp.join(base_path, d)
        if osp.exists(path_to_dir):
            shutil.rmtree(path_to_dir)


def save_videos_from_pil(pil_images, path, fps=8, bitrate="5000k", crf=19, preset="slow"):
    import av

    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width, height = pil_images[0].size

    if save_fmt == ".mp4":
        codec = "libx264"
        container = av.open(path, "w")
        stream = container.add_stream(codec, rate=fps)

        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"  # 设置像素格式

        # 设置编码器参数
        stream.options = {
            "b": bitrate,  # 比特率
            "crf": str(crf),  # 恒定质量参数
            "preset": preset  # 编码速度与质量的平衡
        }
        for pil_image in pil_images:
            # pil_image = Image.fromarray(image_arr).convert("RGB")
            av_frame = av.VideoFrame.from_image(pil_image)
            container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
        container.close()

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
    videos = videos.permute(2, 0, 1, 3, 4)  # rearrange to (t, b, c, h, w)
    height, width = videos.shape[-2:]
    outputs = []

    # Determine the layout based on the aspect ratio
    aspect_ratio = width / height
    if aspect_ratio > 1:
        # Landscape orientation: more columns than rows
        n_rows = 2

    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)  # (c, h, w)
        x = x.permute(1, 2, 0)  # (h, w, c)
        if rescale:
            x = (x + 1.0) / 2.0  # Scale -1,1 to 0,1
        x = (x * 255).numpy().astype(np.uint8)
        x = Image.fromarray(x)

        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_videos_from_pil(outputs, path, fps)

def save_image_grid(images: torch.Tensor, path: str, n_rows=6):
    height, width = images.shape[-2:]
    # Determine the layout based on the aspect ratio
    aspect_ratio = width / height
    if aspect_ratio > 1:
        # Landscape orientation: more columns than rows
        n_rows = 2

    images = torchvision.utils.make_grid(images, nrow=n_rows)  # (c h w)
    images = images.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
    images = (images * 255).numpy().astype(np.uint8)
    images = Image.fromarray(images)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    images.save(path)

def show_image_grid(images: torch.Tensor, n_rows=6):
    images = torchvision.utils.make_grid(images, nrow=n_rows)  # (c h w)
    images = images.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
    images = (images * 255).numpy().astype(np.uint8)
    images = Image.fromarray(images)
    plt.imshow(images)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()


def read_frames(video_path):
    container = av.open(video_path)

    video_stream = next(s for s in container.streams if s.type == "video")
    frames = []
    for packet in container.demux(video_stream):
        for frame in packet.decode():
            image = Image.frombytes(
                "RGB",
                (frame.width, frame.height),
                frame.to_rgb().to_ndarray(),
            )
            frames.append(image)

    return frames


def get_fps(video_path):
    container = av.open(video_path)
    video_stream = next(s for s in container.streams if s.type == "video")
    fps = video_stream.average_rate
    container.close()
    return fps

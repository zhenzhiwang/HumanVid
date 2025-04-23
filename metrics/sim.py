import os
import argparse
from pathlib import Path
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from einops import rearrange
import glob
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch.nn.functional as F
import av

model_path = "/mnt/afs/wangzhenzhi/code/consistent-animate/pretrained_models/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_path).cuda()
processor = CLIPProcessor.from_pretrained(model_path)

def read_frames(video_path):
    video_path = str(video_path)  # Ensure video_path is a string
    #print(f"Reading video: {video_path}")  # Debugging statement
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

@torch.no_grad()
def get_image_features_from_video(video_path):
    frames = read_frames(video_path)
    frames = [frame.resize((512, 512)) for frame in frames]

    image_features = []
    for frame in frames:
        inputs = processor(images=frame, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        image_feature = model.get_image_features(**inputs)
        image_feature = image_feature / image_feature.norm(p=2, dim=-1, keepdim=True)
        image_features.append(image_feature)

    image_features = torch.cat(image_features)
    return image_features

@torch.no_grad()
def main(dir):
    result_dir = Path(dir)
    video_paths = sorted(list(result_dir.glob("*.mp4")))

    if not video_paths:
        print("No video files found in the specified directory.")
        return

    scores = []
    for video_path in tqdm(video_paths):
        #print(f"Processing video: {video_path}")
        image_features = get_image_features_from_video(video_path)
        image_features = F.normalize(image_features, dim=-1, p=2)

        score_matrix = torch.matmul(image_features, image_features.t())  # (f, f)
        score_mask = torch.triu(torch.ones_like(score_matrix), diagonal=1).to(score_matrix.device)

        masked_matrix = score_mask * score_matrix
        score = masked_matrix.sum() / score_mask.sum()

        scores.append(score.item())

    print(scores)
    print(f"dir: {dir.split('/')[-2]}/{(dir.split('/')[-1]).split("_")[0]}")
    print(f"result: {sum(scores) / len(scores)}, number: {len(scores)}")

if __name__ == "__main__":
    dirs = [#"/mnt/afs/wangzhenzhi/code/animate-with-camera/output/stage2-sense-evaluation-real/pexels-test-h_20240612_2253--seed_42-512x896",
            "/mnt/afs/wangzhenzhi/code/animate-with-camera/output/stage2-sense-evaluation-all/pexels-test-v_20240612_2213--seed_42-512x896",
            "/mnt/afs/wangzhenzhi/code/animate-with-camera/output/stage2-sense-evaluation-real/pexels-test-v_20240612_2213--seed_42-512x896"]
    for dir in dirs:
        main(dir=dir)
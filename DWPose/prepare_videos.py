import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from dwpose_utils import DWposeDetector
from decord import VideoReader
from decord import cpu

def process_video(dwprocessor, video_path, output_video_path, detect_resolution):
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()

    first_frame = vr[0].asnumpy()
    height, width, _ = first_frame.shape
    size = (width, height)
    
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for idx in tqdm(range(len(vr)), desc=f"Processing {os.path.basename(video_path)}"):
        frame = vr[idx].asnumpy()
        detected_pose = process(dwprocessor, frame, detect_resolution)
        #import ipdb; ipdb.set_trace()
        video_writer.write(detected_pose[0])
    video_writer.release()

def process(dwprocessor, input_image, detect_resolution):
    if not isinstance(dwprocessor, DWposeDetector):
        dwprocessor = DWposeDetector()

    with torch.no_grad():
        detected_map = dwprocessor(input_image, num_people = -1)
    return detected_map

dwprocessor = DWposeDetector()
dataset_folder = './videos/'
output_dir = dataset_folder.replace('/videos/', '/dwpose/')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
detect_resolution = 768

path = dataset_folder

for video_name in tqdm(os.listdir(path), desc=f"Processing TikTok"):
    if not video_name.endswith(".mp4"):
        continue
    if video_name.endswith("dwpose.mp4") or video_name.endswith("densepose.mp4"):
        continue
    video_path = os.path.join(path, video_name)
    output_video_path = video_path.replace('/videos/', '/dwpose/')
    #output_video_path = os.path.join(path, video_name.split('_')[0] + '_dwpose.mp4')
    if os.path.exists(output_video_path):
        continue
    process_video(dwprocessor, video_path, output_video_path, detect_resolution)
    # check the length of the output video == input video
    vr = VideoReader(output_video_path, ctx=cpu(0))
    if len(vr) != len(VideoReader(video_path, ctx=cpu(0))):
        print(f"Error processing {video_path}")
        os.remove(output_video_path)
        continue


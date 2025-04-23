import argparse
import json
import os
from tqdm import tqdm
import cv2
dataset_names = ['tiktok', 'ubc']
for dataset in dataset_names:
    meta_infos = []
    current_video_dir = f"./data/{dataset}/train/videos"
    current_pose_dir = f"./data/{dataset}/train/dwpose"
    # collect all video_folder paths
    video_mp4_paths = set()
    if os.path.exists(current_video_dir):
        for root, dirs, files in os.walk(current_video_dir):
            for name in files:
                if name.endswith(".mp4"):
                    video_mp4_paths.add(os.path.join(root, name))
        video_mp4_paths = list(video_mp4_paths)

        for video_mp4_path in video_mp4_paths:
            relative_video_name = os.path.relpath(video_mp4_path, current_video_dir)
            video_id = relative_video_name.split("/")[-1].split(".")[0]

            # dwpose exists and file not broken
            kps_path = os.path.join(current_pose_dir, relative_video_name)
            camera_file = video_mp4_path.replace("/videos/", "/camera/").replace(".mp4", ".txt")
    
            if os.path.exists(kps_path) and os.path.exists(camera_file) and os.path.getsize(kps_path) > 0:
                # read the video_mp4_path and decide whether it is horizontal or vertical
                #orientation = determine_video_orientation(video_mp4_path)
                meta_infos.append({"video_path": video_mp4_path, "kps_path": kps_path})

    print(f"Collected {len(meta_infos)} v videos from {dataset}.")
    # Save the collected metadata to a JSON file
    os.makedirs("./data", exist_ok=True)
    json.dump(meta_infos, open(f"./data/{dataset}_vertical_meta.json", "w"))

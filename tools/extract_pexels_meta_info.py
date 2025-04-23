import argparse
import json
import os
from tqdm import tqdm
import cv2
    
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default='pexels')
parser.add_argument("--meta_info_name", type=str, default='pexels')
parser.add_argument("--start_id", type=int, default=0,  help="The starting ID for the directory range.")
parser.add_argument("--end_id", type=int, default=9, help="The ending ID for the directory range.")

args = parser.parse_args()

if args.meta_info_name is None:
    args.meta_info_name = args.dataset_name

meta_infos = {'h':[], 'v':[], 'h_count':0, 'v_count':0}
for orientation in ["h", "v"]:
    # pose root is to replace the last dir of root-path to be dwpose
    root_path = f"./data/pexels-my-{orientation}/videos"
    pose_root_dir = f"./data/pexels-my-{orientation}/dwpose"
    csv_root_dir = f"./data/pexels-csv/"
    # iterate over the specified range of directories
    for dir_id in tqdm(range(args.start_id, args.end_id + 1), desc="Processing directories"):
        formatted_dir_id = f"{str(dir_id)}"
        current_video_dir = os.path.join(root_path, formatted_dir_id)
        current_pose_dir = os.path.join(pose_root_dir, formatted_dir_id)
        csv_file = os.path.join(csv_root_dir, f"pexels-my-{formatted_dir_id}.csv")
        # collect all video_folder paths
        video_mp4_paths = set()
        if os.path.exists(current_video_dir):
            for root, dirs, files in os.walk(current_video_dir):
                for name in files:
                    if name.endswith(".mp4"):
                        video_mp4_paths.add(os.path.join(root, name))
            video_mp4_paths = list(video_mp4_paths)

            for video_mp4_path in video_mp4_paths:
                num_humans = 0
                relative_video_name = os.path.relpath(video_mp4_path, current_video_dir)
                video_id = relative_video_name.split("/")[-1].split(".")[0]
                with open(csv_file, 'r') as f:
                    lines = f.readlines()
                    flag = True
                    visited = False
                    for line in lines:
                        if line.startswith(os.path.basename(video_mp4_path)):
                            visited = True
                            value = line.strip().split(",")
                            if "False" in value:
                                flag = False
                                break
                            # if the (value[3]'s round value - value[3]) > 1e-2, continue
                            num_humans = int(float(value[3]))
                            if abs(round(float(value[3])) - float(value[3])) > 2.5e-2:
                                flag = False
                                break
                                
                    if visited == False:
                        print("No csv record for video {}!".format(video_mp4_path))
                    if flag == False:
                        continue
                # dwpose exists and file not broken
                kps_path = os.path.join(current_pose_dir, relative_video_name)
                camera_file = video_mp4_path.replace("/videos/", "/camera/").replace(".mp4", ".txt")
        
                if os.path.exists(kps_path) and os.path.exists(camera_file) and os.path.getsize(kps_path) > 0:
                    # read the video_mp4_path and decide whether it is horizontal or vertical
                    #orientation = determine_video_orientation(video_mp4_path)
                    meta_infos[orientation].append({"video_path": video_mp4_path, "kps_path": kps_path})
                    meta_infos[f"{orientation}_count"] += num_humans


print(f"Collected {len(meta_infos['h'])} h videos, {len(meta_infos['v'])} v videos.")
print(f"Collected {meta_infos['h_count']} h humans, {meta_infos['v_count']} v humans.")
# Save the collected metadata to a JSON file
os.makedirs("./data", exist_ok=True)
json.dump(meta_infos['h'], open(f"./data/json_files/{args.meta_info_name}_horizontal.json", "w"))
json.dump(meta_infos['v'], open(f"./data/json_files/{args.meta_info_name}_vertical.json", "w"))

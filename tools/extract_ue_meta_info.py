import argparse
import json
import os
from tqdm import tqdm
import cv2
from multiprocessing import Pool

def determine_video_orientation(video_mp4_path):
    # Try to open the video file
    cap = cv2.VideoCapture(video_mp4_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Read the first frame to get the video dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video.")
        cap.release()
        return
    
    # Get the dimensions of the frame
    height, width = frame.shape[:2]
    
    # Release the video capture object
    cap.release()
    
    # Check the orientation based on width and height
    if width > height:
        return "h"
    else:
        return "v"


def process_video(video_mp4_path):
    relative_video_name = os.path.relpath(video_mp4_path, current_video_dir)
    #video_id = relative_video_name.split("/")[-1].split(".")[0]
    # dwpose exists and file not broken
    kps_path = video_mp4_path.replace("/mp4/", "/dwpose/")
    camera_file = video_mp4_path.replace("/mp4/", "/camera/").replace(".mp4", ".txt")
    # check length == camera file
    cap = cv2.VideoCapture(video_mp4_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    cap = cv2.VideoCapture(kps_path)
    kps_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not os.path.exists(kps_path):
        return None, f"Keypoints file does not exist: {kps_path}"
    if not os.path.exists(camera_file):
        return None, f"Camera file does not exist: {camera_file}"
    
    with open(camera_file, "r") as f:
        camera_lines = f.readlines()
    
    if len(camera_lines) != video_length:
        return None, f"Length of camera file does not match video length: {video_mp4_path}"
    if video_length != kps_length:
        return None, f"Length of kps file does not match video length: {video_mp4_path}"
    
    if os.path.getsize(kps_path) > 0:
        return {"video_path": video_mp4_path, "kps_path": kps_path}, None
    
    return None, None

def process_videos(video_mp4_paths):
    num_processes = 10  # Use the number of CPU cores
    with Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(process_video, video_mp4_paths), total=len(video_mp4_paths)))
    
    meta_infos = []
    for result, error_message in results:
        if result:
            meta_infos.append(result)
        elif error_message:
            print(error_message)
    
    return meta_infos
    
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default='ue')
parser.add_argument("--meta_info_name", type=str, default='ue')
args = parser.parse_args()

if args.meta_info_name is None:
    args.meta_info_name = args.dataset_name
# read txt
txt_path = "./data/ue_rendered/training_video/video_set_name.txt"
with open(txt_path, 'r') as f:
    video_set_names = f.readlines()
video_set_names = [x.strip() for x in video_set_names]
meta_infos = []

for idx in range(len(video_set_names)):
    # pose root is to replace the last dir of root-path to be dwpose
    current_video_dir = f"./data/ue_rendered/training_video/{video_set_names[idx]}/mp4"
    current_pose_dir = f"./data/ue_rendered/training_video/{video_set_names[idx]}/dwpose"
    csv_root_dir = f"./data/ue_rendered/csv/"
    csv_file = os.path.join(csv_root_dir, f"ue_rendered-{video_set_names[idx]}.csv")
    # collect all video_folder paths
    video_mp4_paths = set()
    if os.path.exists(current_video_dir):
        for root, dirs, files in os.walk(current_video_dir):
            for name in files:
                if name.endswith(".mp4"):
                    video_mp4_paths.add(os.path.join(root, name))
        video_mp4_paths = list(video_mp4_paths)
        meta_infos.extend(process_videos(video_mp4_paths))

print(f"Collected {len(meta_infos)} h videos.")
# Save the collected metadata to a JSON file
os.makedirs("./data", exist_ok=True)
json.dump(meta_infos, open(f"./data/json_files/{args.meta_info_name}_horizontal_meta.json", "w"))
#json.dump(meta_infos['v'], open(f"./data/{args.meta_info_name}_vertical_meta.json", "w"))

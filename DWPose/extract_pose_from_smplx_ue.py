import os
import cv2
from tqdm import tqdm
import argparse
from dwpose_utils.smplx2openpose import visualizeVideo
import multiprocessing as mp

def process_video(args):
    video_name, input_video_dir, output_video_dir, npz_dir = args
    if not video_name.endswith(".mp4"):
        return
    video_path = os.path.join(input_video_dir, video_name)
    output_video_path = os.path.join(output_video_dir, video_name)
    if not os.path.exists(video_path):
        return
    if os.path.exists(output_video_path):
        # test readable
        cap = cv2.VideoCapture(output_video_path)
        if not cap.isOpened():
            os.remove(output_video_path)
        cap.release()
    if os.path.exists(output_video_path):
        return
        
    npz_file = os.path.join(npz_dir, video_name.replace('.mp4', '.npz'))
    if not os.path.exists(npz_file):
        return
    visualizeVideo(npz_file, video_path, output_video_path, visualize_over_original_video=False)


def main(): 
    parser = argparse.ArgumentParser(description='dwpose.')
    parser.add_argument('--id', type=int, default=0, help='start id of video file name')
    parser.add_argument('--id_end', type=int, default=10, help='end id of video file name')
    parser.add_argument('--num_processes', type=int, default=10, help='number of processes to use')
    args = parser.parse_args()

    dataset_name = "ue_rendered"
    txt_path = f"../data/{dataset_name}/training_video/video_set_name.txt"
    with open(txt_path, 'r') as f:
        video_set_names = f.readlines()
    video_set_names = [x.strip() for x in video_set_names]

    for idx in range(args.id, min(args.id_end + 1, len(video_set_names))):
        input_video_dir = f'../data/{dataset_name}/training_video/{video_set_names[idx]}/mp4/'
        output_video_dir = f'../data/{dataset_name}/training_video/{video_set_names[idx]}/dwpose/'
        npz_dir = f"../data/{dataset_name}/2d_keypoints/{video_set_names[idx]}/"

        if not os.path.exists(output_video_dir):
            os.makedirs(output_video_dir)

        video_names = sorted(os.listdir(input_video_dir), reverse=False)
        
        pool = mp.Pool(processes=args.num_processes)
        
        args_list = [(video_name, input_video_dir, output_video_dir, npz_dir) for video_name in video_names]
        
        for _ in tqdm(pool.imap_unordered(process_video, args_list), total=len(args_list), desc=f"process {video_set_names[idx]}"):
            pass
        
        pool.close()
        pool.join()

if __name__ == "__main__":
    main()
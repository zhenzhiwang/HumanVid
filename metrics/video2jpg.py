import cv2
import os
from tqdm import tqdm
def extract_frames(video_path, frames_count=24, original_stride=1):
    cap = cv2.VideoCapture(video_path)
    
    # Get total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate stride based on the condition
    if total_frames < 72:
        stride = total_frames // frames_count
        assert stride > 0, f"Stride must be greater than 0, got {stride}"
    else:
        stride = original_stride
    
    frames = []
    count = 0
    while len(frames) < frames_count:
        ret, frame = cap.read()
        if not ret:
            break
        if count % stride == 0:
            frames.append(frame)
        count += 1
    
    cap.release()
    return frames

def save_frames(frames, output_dir, base_filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for idx, frame in enumerate(frames):
        frame_filename = f"{base_filename}_frame_{idx+1}.png"
        frame_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_path, frame)

def process_videos(input_dir, output_dir, pexels_test_set, frames_count=24, stride=1):
    print(f"Processing {input_dir}, stride={stride}, frames_count={frames_count}")
    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(('.mp4')):  # Add more video formats if needed
            if not filename.lower().split('_')[-1] in pexels_test_set:
                continue
            video_path = os.path.join(input_dir, filename)
            frames = extract_frames(video_path, frames_count, stride)
            base_filename = os.path.splitext(filename)[0]
            save_frames(frames, output_dir, base_filename)

def get_image_filenames(directory):
    """
    Get a set of image filenames (without extensions) from a directory.
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}  # Add more extensions if needed
    filenames = set()
    for filename in os.listdir(directory):
        #print(filename)
        #import ipdb; ipdb.set_trace()
        if os.path.splitext(filename)[1].lower() in image_extensions:
            if "output_" in filename:
                filenames.add(filename.split("_")[1])
            else:
                filenames.add(os.path.splitext(filename)[0])
    return filenames

def delete_non_matching_images(src_dir, target_dir):
    """
    Delete images in src_dir that do not appear in target_dir.
    """
    src_filenames = get_image_filenames(src_dir)
    target_filenames = get_image_filenames(target_dir)

    non_matching_filenames = src_filenames - target_filenames

    for filename in os.listdir(src_dir):
        if os.path.splitext(filename)[0] in non_matching_filenames:
            file_path = os.path.join(src_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
                
def load_ids_from_file(input_file):
    video_ids_set = set()
    with open(input_file, 'r') as f:
        for line in f:
            video_id = line.strip()
            video_ids_set.add(video_id)
    return video_ids_set

if __name__ == "__main__":
    """pexels_test_set = os.listdir('/mnt/afs/wangzhenzhi/code/animate-with-camera/data/pexels-test-h/videos')
    pexels_test_set.extend(os.listdir('/mnt/afs/wangzhenzhi/code/animate-with-camera/data/pexels-test-v/videos'))
    input_dirs = ['/mnt/afs/wangzhenzhi/code/animate-with-camera/data/pexels-test-h/videos', 
                  '/mnt/afs/wangzhenzhi/code/animate-with-camera/data/pexels-test-v/videos',
                  '/mnt/afs/wangzhenzhi/code/animate-with-camera/output/stage2-sense-evaluation-all/pexels-test-h_20240709_1250--seed_42-512x896',
                    '/mnt/afs/wangzhenzhi/code/animate-with-camera/output/stage2-sense-evaluation-all/pexels-test-v_20240709_1210--seed_42-512x896',
                    '/mnt/afs/wangzhenzhi/code/animate-with-camera/output/stage2-sense-evaluation-real/pexels-test-h_20240709_1300--seed_42-512x896',
                    '/mnt/afs/wangzhenzhi/code/animate-with-camera/output/stage2-sense-evaluation-real/pexels-test-v_20240709_1220--seed_42-512x896',
                    '/mnt/afs/wangzhenzhi/code/animate-with-camera/output/stage2-sense-evaluation-all_ddp/pexels-test-h_20240709_1259--seed_42-512x896',
                    '/mnt/afs/wangzhenzhi/code/animate-with-camera/output/stage2-sense-evaluation-all_ddp/pexels-test-v_20240709_1218--seed_42-512x896']
    output_dirs = ['/mnt/afs/wangzhenzhi/code/animate-with-camera/data/pexels-test-h/frames', 
                   '/mnt/afs/wangzhenzhi/code/animate-with-camera/data/pexels-test-v/frames',
                   '/mnt/afs/wangzhenzhi/code/animate-with-camera/output/stage2-sense-evaluation-all/frame_pexels-test-h_20240709_1250--seed_42-512x896',
                    '/mnt/afs/wangzhenzhi/code/animate-with-camera/output/stage2-sense-evaluation-all/frame_pexels-test-v_20240709_1210--seed_42-512x896',
                    '/mnt/afs/wangzhenzhi/code/animate-with-camera/output/stage2-sense-evaluation-real/frame_pexels-test-h_20240709_1300--seed_42-512x896',
                    '/mnt/afs/wangzhenzhi/code/animate-with-camera/output/stage2-sense-evaluation-real/frame_pexels-test-v_20240709_1220--seed_42-512x896',
                    '/mnt/afs/wangzhenzhi/code/animate-with-camera/output/stage2-sense-evaluation-all_ddp/frame_pexels-test-h_20240709_1259--seed_42-512x896',
                    '/mnt/afs/wangzhenzhi/code/animate-with-camera/output/stage2-sense-evaluation-all_ddp/frame_pexels-test-v_20240709_1218--seed_42-512x896']"""
    
    frames_count = 24  # Number of frames to extract
    pexels_test_set = os.listdir('/mnt/afs/wangzhenzhi/code/animate-with-camera/data/tiktok/test/videos')
    # lower case
    pexels_test_set = [x.lower() for x in pexels_test_set]
    input_dirs = [#'/mnt/afs/wangzhenzhi/code/animate-with-camera/data/ubc/test/videos',
                  '/mnt/afs/wangzhenzhi/code/animate-with-camera/output/stage2-sense-evaluation-tiktok-dpvo-real/videos_20240816_2041--seed_42-512x896']
    output_dirs = [#'/mnt/afs/wangzhenzhi/code/animate-with-camera/data/ubc/test/frames',
                     '/mnt/afs/wangzhenzhi/code/animate-with-camera/output/stage2-sense-evaluation-tiktok-dpvo-real/frames_20240816_2041--seed_42-512x896']
    for input_dir, output_dir in zip(input_dirs,output_dirs):
        if os.path.exists(output_dir):
            print(f"Output directory {output_dir} already exists. Skipping.")
            continue
        if "pexels-test-h/" in input_dir or "pexels-test-v/" in input_dir or 'tiktok/' in input_dir or '/ubc/' in input_dir:
            stride = 3  
        elif "stage2-sense-evaluation" in input_dir:
            stride = 1
        else:
            raise ValueError("Invalid input directory")
        process_videos(input_dir, output_dir, pexels_test_set, frames_count, stride)

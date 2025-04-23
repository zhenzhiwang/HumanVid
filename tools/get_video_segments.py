import os
import json
import math
from decord import VideoReader
import multiprocessing as mp
from tqdm import tqdm
def process_video(video_meta, segment_length):
    segments = []
    video_path = video_meta["video_path"]
    kps_path = video_meta["kps_path"]

    if not os.path.exists(video_path) or not os.path.exists(kps_path):
        return segments

    try:
        video_reader = VideoReader(video_path)
        kps_reader = VideoReader(kps_path)
    except Exception as e:
        return segments

    if len(video_reader) != len(kps_reader):
        return segments

    video_length = len(video_reader)
    fps = video_reader.get_avg_fps()
    target_segment_frames = int(segment_length * fps)

    # Calculate the number of segments
    num_segments = max(1, round(video_length / target_segment_frames))
    
    # Calculate the actual segment length
    segment_frames = math.ceil(video_length / num_segments)

    for i in range(num_segments):
        start_frame = i * segment_frames
        end_frame = min((i + 1) * segment_frames, video_length)
        
        # Ensure the segment is not too short
        if i == num_segments - 1 and end_frame - start_frame < target_segment_frames * 0.5:
            # Merge with the previous segment
            if segments:
                segments[-1]["end_frame"] = end_frame
        else:
            segments.append({
                "video_path": video_path,
                "kps_path": kps_path,
                "start_frame": start_frame,
                "end_frame": end_frame
            })

    return segments

def process_video_wrapper(args):
    video_meta, segment_length = args
    return process_video(video_meta, segment_length)

def get_video_segments(meta_file_path, segment_length):
    with open(meta_file_path, "r") as f:
        meta_data = json.load(f)
    
    total_videos = len(meta_data)
    
    with mp.Pool(processes=20) as pool:
        results = list(tqdm(
            pool.imap(process_video_wrapper, [(video_meta, segment_length) for video_meta in meta_data]),
            total=total_videos,
            desc="Processing videos"
        ))
    
    segments = [segment for result in results for segment in result]
    return segments

def save_segments(segments, output_path):
    with open(output_path, "w") as f:
        json.dump(segments, f, indent=4)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Get video segments")
    parser.add_argument("--meta_file_path", type=str, default="./data/json_files/pexels-ue_horizontal.json", help="Path to the meta file")
    parser.add_argument("--segment_length", type=float, default=10.0, help="Segment length in seconds")
    args = parser.parse_args()

    args.output_path = args.meta_file_path.replace(".json", f"_{args.segment_length}s.json")
    segments = get_video_segments(args.meta_file_path, args.segment_length)
    save_segments(segments, args.output_path)

if __name__ == "__main__":
    main()
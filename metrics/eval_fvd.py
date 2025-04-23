from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from pathlib import Path
from decord import VideoReader
import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import torch
import numpy as np
from utils.fvd import calculate_fvd, load_i3d_pretrained
from datetime import datetime

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
"""
dataset_names = ['ddp_pexels-h', 'ddp_pexels-v']  # 'all_pexels-h', 'all_pexels-v', 'real_pexels-h', 'real_pexels-v', 
res_video_root_paths = [#'/mnt/afs/wangzhenzhi/code/animate-with-camera/output/stage2-sense-evaluation-all/pexels-test-h_20240709_1250--seed_42-512x896',
                        #'/mnt/afs/wangzhenzhi/code/animate-with-camera/output/stage2-sense-evaluation-all/pexels-test-v_20240709_1210--seed_42-512x896',
                        #'/mnt/afs/wangzhenzhi/code/animate-with-camera/output/stage2-sense-evaluation-real/pexels-test-h_20240709_1300--seed_42-512x896',
                        #'/mnt/afs/wangzhenzhi/code/animate-with-camera/output/stage2-sense-evaluation-real/pexels-test-v_20240709_1220--seed_42-512x896',
                        '/mnt/afs/wangzhenzhi/code/animate-with-camera/output/stage2-sense-evaluation-all_ddp/pexels-test-h_20240709_1259--seed_42-512x896',
                        '/mnt/afs/wangzhenzhi/code/animate-with-camera/output/stage2-sense-evaluation-all_ddp/pexels-test-v_20240709_1218--seed_42-512x896']
gt_video_root_paths = [#'/mnt/afs/wangzhenzhi/code/animate-with-camera/data/pexels-test-h/videos/',
                       #'/mnt/afs/wangzhenzhi/code/animate-with-camera/data/pexels-test-v/videos/',
                       #'/mnt/afs/wangzhenzhi/code/animate-with-camera/data/pexels-test-h/videos/',
                       #'/mnt/afs/wangzhenzhi/code/animate-with-camera/data/pexels-test-v/videos/',
                       '/mnt/afs/wangzhenzhi/code/animate-with-camera/data/pexels-test-h/videos/',
                       '/mnt/afs/wangzhenzhi/code/animate-with-camera/data/pexels-test-v/videos/']"""
"""dataset_names = ['tiktok_ddp']  # 'all_pexels-h', 'all_pexels-v', 'real_pexels-h', 'real_pexels-v', 
res_video_root_paths = ['/mnt/afs/wangzhenzhi/code/animate-with-camera/output/stage2-sense-evaluation-real/videos_20240722_2203--seed_42-512x896']
gt_video_root_paths = ['/mnt/afs/wangzhenzhi/code/animate-with-camera/data/tiktok/test/videos']"""
"""dataset_names = ['ubc']  # 'all_pexels-h', 'all_pexels-v', 'real_pexels-h', 'real_pexels-v', 
res_video_root_paths = ['output/stage2-sense-evaluation-ubc/test_20240813_1919--seed_42-512x672']
gt_video_root_paths = ['/mnt/afs/wangzhenzhi/code/animate-with-camera/data/ubc/test/videos']"""
dataset_names = ['tiktok-ue-1st-stage']  # 'all_pexels-h', 'all_pexels-v', 'real_pexels-h', 'real_pexels-v', 
res_video_root_paths = ['/mnt/afs/wangzhenzhi/code/animate-with-camera/output/stage2-sense-evaluation-tiktok-dpvo-real/videos_20240816_2041--seed_42-512x896']
gt_video_root_paths = ['/mnt/afs/wangzhenzhi/code/animate-with-camera/data/tiktok/test/videos']

setting = 'sample-all'


for dataset_name, res_video_root_path, gt_video_root_path in zip(dataset_names, res_video_root_paths, gt_video_root_paths):

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    save_dir = Path(f"output/fvd/{date_str}-{time_str}_{dataset_name}")
    save_dir.mkdir(exist_ok=True, parents=True)

    all_videos = os.listdir(res_video_root_path)
    all_videos = [os.path.join(res_video_root_path, f) for f in all_videos]
    all_videos.sort()
    all_videos = [ video for video in all_videos if os.path.splitext(video)[1] == '.mp4']

    device = torch.device("cuda")
    i3d = load_i3d_pretrained(device=device)

    video1_all_tensors = []
    video2_all_tensors = []
    result_all = []
    for ind, video_path in tqdm(enumerate(all_videos)):
        #print(f'ind {ind} in {len(all_videos)}')
        res_video_reader = VideoReader(video_path)
        vid = video_path.split('/')[-1].split('.')[0].split('_')[-1]
        gt_video_path = os.path.join(gt_video_root_path, f'{vid}.mp4')
        if not os.path.exists(gt_video_path):
            print(f'gt video not found: {gt_video_path}')
            continue
        gt_video_reader = VideoReader(gt_video_path)
        evaluate_inds = list(range(0, len(gt_video_reader)))
        if len(evaluate_inds) > 72:
            evaluate_inds = evaluate_inds[::3]
        evaluate_inds = evaluate_inds[:24]

        width, height = None, None
        video1_images = []
        for frame_ind, frame in enumerate(res_video_reader):
            res_frame = frame.asnumpy()
            res_frame = Image.fromarray(res_frame).convert('RGB')
            width, height = res_frame.size

            image_transform = transforms.Compose(
                        [transforms.Resize((height, width)), transforms.ToTensor()])
            video1_images.append(image_transform(res_frame))
        video1 = torch.stack(video1_images, dim=0)
        
        video2_images = []
        for eval_ind in evaluate_inds:
            gt_frame = gt_video_reader[eval_ind].asnumpy()
            gt_frame = Image.fromarray(gt_frame).convert('RGB')

            image_transform = transforms.Compose(
                        [transforms.Resize((height, width)), transforms.ToTensor()])
            video2_images.append(image_transform(gt_frame))
        video2 = torch.stack(video2_images, dim=0)

        video1 = video1.unsqueeze(0)
        video2 = video2.unsqueeze(0)

        #print(video1.shape, video2.shape)
        result = calculate_fvd(video1, video2, device, i3d=i3d, method='styleganv')

        values = list(result['value'].values())
        result_all.extend(values)

    fvd_result = torch.tensor(result_all).mean().item()
    print(f'{dataset_name}: FVD = {fvd_result}')
    result_dict = {"fvd": fvd_result}
    with open(os.path.join(save_dir, "fvd_results.json"), 'w') as fp:
        json.dump(result_dict, fp, indent=True)
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
import torch
import torchvision
import numpy as np
from datetime import datetime
import multiprocessing as mp

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

# moore result
# dataset_name = 'pexels-test-h'
# res_video_root_path = '/mnt/petrelfs/liuwenran/forks/Moore-AnimateAnyone/output/20240604/2112--video_pexels-v--seed_42-768x768'
# gt_video_root_path = '/mnt/hwfile/mm_lol/liuwenran/pexels-test-case/pexels-test-v/videos'

# dataset_name = 'tiktok'
# res_video_root_path = '/mnt/petrelfs/liuwenran/forks/Moore-AnimateAnyone/output/20240604/2113--video_tiktok--seed_42-768x768'
# gt_video_root_path = '/mnt/hwfile/mm_lol/fangyq/share_data/tiktok_video'

# magic animate result
# dataset_name = 'magic-animate_tiktok'
# res_video_root_path = '/mnt/petrelfs/liuwenran/forks/magic-animate/samples/tiktok--2024-06-05T16-33-58'
# gt_video_root_path = '/mnt/hwfile/mm_lol/fangyq/share_data/tiktok_video'

# dataset_name = 'magic-animate_pexels-test-h'
# res_video_root_path = '/mnt/petrelfs/liuwenran/forks/magic-animate/samples/pexels-h--2024-06-05T16-39-26'
# gt_video_root_path = '/mnt/hwfile/mm_lol/liuwenran/pexels-test-case/pexels-test-h/videos'

# dataset_name = 'magic-animate_pexels-test-v'
# res_video_root_path = '/mnt/petrelfs/liuwenran/forks/magic-animate/samples/pexels-v--2024-06-05T16-57-53'
# gt_video_root_path = '/mnt/hwfile/mm_lol/liuwenran/pexels-test-case/pexels-test-v/videos'

# champ result
# dataset_name = 'champ_tiktok'
# res_video_root_path = '/mnt/petrelfs/liuwenran/repos/champ/results/tiktok-2024-06-05T12-05-19'
# gt_video_root_path = '/mnt/hwfile/mm_lol/fangyq/share_data/tiktok_video'

# dataset_name = 'champ_pexels-h'
# res_video_root_path = '/mnt/petrelfs/liuwenran/repos/champ/results/pexels-h-2024-06-05T19-08-03'
# gt_video_root_path = '/mnt/hwfile/mm_lol/liuwenran/pexels-test-case/pexels-test-h/videos'

"""dataset_names = ['ddp_pexels-h', 'ddp_pexels-v']  # 'all_pexels-h', 'all_pexels-v', 'real_pexels-h', 'real_pexels-v', 
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

dataset_names = ['tiktok-ue-1st-stage']  # 'all_pexels-h', 'all_pexels-v', 'real_pexels-h', 'real_pexels-v', 
res_video_root_paths = ['/mnt/afs/wangzhenzhi/code/animate-with-camera/output/stage2-sense-evaluation-tiktok-dpvo-real/videos_20240816_2041--seed_42-512x896']
gt_video_root_paths = ['/mnt/afs/wangzhenzhi/code/animate-with-camera/data/tiktok/test/videos']

"""dataset_names = ['ubc']  # 'all_pexels-h', 'all_pexels-v', 'real_pexels-h', 'real_pexels-v', 
res_video_root_paths = ['output/stage2-sense-evaluation-ubc/test_20240813_1919--seed_42-512x672']
gt_video_root_paths = ['/mnt/afs/wangzhenzhi/code/animate-with-camera/data/ubc/test/videos']"""

setting = 'sample-once'


def process_frame(args):
    res_frame, gt_frame, height, width = args
    image_transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor()
    ])
    image = torch.stack([image_transform(res_frame)], dim=0).cuda()
    gt_img = torch.stack([image_transform(gt_frame)], dim=0).cuda()
    return {
        'ssim': ssim(image, gt_img).item(),
        'psnr': psnr(image, gt_img).item(),
        'lpips': lpips(image, gt_img, net_type='vgg').item()
    }

def process_video(args):
    video_path, gt_video_root_path, setting, save_dir = args
    res_video_reader = VideoReader(video_path)
    vid = video_path.split('/')[-1].split('.')[0].split('_')[1]
    gt_video_path = os.path.join(gt_video_root_path, f'{vid}.mp4')
    if not os.path.exists(gt_video_path):
        return None
    gt_video_reader = VideoReader(gt_video_path)
    evaluate_inds = list(range(0, len(gt_video_reader)))
    if len(evaluate_inds) > 72:
        evaluate_inds = evaluate_inds[::3]
    else:
        stride = len(evaluate_inds) // 24
        evaluate_inds = evaluate_inds[::stride]

    if setting == 'sample-once':
        sample_ind = np.random.randint(0, len(res_video_reader))
        frame_inds = [sample_ind]
    elif setting == 'sample-all':
        frame_inds = range(len(res_video_reader))
    else:
        raise ValueError(f"Unknown setting: {setting}")

    frame_args = []
    for frame_ind in frame_inds:
        res_frame = res_video_reader[frame_ind].asnumpy()
        res_frame = Image.fromarray(res_frame).convert('RGB')
        width, height = res_frame.size

        eval_ind = evaluate_inds[frame_ind]
        gt_frame = gt_video_reader[eval_ind].asnumpy()
        gt_frame = Image.fromarray(gt_frame).convert('RGB')

        frame_args.append((res_frame, gt_frame, height, width))

    if setting == 'sample-all':
        with mp.Pool(processes=6) as pool:
            results = list(tqdm(pool.imap(process_frame, frame_args), 
                                total=len(frame_args), 
                                desc=f"Processing {vid}", 
                                leave=False))
    else:
        results = [process_frame(args) for args in frame_args]

    return results

def main(dataset_names, res_video_root_paths, gt_video_root_paths, setting):
    for dataset_name, res_video_root_path, gt_video_root_path in zip(dataset_names, res_video_root_paths, gt_video_root_paths):
        date_str = datetime.now().strftime("%Y%m%d")
        time_str = datetime.now().strftime("%H%M")
        save_dir = Path(f"output/images_metrics_result/{date_str}-{time_str}_{dataset_name}")
        save_dir.mkdir(exist_ok=True, parents=True)

        all_videos = [os.path.join(res_video_root_path, f) for f in os.listdir(res_video_root_path) if f.endswith('.mp4')]
        all_videos.sort()

        args_list = [(video_path, gt_video_root_path, setting, save_dir) for video_path in all_videos]

        if setting == 'sample-all':
            all_results = []
            for args in args_list:
                results = process_video(args)
                all_results.append(results)
        elif setting == 'sample-once':
            with mp.Pool(processes=10) as pool:
                all_results = list(tqdm(pool.imap(process_video, args_list), 
                                        total=len(args_list), 
                                        desc=f"Processing {dataset_name}", 
                                        leave=False))
        else:
            raise ValueError(f"Unknown setting: {setting}")

        # Aggregate results
        ssims = []
        psnrs = []
        lpipss = []
        for video_results in all_results:
            if video_results is not None:
                for result in video_results:
                    ssims.append(result['ssim'])
                    psnrs.append(result['psnr'])
                    lpipss.append(result['lpips'])

        ssim_result = np.mean(ssims)
        psnr_result = np.mean(psnrs)
        lpips_result = np.mean(lpipss)
        
        result_dict = {"SSIM": ssim_result, "PSNR": psnr_result, "LPIPS": lpips_result}
        print(f"\nResults for {dataset_name}:")
        print("SSIM : {:>12.7f}".format(ssim_result))
        print("PSNR : {:>12.7f}".format(psnr_result))
        print("LPIPS: {:>12.7f}".format(lpips_result))
        
        with open(os.path.join(save_dir, "results.json"), 'w') as fp:
            json.dump(result_dict, fp, indent=True)

if __name__ == '__main__':
    main(dataset_names, res_video_root_paths, gt_video_root_paths, setting)
# HumanVid [NeurIPS D&B Track 2024]
<div align='Center'>
    <a href='https://humanvid.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
    <a href='https://arxiv.org/abs/2407.17438'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
</div>

This repository is the official implementation of the paper:
>[**HumanVid: : Demystifying Training Data for Camera-controllable Human Image Animation**](https://arxiv.org/abs/2407.17438) <br>
> [Zhenzhi Wang](https://zhenzhiwang.github.io/), [Yixuan Li](https://yixuanli98.github.io/), [Yanhong Zeng](https://zengyh1900.github.io/), [Youqing Fang](#), [Yuwei Guo](https://guoyww.github.io/), <br> [Wenran Liu](#), [Jing Tan](https://sparkstj.github.io/), [Kai Chen](https://chenkai.site/), [Tianfan Xue](https://tianfan.info/), [Bo Dai](https://daibo.info/), [Dahua Lin](http://dahua.site/) <br>
CUHK, Shanghai AI Lab

## TL;DR
HumanVid is a new dataset for camera-controllable human image animation, which enables training video diffusion models to generate videos with both camera and subject movements like real movie clips. As a by-product, it could also enable reproducing methods like Animate-Anyone by just setting the camera to be static in the inference. We show that models only trained on videos with camera movement could achieve very good static background appearance as long as the camera annotations in the training set is accurate, reducing the difficulty of static-camera video collection. To verify our statements, our proposed baseline model CamAnimate shows impressive results, which could be found in the [website](https://humanvid.github.io/). This repo will provide all the data and code to achieve the performance shown in our demo.

## Framework
![framework](assets/framework.png)

## News
- **`2025/04/23`**: We released all training code, inference code, and checkpoints.
- **`2025/01/03`** : More accurate camera parameters are released in the Google Drive named `camera_tram.zip`. It is predicted from the original [tram](https://github.com/yufu-wang/tram) method with SAM masks and Driod-SLAM. Such camera annotations cost much more GPU hours than the previous version, and it produces better camera control ability by only using Internet videos. We also updated `Camera` class in `src/dataset/img_dataset.py`.
- **`2024/10/20`**: The UE synthetic video part of HumanVid is released. Please download the videos, human poses and camera parameters from [here](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155189552_link_cuhk_edu_hk/EoLw3qmoeFFEq88f87TZRfkB6w7FWFsnfeedfu52wk1rgw?e=yYH0n1), password `humanvid_ue`.
- **`2024/09/27`**: Our paper is accepted by NeurIPS D&B Track 2024.
- **`2024/09/02`**: The Internet video part of HumanVid is released. Please download the video urls and camera parameters from [here](https://drive.google.com/drive/folders/1UGEkOKXYX9BGUFz0ao6lOGXkZjQGoJcZ?usp=sharing). 

## Video Data

### Pexels videos
The pexels video data is collected from the Internet and we cannot redistribute them. We provide the video urls and camera parameters for each video. The camera parameters are stored in the `camera.zip` in the Google Drive. The videos could be downloaded by scripts from urls.

**Updates:** The video urls in Pexels.com are changed by the website team. We have updated the video urls in the txt file ending with `new`. Please use the new urls for downloading the videos.

### Unreal Engine rendered videos
The videos are in the OneDrive link (the `training_video` folder). `3d_video_*` means videos are rendered from 3D scene background and `generated_video_*` means videos are rendered from HDRI images as background. The final file structure should be like:
```
2d_keypoints/
training_video/
├── 3d_video_1/
│   ├── camera/
│   ├── dwpose/
│   └── mp4/
├── 3d_video_2/
│   ├── camera/
│   ├── dwpose/
│   └── mp4/
├── 3d_video_3/
├── 3d_video_4/
├── 3d_video_5/
├── 3d_video_6/
├── 3d_video_7/
├── 3d_video_8/
├── 3d_video_9/
├── 3d_video_10/
├── generated_video_1/
├── generated_video_2/
├── generated_video_3/
├── generated_video_4/
├── generated_video_5/
├── generated_video_6/
├── generated_video_7/
├── generated_video_8/
├── generated_video_9/
└── generated_video_10/
```
The `training_video` folder of OneDrive link contains all files in the `mp4` folder. Please first unzip `ue_camera.zip` and put each sub-folder to the corresponding position in `training_video`. Then unzip `2d_keypoints.zip` and use `python extract_pose_from_smplx_ue.py` to produce mp4 dwpose files from the 2d keypoints information saved in the `2d_keypoints` folder.


## Camera Trajectory Format
We follow [Droid-SLAM](https://github.com/princeton-vl/DROID-SLAM) and [DPVO](https://github.com/princeton-vl/DPVO) use [TUM Camera Format](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats) `timestamp tx ty tz qx qy qz qw` format for camera trajectory. The timestamp is number of frame. The tx, ty, tz are the translation of the camera in meters. The qx, qy, qz, qw are the quaternion of the camera rotation. For camera intrinsics, assuming the camera has a standard 36mm CMOS, we heuristically set the focal length to 50mm (horizontal) and 75mm (vertical) and the principal point to the center of the image, based on the observation on Internet videos. We empirically find that it works well.

For Unreal Engine rendered videos, we provide the camera parameters in the `ue_camera.zip` file. The format is `timestamp tx ty tz qx qy qz qw fx fy`, where the first 8 numbers are the same as the TUM format. The fx and fy are the normalized focal lengths (intrinsics).

To better understand our camera parameter processing scripts (e.g., modifications over CameraCtrl), the camera processing code is in `src/dataset/img_dataset.py`. The complete code will be released later.


## Human Pose Extraction
Please refer to the `DWPose` folder for scripts of extracting and visualizing whole-body poses. Note that I have added a little modification on foot by also visualizing the keypoints on the foot. It also contains the keypoints convertion from SMPL-X to COCO Keypoints format. For pretrained checkpoints, please refer to the [DWPose](https://github.com/IDEA-Research/DWPose) repository.


### Usage
This script will extract the whole-body pose for all videos in a given folder, e.g., `videos`. The extracted poses will be stored in the `dwpose` folder.
```
cd DWPose
python prepare_video.py
```

### SMPL-X to COCO Whole-body Keypoints
This script could read existing 2D SMPL-X keypoints (i.e., already projected to a camera space) and convert them to COCO whole-body keypoints format and visualize them like the `DWPose`'s output. The projection script from 3D SMPL-X keypoints to 2D could be found in [here](https://github.com/pixelite1201/BEDLAM/blob/master/data_processing/df_full_body.py). The SMPL-X keypoints is in the `2d_keypoints.zip` of OneDrive link and camera parameters is the `ue_camera.zip`. Use the following command to extract the whole-body pose videos from SMPL-X keypoints.
```
python extract_pose_from_smplx_ue.py
```


## Training and Inference

### Conda Environment
Please prepare conda environment following [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone). We also provided a `environment.yml` file for reference.

### Download weights
Please prepare weights following [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone). Please also download the SD1.5 verison weight of [CameraCtrl](https://github.com/hehao13/CameraCtrl) and put it to `pretrained_weights/cameractrl`.

Our pretrained checkpoints could be accessed in [HuggingFace](https://huggingface.co/zhenzhiwang/humanvid). This checkpoint should be equal to the performance shown in our [homepage](https://humanvid.github.io/).

### Usage
**Training**, stage1: `bash scripts/train_s1.sh` and stage2: `bash scripts/train_s2.sh`.

**Inference**: `bash scripts/eval.sh`.

Our code structure is very similar to [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone). Please check their readme for more details.

## Todo
- [x] Release the synthetic data part.
- [x] Release the inference code.
- [x] Release the training code and checkpoint.

Please give us a star if you are interested in our work. Thanks!
## Bibtex
```
@inproceedings{DBLP:conf/nips/00010ZF0LTCX0L24,
  author       = {Zhenzhi Wang and
                  Yixuan Li and
                  Yanhong Zeng and
                  Youqing Fang and
                  Yuwei Guo and
                  Wenran Liu and
                  Jing Tan and
                  Kai Chen and
                  Tianfan Xue and
                  Bo Dai and
                  Dahua Lin},
  title        = {HumanVid: Demystifying Training Data for Camera-controllable Human
                  Image Animation},
  booktitle    = {NeurIPS},
  year         = {2024}
}
```

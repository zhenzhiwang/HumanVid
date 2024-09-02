from .keypoints_format import SMPLX_KEYPOINTS, COCO_WHOLEBODY_KEYPOINTS
import numpy as np
from .util import *
import cv2
from tqdm import tqdm
import os
from matplotlib import cm
import colorsys
import imageio
import random

def draw_pose(pose, H, W, source_frame = None):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    foots = pose['foots']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8) if source_frame is None else source_frame.copy()

    scale = max(H, W) / 1920
    canvas = draw_bodypose(canvas, candidate, subset, scale=scale)
    canvas = draw_handpose(canvas, hands, scale = scale)
    canvas = draw_facepose(canvas, faces, scale = scale)
    canvas = draw_footpose(canvas, foots, scale = scale)
    return canvas

def smplx_to_coco_wholebody(smplx_keypoints):
    # Create a mapping from SMPLX keypoint names to their indices
    smplx_keypoint_dict = {name: i for i, name in enumerate(SMPLX_KEYPOINTS)}
    
    # Create an output array with the same number of people, but with COCO Wholebody keypoint count
    coco_keypoints = np.zeros((len(COCO_WHOLEBODY_KEYPOINTS), 3))
    
    # Map SMPLX keypoints to COCO Wholebody keypoints
    for i, coco_kpt in enumerate(COCO_WHOLEBODY_KEYPOINTS):
        if coco_kpt in smplx_keypoint_dict:
            coco_keypoints[ i] = smplx_keypoints[smplx_keypoint_dict[coco_kpt]]
    
    # Handle special cases
    # Left hand root (use left wrist)
    coco_keypoints[COCO_WHOLEBODY_KEYPOINTS.index('left_hand_root')] = smplx_keypoints[ smplx_keypoint_dict['left_wrist']]
    
    # Right hand root (use right wrist)
    coco_keypoints[COCO_WHOLEBODY_KEYPOINTS.index('right_hand_root')] = smplx_keypoints[ smplx_keypoint_dict['right_wrist']]
    
    return coco_keypoints



def DrawDWposeFrame(candidate, subset, H, W, source_frame= None):
    copy_score = subset[:,:18].copy()
    nums, keys, locs = candidate.shape
    candidate[..., 0] /= float(W)
    candidate[..., 1] /= float(H)
    body = candidate[:,:18].copy()
    body = body.reshape(nums*18, locs)
    score = subset[:,:18]
    for i in range(len(score)):
        for j in range(len(score[i])):
            if score[i][j] > 0.3:
                score[i][j] = int(18*i+j)
            else:
                score[i][j] = -1

    un_visible = subset < 0.3
    #import ipdb; ipdb.set_trace()
    candidate[un_visible] = -1

    foots = candidate[:,18:24]  # 6 points
    left_body_foot = candidate[:,13:14]
    right_body_foot = candidate[:,10:11]
    foots = np.concatenate((left_body_foot, right_body_foot, foots), axis=1)

    faces = candidate[:,24:92] # 68 points

    hands = candidate[:,92:113]  # 21 points for each hand
    
    hands = np.vstack([hands, candidate[:,113:]])  # 21 points for each hand
    
    bodies = dict(candidate=body, subset=score)
    pose = dict(bodies=bodies, hands=hands, faces=faces, foots = foots)
    
    #min_keypoint_ratio = 0.3  # Define the minimum number of keypoints required for a valid pose.
    #human_scores = copy_score.mean(-1)
    #valid_humans = (human_scores >= min_keypoint_ratio).sum()  # Count the number of valid human poses detected.
    #pose_array = np.concatenate((candidate[0,:18].copy(), hands[0], faces[0]), axis=0)
    #human_part_info = {"score": human_scores.max(), "num_humans": valid_humans, "largest_ratio": largest_person_ratio, "keypoints_motion": pose_array}
    return draw_pose(pose, H, W, source_frame=source_frame)

def expand_array(arr):
    n, c, _ = arr.shape
    zeros = np.zeros((n, 1, 3))
    return np.concatenate((arr, zeros), axis=1)

def visualizeVideo(npz_file, input_video, output_file, visualize_over_original_video=False):
    # Load the npz file
    data = dict(np.load(npz_file))
    gtkps = data['gtkps']
    frame_idxs = data['frame_idxs']
    kp_len = len(gtkps)
    gtkps = np.array([smplx_to_coco_wholebody(gtkp) for gtkp in gtkps])
    keypoints, scores = gtkps[...,0:2], gtkps[...,2]
    keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)
    # compute neck joint
    neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
    # neck score when visualizing pred
    neck[:, 2:4] = np.logical_and(
        keypoints_info[:, 5, 2:4] > 0.3,
        keypoints_info[:, 6, 2:4] > 0.3).astype(int)
    new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
    mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
    openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
    new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
    keypoints_info = new_keypoints_info[:, None, :]
    keypoints, scores = keypoints_info[..., :2], keypoints_info[..., 2]

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames != kp_len:
        print(f"Error: Number of frames in video ({total_frames}) does not match number of keypoints ({kp_len}).")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (W, H))

    frame_idx_set = set(frame_idxs)
    candidate_idx = 0

    for current_frame in range(total_frames):
        if visualize_over_original_video:
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read frame {current_frame}.")
                break
            canvas = frame
        else:
            canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

        if current_frame in frame_idx_set:
            canvas = DrawDWposeFrame(keypoints[candidate_idx], scores[candidate_idx], H, W, canvas)
            candidate_idx += 1
        out.write(canvas)

    cap.release()
    out.release()


def visualizeVideo_point_tracking(npz_file, input_video, output_file, visualize_over_original_video=False):
    # Load the npz file
    data = dict(np.load(npz_file))
    # numbers = random.sample(range(21), 5)
    # numbers_1 = random.sample(range(76, 144), 2)
    # numbers.extend(numbers_1)
    # print(numbers)
    numbers=[5, 67, 85, 17, 110]
    gtkps = data['gtkps'][:,numbers,:]
    frame_idxs = data['frame_idxs']
    kp_len = len(gtkps)
    color_map = cm.get_cmap("jet")
    num_pts = gtkps.shape[1]

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames != kp_len:
        print(f"Error: Number of frames in video ({total_frames}) does not match number of keypoints ({kp_len}).")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(output_file, fourcc, fps, (W, H))

    frame_idx_set = set(frame_idxs)
    frames=[]

    for current_frame in range(total_frames):
        ret, img_curr = cap.read()
        if not ret:
            print(f"Error: Could not read frame {current_frame}.")
            break

        for t in range(current_frame): 
            img1 = img_curr.copy()
            # changing opacity
            alpha = max(1 - 0.9 * ((current_frame - t) / ((current_frame + 1) * .99)), 0.1)
            for j in range(num_pts):
                color = np.array(color_map(j/max(1, float(num_pts - 1)))[:3]) * 255

                color_alpha = 1

                hsv = colorsys.rgb_to_hsv(color[0], color[1], color[2])
                color = colorsys.hsv_to_rgb(hsv[0], hsv[1]*color_alpha, hsv[2])

                pt1 = gtkps[t, j]
                pt2 = gtkps[t+1, j]
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))

                cv2.line(img1, p1, p2, color, thickness=2, lineType=16)
            img_curr = cv2.addWeighted(img1, alpha, img_curr, 1 - alpha, 0)
        
        if current_frame in frame_idx_set:
            for j in range(num_pts):
                color = np.array(color_map(j/max(1, float(num_pts - 1)))[:3]) * 255
                pt1 = gtkps[current_frame, j]
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                cv2.circle(img_curr, p1, 4, color, -1, lineType=16)
        # out.write(img_curr)
        frames.append(img_curr[..., [2, 1, 0]])
    imageio.mimwrite(output_file, frames, quality=8, fps=fps)

    cap.release()
    # out.release()


if __name__ == "__main__":
    npz_file = '../data/ue_rendered/2d_keypoints/generated_video_1/seq_000000.npz'
    input_video = npz_file.replace('.npz', '.mp4').replace('2d_keypoints', 'training_video').replace('/seq', '/mp4/seq')
    output_file = input_video.replace('/mp4/', '/dwpose/')
    output_folder = '/'.join(output_file.split('/')[:-1])
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    visualizeVideo(npz_file, input_video, output_file, visualize_over_original_video=True)



    
        







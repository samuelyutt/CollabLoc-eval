import matplotlib.style as mplstyle
mplstyle.use('fast')

import numpy as np
from utils import *

dataset = 'example_dataset'
test_id = '2023-05-16_15-34-13'

# Red, Purple,   Blue,      Orange,           Yellow
dis_s, dis_c_on, dis_c_off, dis_sampled_c_on, dis_sampled_c_off = True, True, False, False, False
dis_c_offset = 10

points3D_fname = f'../data/{dataset}/model/points3D.txt'
gtposes_fname = f'../data/{dataset}/ar_log/{test_id}/gt_poses.txt'
sampledimgs_fname = f'../data/{dataset}/ar_log/{test_id}/sampled_imgs_log.txt'
log_fname = f'../data/{dataset}/ar_log/{test_id}/log.txt'


def main():
    # Load checkpoint poses
    eval_scale = np.asarray(1.0)
    scale = eval_scale
    cam_poses = []
    ckpts = CheckPoints()
    log_lines = read_file_lines(log_fname)

    for line_idx, line in enumerate(log_lines):
        tokens = parse_log_line(line)
        if len(tokens) < 1:
            continue

        if tokens['log_type'] == 'Received':
            q, t = parse_arpose(tokens['ckpt_s_world_pose'])
            ckpt_s_pose = Pose(q, t)
            if dis_s:
                cam_poses.append((ckpt_s_pose, 'red' if tokens['is_ckpt_avail'] == 'true' else 'black'))

            q, t = parse_arpose(tokens['ckpt_c_abs_pose'])
            ckpt_c_pose = Pose(q, t)

            prev_c_pose, prev_s_pose, _, _ = ckpts.get(tokens['ckpt_image_idx'], accept_earlier=True)
            
            scale = float(tokens['cur_scale'])

            ckpts.add(tokens['ckpt_image_idx'], ckpt_c_pose, ckpt_s_pose, scale, 1.0)


    # Load client poses: not necessary for error measurement
    for line_idx, line in enumerate(log_lines):
        tokens = parse_log_line(line)
        if len(tokens) < 1:
            continue
        if skip(test_id, tokens):
            continue

        if tokens['log_type'] == 'ARPose':
            if line_idx % dis_c_offset != 0:
                continue
            if 'null' in tokens['cur_fused_world_pose']:
                continue

            # Fused pose by client
            q, t = parse_arpose(tokens['cur_fused_world_pose'])
            fused_pose = Pose(q, t)
            if dis_c_on:
                cam_poses.append((fused_pose, 'purple'))

            # Fused pose by offline calculation
            q, t = parse_arpose(tokens['cur_c_abs_pose'])
            cur_c_pose = Pose(q, t)
            ckpt_c_pose, ckpt_s_pose, scale, confidence = ckpts.get(int(tokens['cur_image_idx']) - 1, accept_earlier=True)
            tmp_fused_pose = calc_fused_pose(cur_c_pose, ckpt_c_pose, ckpt_s_pose, scale)
            if dis_c_off:
                cam_poses.append((tmp_fused_pose, 'blue'))


    # Load client sample
    sampledimgs_lines = read_file_lines(sampledimgs_fname)

    for line_idx, line in enumerate(sampledimgs_lines):
        tokens = parse_log_line(line)
        if len(tokens) < 1:
            continue
        if skip(test_id, tokens):
            continue

        if tokens['log_type'] == 'SampledPose':
            # Sampled fused pose by client
            q, t = parse_arpose(tokens['fused_world_pose'])
            sampled_pose = Pose(q, t)
            if dis_sampled_c_on:
                cam_poses.append((sampled_pose, 'orange'))

            # Sampled fused pose by offline calculation
            q, t = parse_arpose(tokens['c_abs_pose'])
            sampled_c_pose = Pose(q, t)
            ckpt_c_pose, ckpt_s_pose, scale, confidence = ckpts.get(tokens['latest_avail_ckpt_idx'], accept_earlier=True)
            sampled_fused_pose = calc_fused_pose(sampled_c_pose, ckpt_c_pose, ckpt_s_pose, scale)
            if dis_sampled_c_off:
                cam_poses.append((sampled_fused_pose, 'yellow'))

    display_pose(cam_poses, load_point_cloud(points3D_fname), False)


if __name__ == '__main__':
    main()

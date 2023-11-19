import matplotlib.style as mplstyle
mplstyle.use('fast')

import numpy as np
from utils import *

dataset = 'example_dataset'
test_id = '2023-05-16_15-32-49'

eval_scale = np.asarray(0.5321026860470256)
fused_offline = False
display = True


points3D_fname = f'../data/{dataset}/model/points3D.txt'
gtposes_fname = f'../data/{dataset}/ar_log/{test_id}/gt_poses.txt'
sampledimgs_fname = f'../data/{dataset}/ar_log/{test_id}/sampled_imgs_log.txt'
log_fname = f'../data/{dataset}/ar_log/{test_id}/log.txt'


def main():
    # Load ground truth
    gt_poses = load_gt_poses(gtposes_fname)
    c_idxs, c_dists, c_angs, s_idxs, s_dists, s_angs = [], [], [], [], [], []

    # Calculate accumlated client images counts
    sampledimgs_lines = read_file_lines(sampledimgs_fname)
    acc_img_cnts = calc_acc_img_cnts(sampledimgs_lines)


    # Load checkpoint poses
    scale = 1
    scales = []
    cam_poses = []
    ckpts = CheckPoints()
    log_lines = read_file_lines(log_fname)

    for line_idx, line in enumerate(log_lines):
        # Tokens format
        # 0:Received|1:ckpt_image_idx|2:ckpt_c_pose|3:ckpt_c_abs_pose|4:ckpt_s_pose|5:ckpt_s_world_pose|6:cur_scale|7:ckpt_fused_world_pose|8:old_fused_world_pose
        # 0:ARPose|1:ckpt_image_idx|2:ckpt_c_pose|3:ckpt_c_abs_pose|4:ckpt_s_pose|5:ckpt_s_world_pose|6:cur_c_pose|7:cur_c_abs_pose|8:cur_fused_world_pose|9:cur_image_idx
        tokens = parse_log_line(line)
        if len(tokens) < 1:
            continue
        if skip(test_id, tokens):
            continue

        if tokens['log_type'] == 'Received':
            q, t = parse_arpose(tokens['ckpt_s_world_pose'])
            ckpt_s_pose = Pose(q, t)
            
            # Evaluation on distance and angle
            dist, ang = gt_poses.compare(f'server-{tokens["ckpt_image_idx"]}', ckpt_s_pose)
            if dist is not None and ang is not None:
                s_dists.append(dist)
                s_angs.append(ang)
                s_idxs.append(acc_img_cnts[int(tokens['ckpt_image_idx'])])

            q, t = parse_arpose(tokens['ckpt_c_abs_pose'])
            ckpt_c_pose = Pose(q, t)

            prev_c_pose, prev_s_pose, _, _ = ckpts.get(tokens['ckpt_image_idx'], accept_earlier=True)
            if prev_s_pose is not None:
                dist_s_poses = calc_dist(ckpt_s_pose.pos(), prev_s_pose.pos())
                dist_c_poses = calc_dist(ckpt_c_pose.pos(), prev_c_pose.pos())
                tmp_scale = dist_s_poses / dist_c_poses
                # scales.append(tmp_scale)
                # scale = np.mean(scales)
                scale = eval_scale
            ckpts.add(tokens['ckpt_image_idx'], ckpt_c_pose, ckpt_s_pose, scale, 0.8)

            if dist is not None:
                tmp_dist = dist / eval_scale
                if tokens['is_ckpt_avail'] == 'true':
                    color = 'black'
                else:
                    color = get_color_by_dist(tmp_dist)

                cam_poses.append((ckpt_s_pose, color, 3))

                if tmp_dist > 0.3:
                    print(f'server-{tokens["ckpt_image_idx"]}', tmp_dist, '*' if tmp_dist > 1 else '')
                    tmp_pose = gt_poses.get(f'server-{tokens["ckpt_image_idx"]}')
                    cam_poses.append((tmp_pose, 'blue', 2))

    # Load client sample
    for line_idx, line in enumerate(sampledimgs_lines):
        # Tokens format
        # 0:SampledPose|1:cur_image_idx|2:cur_image_idx_stamp|3:c_abs_pose|4:fused_world_pose|5:latest_avail_ckpt_idx
        tokens = parse_log_line(line)
        if len(tokens) < 1:
            continue
        if skip(test_id, tokens):
            continue

        if tokens['log_type'] == 'SampledPose':
            if not fused_offline:
                # Sampled fused pose by client
                q, t = parse_arpose(tokens['fused_world_pose'])
                sampled_fused_pose = Pose(q, t)
            else:
                # Sampled fused pose by offline calculation
                q, t = parse_arpose(tokens['c_abs_pose'])
                sampled_c_pose = Pose(q, t)
                ckpt_c_pose, ckpt_s_pose, scale, confidence = ckpts.get(int(tokens['cur_image_idx']) - 1, accept_earlier=True)
                sampled_fused_pose = calc_fused_pose(sampled_c_pose, ckpt_c_pose, ckpt_s_pose, scale)

            # Evaluation on distance and angle
            dist, ang = gt_poses.compare(f'client-{tokens["cur_image_idx_stamp"]}', sampled_fused_pose)
            if dist is not None and ang is not None:
                c_dists.append(dist)
                c_angs.append(ang)
                tmp_tok = tokens['cur_image_idx_stamp'].split('-')
                prev_acc_imgs_cnts = 0
                for i in range(int(tmp_tok[0]) - 1, 0, -1):
                    if i in acc_img_cnts:
                        prev_acc_imgs_cnts = acc_img_cnts[i]
                        break
                c_idxs.append(prev_acc_imgs_cnts + int(tmp_tok[1]))
                tmp_dist = dist / eval_scale
                color = get_color_by_dist(tmp_dist)
                cam_poses.append((sampled_fused_pose, color, 0.5))

                if tmp_dist > 0.3:
                    print(f'client-{tokens["cur_image_idx_stamp"]}', tmp_dist, '*' if tmp_dist > 1 else '')
                    tmp_pose = gt_poses.get(f'client-{tokens["cur_image_idx_stamp"]}')
                    cam_poses.append((tmp_pose, 'blue', 0.5))

    s_dists = s_dists / eval_scale * 100
    c_dists = c_dists / eval_scale * 100

    print('Evaluation Report')
    print(dataset, test_id)
    print(f'Sampled client poses are fused {"offline" if fused_offline else "online"}')
    print('eval_scale', eval_scale)
    print('dist server median', np.median(s_dists))
    print('dist server mean', np.mean(s_dists))
    print('dist server max', np.max(s_dists))
    print('dist server min', np.min(s_dists))
    print('dist client median', np.median(c_dists))
    print('dist client mean', np.mean(c_dists))
    print('dist client max', np.max(c_dists))
    print('dist client min', np.min(c_dists))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(s_idxs, s_dists, label='SSL pose')
    ax.plot(c_idxs, c_dists, label='fused pose')
    ax.legend()
    plt.draw()

    if display:
        display_pose(cam_poses, load_point_cloud(points3D_fname), only_pos=True)


if __name__ == '__main__':
    main()

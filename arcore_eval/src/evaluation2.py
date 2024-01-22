import matplotlib.style as mplstyle
mplstyle.use('fast')

import numpy as np
import argparse
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--test_id', type=str, required=True, nargs='+')
parser.add_argument('--display', action='store_true')
parser.add_argument('--display_only_position', action='store_true')
parser.add_argument('--fused_offline', action='store_true')
args = parser.parse_args()


dataset = args.model_name
test_ids = args.test_id
fused_offline = args.fused_offline
display = args.display


points3D_fname = f'../data/{dataset}/model/points3D.txt'
colors = [
    'crimson', 'darkviolet', 'dodgerblue', 'c', 'darkslategray', 'olive',
    'darkgoldenrod', 'coral', 'maroon', 'fuchsia', 'slateblue', 'g', 
    'darkgoldenrod', 'sienna', 'violet', 'palegreen', 'teal', 'slategray'
]

def main():
    c_idxs, c_dists, c_angs, s_idxs, s_dists, s_angs = [], [], [], [], [], []
    cam_poses = []
    
    for track_idx, test_id in enumerate(test_ids):
        gtposes_fname = f'../data/{dataset}/ar_log/{test_id}/gt_poses.txt'
        sampledimgs_fname = f'../data/{dataset}/ar_log/{test_id}/sampled_imgs_log.txt'
        log_fname = f'../data/{dataset}/ar_log/{test_id}/log.txt'
        
        # Load ground truth
        gt_poses = load_gt_poses(gtposes_fname)

        # Load checkpoint poses
        eval_scale = np.asarray(1.0)
        scale = 1
        scales = []
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
                cam_poses.append((ckpt_s_pose, 'red'))
                
                # Evaluation on distance and angle
                dist, ang = gt_poses.compare(f'server-{tokens["ckpt_image_idx"]}', ckpt_s_pose)
                if dist is not None and ang is not None:
                    s_dists.append(dist)
                    s_angs.append(ang)
                    s_idxs.append(int(tokens['ckpt_image_idx']) * 30)

                q, t = parse_arpose(tokens['ckpt_c_abs_pose'])
                ckpt_c_pose = Pose(q, t)

                prev_c_pose, prev_s_pose, _, _ = ckpts.get(tokens['ckpt_image_idx'], accept_earlier=True)
                if fused_offline:
                    scale = float(tokens['cur_scale'])
                elif prev_s_pose is not None:
                    scale = float(tokens['cur_scale'])
                eval_scale = np.asarray(scale)

                ckpts.add(tokens['ckpt_image_idx'], ckpt_c_pose, ckpt_s_pose, scale, 1.0)

        # Load client sample
        sampledimgs_lines = read_file_lines(sampledimgs_fname)

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
                    # cam_poses.append((sampled_fused_pose, 'orange'))
                    cam_poses.append((sampled_fused_pose, colors[track_idx % len(colors)]))
                else:
                    # Sampled fused pose by offline calculation
                    q, t = parse_arpose(tokens['c_abs_pose'])
                    sampled_c_pose = Pose(q, t)
                    ckpt_c_pose, ckpt_s_pose, scale, confidence = ckpts.get(int(tokens['cur_image_idx']) - 1, accept_earlier=True)
                    sampled_fused_pose = calc_fused_pose(sampled_c_pose, ckpt_c_pose, ckpt_s_pose, scale)
                    # cam_poses.append((sampled_fused_pose, 'yellow'))
                    cam_poses.append((sampled_fused_pose, colors[track_idx % len(colors)]))

                # Evaluation on distance and angle
                dist, ang = gt_poses.compare(f'client-{tokens["cur_image_idx_stamp"]}', sampled_fused_pose)
                if dist is not None and ang is not None:
                    c_dists.append(dist)
                    c_angs.append(ang)
                    tmp_tok = tokens['cur_image_idx_stamp'].split('-')
                    c_idxs.append(int(tmp_tok[0]) * 30 + int(tmp_tok[1]))

    s_dists = np.array(s_dists) / eval_scale * 100
    c_dists = np.array(c_dists) / eval_scale * 100
    s_angs = np.array(s_angs)
    c_angs = np.array(c_angs)


    print(dataset, test_id)
    print(f'Sampled client poses are fused {"offline" if fused_offline else "online"}')
    # print(
    #     len(s_dists),
    #     f'{np.round(np.median(s_dists), 2)}/{np.round(np.median(s_angs), 2)}',
    #     f'{np.round(np.mean(s_dists), 2)}/{np.round(np.mean(s_angs), 2)}',
    #     f'{np.round(np.max(s_dists), 2)}/{np.round(np.max(s_angs), 2)}',
    #     f'{np.round(np.min(s_dists), 2)}/{np.round(np.min(s_angs), 2)}',
    # )
    # print(
    #     len(c_dists),
    #     f'{np.round(np.median(c_dists), 2)}/{np.round(np.median(c_angs), 2)}',
    #     f'{np.round(np.mean(c_dists), 2)}/{np.round(np.mean(c_angs), 2)}',
    #     f'{np.round(np.max(c_dists), 2)}/{np.round(np.max(c_angs), 2)}',
    #     f'{np.round(np.min(c_dists), 2)}/{np.round(np.min(c_angs), 2)}',
    # )

    print('Evaluation Report')
    print(dataset, test_ids)
    print(f'Sampled client poses are fused {"offline" if fused_offline else "online"}')
    print('eval_scale', eval_scale)
    print('dist (cm), angle (deg)')
    print('server images #', len(s_dists))
    print('client images #', len(c_dists))
    
    print('dist server median', np.median(s_dists))
    print('dist server mean', np.mean(s_dists))
    print('dist server max', np.max(s_dists))
    print('dist server min', np.min(s_dists))
    print('dist client median', np.median(c_dists))
    print('dist client mean', np.mean(c_dists))
    print('dist client max', np.max(c_dists))
    print('dist client min', np.min(c_dists))

    c_dists_2std = c_dists[c_dists < np.mean(c_dists) + 2 * np.std(c_dists)]
    print('dist client std', np.std(c_dists))
    print('dist client median (2 std)', np.median(c_dists_2std))
    print('dist client mean (2 std)', np.mean(c_dists_2std))
    print('dist client max (2 std)', np.max(c_dists_2std))
    print('dist client min (2 std)', np.min(c_dists_2std))

    print('angle server median', np.median(s_angs))
    print('angle server mean', np.mean(s_angs))
    print('angle server max', np.max(s_angs))
    print('angle server min', np.min(s_angs))
    print('angle client median', np.median(c_angs))
    print('angle client mean', np.mean(c_angs))
    print('angle client max', np.max(c_angs))
    print('angle client min', np.min(c_angs))

    c_angs_2std = c_angs[c_dists < np.mean(c_dists) + 2 * np.std(c_dists)]
    # print('angle client std', np.std(c_angs))
    print('angle client median (2 std)', np.median(c_angs_2std))
    print('angle client mean (2 std)', np.mean(c_angs_2std))
    print('angle client max (2 std)', np.max(c_angs_2std))
    print('angle client min (2 std)', np.min(c_angs_2std))

    c_dists_3std = c_dists[c_dists < np.mean(c_dists) + 3 * np.std(c_dists)]
    c_dists_4std = c_dists[c_dists < np.mean(c_dists) + 4 * np.std(c_dists)]
    c_dists_5std = c_dists[c_dists < np.mean(c_dists) + 5 * np.std(c_dists)]
    c_dists_6std = c_dists[c_dists < np.mean(c_dists) + 6 * np.std(c_dists)]
    c_dists_7std = c_dists[c_dists < np.mean(c_dists) + 7 * np.std(c_dists)]
    c_dists_8std = c_dists[c_dists < np.mean(c_dists) + 8 * np.std(c_dists)]
    c_dists_9std = c_dists[c_dists < np.mean(c_dists) + 9 * np.std(c_dists)]
    c_dists_10std = c_dists[c_dists < np.mean(c_dists) + 10 * np.std(c_dists)]

    #
    print()
    print('n_samples')
    print('total', len(c_dists))
    print('smaller than mean (2 std)', len(c_dists_2std), len(c_dists_2std) / len(c_dists))
    n_3std = len(c_dists_3std) - len(c_dists_2std)
    print('mean (2 std) ~ mean (3 std)', n_3std, n_3std / len(c_dists))
    n_4std = len(c_dists_4std) - len(c_dists_3std)
    print('mean (3 std) ~ mean (4 std)', n_4std, n_4std / len(c_dists))
    n_5std = len(c_dists_5std) - len(c_dists_4std)
    print('mean (4 std) ~ mean (5 std)', n_5std, n_5std / len(c_dists))
    n_6std = len(c_dists_6std) - len(c_dists_5std)
    print('mean (5 std) ~ mean (6 std)', n_6std, n_6std / len(c_dists))
    n_7std = len(c_dists_7std) - len(c_dists_6std)
    print('mean (6 std) ~ mean (7 std)', n_7std, n_7std / len(c_dists))
    n_8std = len(c_dists_8std) - len(c_dists_7std)
    print('mean (7 std) ~ mean (8 std)', n_8std, n_8std / len(c_dists))
    n_9std = len(c_dists_9std) - len(c_dists_8std)
    print('mean (8 std) ~ mean (9 std)', n_9std, n_9std / len(c_dists))
    n_10std = len(c_dists_10std) - len(c_dists_9std)
    print('mean (9 std) ~ mean (10 std)', n_10std, n_10std / len(c_dists))
    n_others = len(c_dists) - len(c_dists_10std)
    print('bigger than mean (10 std)', n_others, n_others / len(c_dists))
    #

    if display:
        display_pose(cam_poses, load_point_cloud(points3D_fname), only_pos=args.display_only_position)


if __name__ == '__main__':
    main()

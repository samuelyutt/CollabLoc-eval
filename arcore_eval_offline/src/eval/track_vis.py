import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import argparse
from pathlib import Path

from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--tracks', type=str, required=True, nargs='+')
parser.add_argument('--src', type=str, default='gt')
parser.add_argument('--offline_log_path', type=Path)
args = parser.parse_args()


# Configuration
model_name = args.model_name
tracks = args.tracks
colors = [
    'crimson', 'darkviolet', 'dodgerblue', 'c', 'darkslategray', 'olive',
    'darkgoldenrod', 'coral', 'maroon', 'fuchsia', 'slateblue', 'g', 
    'darkgoldenrod', 'sienna', 'violet', 'palegreen', 'teal', 'slategray'
]


# Path
model_path = '../../models'
arcore_log_path = '../../arcore_log'
gt_poses_path = '../../gt_poses'


fig, ax = initial_plot(f'{model_path}/{model_name}/points3D.txt')
ax.view_init(elev=-5, azim=-90, roll=15)
plt.draw()
plt.pause(0.2)

for track_idx, track in enumerate(tracks):
    if args.src == 'gt':
        # Display poses from gt_poses.txt (ground truth poses)
        with open(f'{arcore_log_path}/{track}/sampled_imgs_log.txt', 'r') as f:
            lines = f.readlines()

        samples = []
        for line in lines:
            tokens = line.split('|')
            samples.append(f'client-{tokens[2]}')

        gt_poses = load_gt_poses(f'{gt_poses_path}/{model_name}/{track}/gt_poses.txt')

        for image_name, pose in gt_poses.items():
            if image_name.split('.')[0] in samples:
                ax = display_pose(ax, pose, colors[track_idx % len(colors)], only_position=True)

    elif args.src == 'sampled_imgs_log':
        # Display poses from sampled_imgs_log.txt (online poses)
        for line in read_file_lines(f'{arcore_log_path}/{track}/sampled_imgs_log.txt'):
            tokens = parse_log_line(line)
            if len(tokens) < 1:
                continue
            if tokens['log_type'] == 'SampledPose':
                pose = pose_from_arpose_str(tokens['fused_world_pose'])
                ax = display_pose(ax, pose, colors[track_idx % len(colors)], only_position=True)
    
    elif args.src == 'offline_logs':
        # Display poses from {track}_logs.txt (offline poses)
        assert args.offline_log_path is not None, 'offline_log_path should be given'

        with open(args.offline_log_path / f'{track}_logs.txt', 'r') as f:
            lines = f.readlines()
        for line in lines[3:]:
            if 'client' == line[:6]:
                tokens = line.split('|')
                tok = tokens[1].split()
                q = [tok[0], tok[1], tok[2], tok[3]]
                t = [tok[4], tok[5], tok[6]]
                pose = Pose(q, t)
                ax = display_pose(ax, pose, colors[track_idx % len(colors)], only_position=True)

    else:
        print('Unrecognized type')

    plt.draw()

plt.show()

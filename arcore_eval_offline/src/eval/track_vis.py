import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import argparse

from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--tracks', type=str, required=True, nargs='+')
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

    plt.draw()

plt.show()

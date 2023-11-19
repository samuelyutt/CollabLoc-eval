import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils import *

# Configuration
model_name = 'nycu_hist_museum'
tracks = ['2023-00-00_00-00-00']
colors = ['crimson', 'darkviolet', 'dodgerblue', 'c', 'darkslategray', 'olive', 'darkgoldenrod', 'coral', 'maroon', 'fuchsia', 'slateblue', 'g']

# Path
model_path = '../../models'
arcore_log_path = '../../arcore_log'
gt_poses_path = '../../gt_poses'


fig, ax = initial_plot(f'{model_path}/{model_name}/points3D.txt')
ax.view_init(elev=-5, azim=-90, roll=15)
plt.draw()
plt.pause(0.2)

for track_idx, track in enumerate(tracks):
    gt_poses = load_gt_poses(f'{gt_poses_path}/{model_name}/{track}/gt_poses.txt')

    for image_name, pose in gt_poses.items():
        if 'client' in image_name:
            ax = display_pose(ax, pose, colors[track_idx], only_position=True)

    plt.draw()

plt.show()

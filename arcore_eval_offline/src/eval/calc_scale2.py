import matplotlib.style as mplstyle
mplstyle.use('fast')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from utils import *

# Configuration
model_name = 'nycu_hist_museum'
display = True

# Pose 1
pose1_arpose_str = 't:[x:1.163, y:0.071, z:-0.925], q:[x:-0.95, y:0.01, z:-0.31, w:-0.06]'
pose1_gtpose_str = '573 0.974966 0.0233523 -0.220845 -0.0111344 -0.714435 -0.521791 5.30177 50 client-2-180.jpg'

# Pose 2
pose2_arpose_str = 't:[x:21.528, y:0.004, z:-31.368], q:[x:-0.86, y:0.05, z:-0.50, w:-0.09]'
pose2_gtpose_str = '772 0.906685 0.0602879 -0.414428 -0.0503773 1.53877 0.818292 -4.1871 248 client-44-24.jpg'

# Path
model_path = '../../models'


def main():
    if display:
        fig, ax = initial_plot(f'{model_path}/{model_name}/points3D.txt')
    gtposes = []
    for pose_str in [pose1_gtpose_str, pose2_gtpose_str]:
        # Tokens format
        # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        tokens = pose_str.split()
        q = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4])])
        t = np.array([float(tokens[5]), float(tokens[6]), float(tokens[7])])
        pose = Pose(q, t, 'colmap')
        gtposes.append(pose)
        if display:
            ax = display_pose(ax, pose, 'red')
    gtposes_dist = gtposes[0].dist_to(gtposes[1])

    arposes = []
    for pose_str in [pose1_arpose_str, pose2_arpose_str]:
        pose = pose_from_arpose_str(pose_str)
        arposes.append(pose)
        # ax = display_pose(ax, pose, 'red')
    arposes_dist = arposes[0].dist_to(arposes[1])

    print('dist (arcore)', arposes_dist)
    print('dist (colmap)', gtposes_dist)
    print('scale', gtposes_dist / arposes_dist) # dist_colmap / dist_meter

    if display:
        plt.draw()
        plt.show()


if __name__ == '__main__':
    main()

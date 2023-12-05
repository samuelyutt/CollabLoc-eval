import matplotlib.style as mplstyle
mplstyle.use('fast')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import argparse

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--display', action='store_true')
parser.add_argument('--pose1_arpose', type=str, required=True)
parser.add_argument('--pose1_gtpose', type=str, required=True)
parser.add_argument('--pose2_arpose', type=str, required=True)
parser.add_argument('--pose2_gtpose', type=str, required=True)
args = parser.parse_args()


# Configuration
model_name = args.model_name
display = args.display
pose1_arpose_str = args.pose1_arpose
pose1_gtpose_str = args.pose1_gtpose
pose2_arpose_str = args.pose2_arpose
pose2_gtpose_str = args.pose2_gtpose


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

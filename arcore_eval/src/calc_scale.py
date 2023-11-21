import matplotlib.style as mplstyle
mplstyle.use('fast')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import argparse

from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--dist_meter', type=float, required=True)
parser.add_argument('--pose1_gtpose', type=str, required=True)
parser.add_argument('--pose2_gtpose', type=str, required=True)
args = parser.parse_args()


dataset = args.model_name
dist_meter = args.dist_meter

points3D_fname = f'../data/{dataset}/model/points3D.txt'


def main():
    # Load ground truth
    lines = [
        args.pose1_gtpose,
        args.pose2_gtpose,
    ]

    poses = []
    cam_poses = []

    for line_idx, line in enumerate(lines):
        # Tokens format
        # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        tokens = line.split()
        if len(tokens) < 1:
            continue
        q = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4])])
        t = np.array([float(tokens[5]), float(tokens[6]), float(tokens[7])])
        gt_pose = Pose(q, t, 'colmap')
        poses.append(gt_pose)
        cam_poses.append((gt_pose, 'red' if line_idx == 0 else 'blue'))
        print(gt_pose.to_arcore_pose_str())

    dist = calc_dist(poses[0].pos(), poses[1].pos())

    print('dist (meter)', dist_meter)
    print('dist (colmap)', dist)
    print('scale', dist / dist_meter)

    points_fname = f'../data/{dataset}/model/points3D.txt'
    display_pose(cam_poses, load_point_cloud(points3D_fname))


if __name__ == '__main__':
    main()

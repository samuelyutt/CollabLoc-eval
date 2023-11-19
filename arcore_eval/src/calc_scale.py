import matplotlib.style as mplstyle
mplstyle.use('fast')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from utils import *

dataset = 'example_dataset'
dist_meter = 6.92

points3D_fname = f'../data/{dataset}/model/points3D.txt'


def main():
    # Load ground truth
    lines = [
        '761 0.27163 -0.0438954 0.957457 0.0869847 -1.07724 -0.30125 1.18877 2 add_P_20230515_150621.jpg',
        '760 0.618076 -0.0626705 0.781529 0.0571548 1.72471 -0.322014 1.63988 2 add_P_20230515_150601.jpg',
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

import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import numpy as np
import numpy.linalg
import argparse
from pathlib import Path

from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--test_id', type=str, required=True)
parser.add_argument('--display', action='store_true')
parser.add_argument('--display_only_position', action='store_true')
parser.add_argument('--offline_log_path', type=Path)
args = parser.parse_args()


# Path
model_path = '../../models'
arcore_log_path = '../../arcore_log'
gt_poses_path = '../../gt_poses'
onsite_poses_path = '../../onsite_poses'


# Function `umeyama` calculates: relative scale, R, and t between the given two point clouds
# Referenced to function `rigid-transform-with-scale.py` from nh2:
# https://gist.github.com/nh2/bc4e2981b0e213fefd4aaa33edfb3893
def umeyama(P, Q):
    # Relevant links:
    #   - http://stackoverflow.com/a/32244818/263061 (solution with scale)
    #   - "Least-Squares Rigid Motion Using SVD" (no scale but easy proofs and explains how weights could be added)

    # Rigidly (+scale) aligns two point clouds with know point-to-point correspondences
    # with least-squares error.
    # Returns (scale factor c, rotation matrix R, translation vector t) such that
    #   Q = P*cR + t
    # if they align perfectly, or such that
    #   SUM over point i ( | P_i*cR + t - Q_i |^2 )
    # is minimised if they don't align perfectly.

    assert P.shape == Q.shape
    n, dim = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    C = np.dot(np.transpose(centeredP), centeredQ) / n

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)

    varP = np.var(P, axis=0).sum()
    c = 1/varP * np.sum(S) # scale factor

    t = Q.mean(axis=0) - P.mean(axis=0).dot(c*R)

    return c, R, t


def example():
    # Testing
    np.set_printoptions(precision=3)

    a1 = np.array([
        [0, 0, -1],
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ])

    a2 = np.array([
        [0, 0, 1],
        [0, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
        [-1, 0, 0],
    ])
    a2 *= 2 # for testing the scale calculation
    a2 += 3 # for testing the translation calculation

    c, R, t = umeyama(a1, a2)
    print("R =\n", R)
    print("c =", c)
    print("t =\n", t)
    print()
    print("Check:  a1*cR + t = a2  is", np.allclose(a1.dot(c*R) + t, a2))
    err = ((a1.dot(c * R) + t - a2) ** 2).sum()
    print("Residual error", err)


def main():
    # Visualize
    if args.display:
        fig, ax = initial_plot()
        ax.view_init(elev=-5, azim=-90, roll=15)
        plt.draw()
        plt.pause(0.2)

    # Read onsite poses
    with open(f'{onsite_poses_path}/{args.model_name}/{args.test_id}/onsite_poses.txt', 'r') as f:
        lines = f.readlines()
    onsite_poses = {}
    for line in lines:
        tokens = line.split()
        q = [tokens[0], tokens[1], tokens[2], tokens[3]]
        t = [tokens[4], tokens[5], tokens[6]]
        onsite_pose = Pose(q, t)
        onsite_poses[tokens[7].replace('client-', '').split('.')[0]] = onsite_pose
        if args.display:
            ax = display_pose(ax, onsite_pose, 'green', only_position=args.display_only_position)

    # Read client fused poses
    client_poses = {}
    if args.offline_log_path:
        # Read client fused poses from offline logs
        with open(args.offline_log_path / f'{args.test_id}_logs.txt', 'r') as f:
            lines = f.readlines()
        for line in lines[3:]:
            if 'client' == line[:6]:
                tokens = line.split('|')
                tok = tokens[1].split()
                q = [tok[0], tok[1], tok[2], tok[3]]
                t = [tok[4], tok[5], tok[6]]
                client_pose = Pose(q, t)
                client_poses[tokens[5].replace('client-', '').split('.')[0]] = client_pose
                if args.display:
                    ax = display_pose(ax, client_pose, 'blue', only_position=args.display_only_position)
        
    else:
        # Read client fused poses from online logs
        samples = read_sampled_imgs_log(f'{arcore_log_path}/{args.test_id}/sampled_imgs_log.txt')
        for sample in samples:
            if sample['cur_image_idx_stamp'] in onsite_poses:
                # client_pose = pose_from_arpose_str(sample['c_abs_pose']) # only for pre-testing, must be fused_world_pose
                client_pose = pose_from_arpose_str(sample['fused_world_pose'])
                client_poses[sample['cur_image_idx_stamp']] = client_pose
                if args.display:
                    ax = display_pose(ax, client_pose, 'blue', only_position=args.display_only_position)

    # Remove redundant poses because they do not appear in samples
    redundant_keys = []
    for key in onsite_poses.keys():
        if key not in client_poses:
            redundant_keys.append(key)
    print(f'client-{redundant_keys} does not exist')
    for key in redundant_keys:
        del onsite_poses[key]

    # a1 (client pose) -> a2 (onsite pose)
    a1 = np.array([client_pose.t() for _, client_pose in client_poses.items()])
    a2 = np.array([onsite_pose.t() for _, onsite_pose in onsite_poses.items()])

    # Get transformation scale, R, t
    # a1*cR + t = a2
    c, R, t = umeyama(a1, a2)

    dists, angles = [], []
    for key, client_pose in client_poses.items():
        # Transformation
        tsfm_client_pose_q = quaternion.from_rotation_matrix(np.matmul(
            R,
            quaternion.as_rotation_matrix(client_pose.q()),
        ))
        tsfm_client_pose_t = np.matmul(
            client_pose.t(),
            R,
        ) * c + t
        tsfm_client_pose = Pose(quaternion.as_float_array(tsfm_client_pose_q), tsfm_client_pose_t)
        
        # Error calculation
        onsite_pose = onsite_poses[key]
        dist = onsite_pose.dist_to(tsfm_client_pose)
        angle = onsite_pose.angle_to(tsfm_client_pose)
        dists.append(dist)
        angles.append(angle)

        # Display
        if args.display:
            ax = display_pose(ax, tsfm_client_pose, 'purple', only_position=args.display_only_position)

    # Statistics
    dists = np.array(dists)
    angles = np.array(angles)

    print(f'Client poses are fused {"offline" if args.offline_log_path else "online"}')
    print('scale (fused -> onsite)', c, '(onsite -> fused)', 1 / c)
    print('dists in meter, angles in degree')
    print('n_samples', len(dists))
    print('median', np.median(dists), np.median(angles))
    print('mean', np.mean(dists), np.mean(angles))

    # Display
    if args.display:
        plt.draw()
        plt.show()


def create_tmp_file():
    scale = 0.2693774583633444
    gt_poses = load_gt_poses(f'{gt_poses_path}/{args.model_name}/{args.test_id}/gt_poses.txt')
    gt_poses_list = []
    for key, pose in gt_poses.items():
        if 'client' in key:
            tokens = key.split('.')[0].split('-')
            pose.x /= scale
            pose.y /= scale
            pose.z /= scale
            gt_poses_list.append((int(tokens[1]), int(tokens[2]), f"{pose.to_formatted_str('%qw %qx %qy %qz %tx %ty %tz')} {key}\n"))
    gt_poses_list.sort()
    
    with open(f'{onsite_poses_path}/{args.model_name}/{args.test_id}/onsite_poses.txt', 'w') as f:
        for _, _, tmp_str in gt_poses_list:
            f.writelines(tmp_str)


if __name__ == '__main__':
    main()
    # create_tmp_file()
    # example()
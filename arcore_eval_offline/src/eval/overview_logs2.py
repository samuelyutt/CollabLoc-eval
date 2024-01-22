import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt

import os
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=Path, required=True)
parser.add_argument('--display', action='store_true')
parser.add_argument('--display_dist_req', type=float, default=10)
parser.add_argument('--display_angle_req', type=float, default=2)
args = parser.parse_args()


def main():

    dists_dict, angles_dict = {}, {}
    dists_all, angles_all = [], []
    cur_scale = 1.0

    for file_name in os.listdir(args.path):
        if 'logs.txt' in file_name:
            with open(args.path / file_name, 'r') as f:
                lines = f.readlines()
            
            dists, angles = [], []
            for line in lines[:3]:
                if 'curScale' == line[:8]:
                    cur_scale = float(line.replace('curScale=', ''))

            for line in lines[3:]:
                if 'client' == line[:6]:
                    tokens = line.split('|')
                    dist = np.float32(tokens[3])
                    dists.append(dist / cur_scale)
                    angle = np.float32(tokens[4])
                    angles.append(angle)

            name = file_name.replace('_logs.txt', '')
            dists_dict[name] = np.array(dists)
            dists_all += dists
            angles_dict[name] = np.array(angles)
            angles_all += angles
    
    dists_all = np.array(dists_all)
    angles_all = np.array(angles_all)

    print('dists in meter')
    print('n_clients', len(dists_dict))
    print('n_samples', len(dists_all), len(angles_all))
    print('median', np.median(dists_all), np.median(angles_all))
    
    dist_mean = np.mean(dists_all)
    angle_mean = np.mean(angles_all)
    dist_std = np.std(dists_all)
    print('mean', dist_mean, angle_mean)
    print('dist_std', dist_std)

    # dists_3std = dists_all[dists_all < dist_mean + 3 * dist_std]
    dists_2std = dists_all[dists_all < dist_mean + 2 * dist_std]
    # dists_1std = dists_all[dists_all < dist_mean + 1 * dist_std]
    # angles_3std = angles_all[dists_all < dist_mean + 3 * dist_std]
    angles_2std = angles_all[dists_all < dist_mean + 2 * dist_std]
    # angles_1std = angles_all[dists_all < dist_mean + 1 * dist_std]
    # print('mean (3 std)', np.mean(dists_3std), np.mean(angles_3std), len(dists_3std), len(dists_3std) / len(dists_all))
    print('mean (2 std)', np.mean(dists_2std), np.mean(angles_2std), len(dists_2std), len(dists_2std) / len(dists_all))
    # print('mean (1 std)', np.mean(dists_1std), np.mean(angles_1std), len(dists_1std), len(dists_1std) / len(dists_all))

    print('scale', cur_scale)

    #
    dists_3std = dists_all[dists_all < dist_mean + 3 * dist_std]
    dists_4std = dists_all[dists_all < dist_mean + 4 * dist_std]
    dists_5std = dists_all[dists_all < dist_mean + 5 * dist_std]
    dists_6std = dists_all[dists_all < dist_mean + 6 * dist_std]
    dists_7std = dists_all[dists_all < dist_mean + 7 * dist_std]
    dists_8std = dists_all[dists_all < dist_mean + 8 * dist_std]
    dists_9std = dists_all[dists_all < dist_mean + 9 * dist_std]
    dists_10std = dists_all[dists_all < dist_mean + 10 * dist_std]

    print()
    print('n_samples')
    print('total', len(dists_all))
    print('smaller than mean (2 std)', len(dists_2std), len(dists_2std) / len(dists_all))
    n_3std = len(dists_3std) - len(dists_2std)
    print('mean (2 std) ~ mean (3 std)', n_3std, n_3std / len(dists_all))
    n_4std = len(dists_4std) - len(dists_3std)
    print('mean (3 std) ~ mean (4 std)', n_4std, n_4std / len(dists_all))
    n_5std = len(dists_5std) - len(dists_4std)
    print('mean (4 std) ~ mean (5 std)', n_5std, n_5std / len(dists_all))
    n_6std = len(dists_6std) - len(dists_5std)
    print('mean (5 std) ~ mean (6 std)', n_6std, n_6std / len(dists_all))
    n_7std = len(dists_7std) - len(dists_6std)
    print('mean (7 std) ~ mean (6 std)', n_7std, n_7std / len(dists_all))
    n_8std = len(dists_8std) - len(dists_7std)
    print('mean (8 std) ~ mean (7 std)', n_8std, n_8std / len(dists_all))
    n_9std = len(dists_9std) - len(dists_8std)
    print('mean (9 std) ~ mean (8 std)', n_9std, n_9std / len(dists_all))
    n_10std = len(dists_10std) - len(dists_9std)
    print('mean (10 std) ~ mean (9 std)', n_10std, n_10std / len(dists_all))
    n_others = len(dists_all) - len(dists_10std)
    print('bigger than mean (10 std)', n_others, n_others / len(dists_all))
    #

    if args.display:
        fig = plt.figure()

        # Draw dist error
        dists_2std_cm = dists_2std * 100
        len_dists_2std_cm = len(dists_2std_cm)
        mean_dists_2std_cm = np.mean(dists_2std_cm)
        dist_reqirement_cm = args.display_dist_req
        lower_dists_2std_cm = np.quantile(dists_2std_cm, 0.25)
        mid_dists_2std_cm = np.quantile(dists_2std_cm, 0.5)
        higher_dists_2std_cm = np.quantile(dists_2std_cm, 0.75)

        ax = fig.add_subplot(1, 2, 1)
        ax.figure.set_figwidth(15)
        ax.set_ylim([0, max(np.quantile(dists_2std_cm, 0.90), higher_dists_2std_cm, mean_dists_2std_cm, dist_reqirement_cm + 5)])
        ax.hlines(higher_dists_2std_cm, 0, len_dists_2std_cm, 'gray', label=f'75%={higher_dists_2std_cm}cm')
        ax.hlines(mid_dists_2std_cm, 0, len_dists_2std_cm, 'orange', label=f'50%={mid_dists_2std_cm}cm')
        ax.hlines(lower_dists_2std_cm, 0, len_dists_2std_cm, 'gray', label=f'25%={lower_dists_2std_cm}cm')
        ax.hlines(mean_dists_2std_cm, 0, len_dists_2std_cm, 'red', label=f'mean={mean_dists_2std_cm}cm')
        ax.hlines(dist_reqirement_cm, 0, len_dists_2std_cm, 'green', label=f'requirement={dist_reqirement_cm}cm')
        ax.plot([i for i in range(len_dists_2std_cm)], np.sort(dists_2std_cm), '.', markersize=0.5, label='Fused pose dist error')
        ax.legend()
        ax.set_xlabel('#')
        ax.set_ylabel('error (cm)')
        ax.set_title('Dist error')

        # Draw angle error
        len_angles_2std = len(angles_2std)
        mean_angles_2std = np.mean(angles_2std)
        angle_reqirement = args.display_angle_req
        lower_angles_2std = np.quantile(angles_2std, 0.25)
        mid_angles_2std = np.quantile(angles_2std, 0.5)
        higher_angles_2std = np.quantile(angles_2std, 0.75)

        ax = fig.add_subplot(1, 2, 2)
        ax.set_ylim([0, max(np.quantile(angles_2std, 0.90), higher_angles_2std, mean_angles_2std, angle_reqirement + 1)])
        ax.hlines(higher_angles_2std, 0, len_angles_2std, 'gray', label=f'75%={higher_angles_2std}deg')
        ax.hlines(mid_angles_2std, 0, len_angles_2std, 'orange', label=f'50%={mid_angles_2std}deg')
        ax.hlines(lower_angles_2std, 0, len_angles_2std, 'gray', label=f'25%={lower_angles_2std}deg')
        ax.hlines(mean_angles_2std, 0, len_angles_2std, 'red', label=f'mean={mean_angles_2std}deg')
        ax.hlines(2, 0, len_dists_2std_cm, 'green', label='requirement=2deg')
        ax.plot([i for i in range(len_angles_2std)], np.sort(angles_2std), '.', markersize=0.5, label='Fused pose angle error')
        ax.legend()
        ax.set_xlabel('#')
        ax.set_ylabel('error (deg)')
        ax.set_title('Angle error')

        plt.draw()
        plt.show()


if __name__ == '__main__':
    main()

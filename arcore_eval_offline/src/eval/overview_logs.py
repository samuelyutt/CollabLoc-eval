import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt

import os
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=Path, required=True)
args = parser.parse_args()


def main():

    dists_dict = {}
    dists_all = []
    cur_scale = 1.0

    for file_name in os.listdir(args.path):
        if 'logs.txt' in file_name:
            with open(args.path / file_name, 'r') as f:
                lines = f.readlines()
            
            dists = []
            for line in lines[:3]:
                if 'curScale' == line[:8]:
                    cur_scale = float(line.replace('curScale=', ''))

            for line in lines[3:]:
                if 'client' == line[:6]:
                    tokens = line.split('|')
                    dist = np.float32(tokens[3])
                    dists.append(dist / cur_scale)

            dists_dict[file_name.replace('_logs.txt', '')] = np.array(dists)
            dists_all += dists
    
    dists_all = np.array(dists_all)

    print('dists in meter')
    print('n_clients', len(dists_dict))
    print('n_samples', len(dists_all))
    print('median', np.median(dists_all))
    
    dist_mean = np.mean(dists_all)
    dist_std = np.std(dists_all)
    print('mean', dist_mean)
    print('std', dist_std)

    dists_1std = dists_all[dists_all < dist_mean + 1 * dist_std]
    dists_2std = dists_all[dists_all < dist_mean + 2 * dist_std]
    dists_3std = dists_all[dists_all < dist_mean + 3 * dist_std]
    print('mean (3 std)', np.mean(dists_3std), len(dists_3std), len(dists_3std) / len(dists_all))
    print('mean (2 std)', np.mean(dists_2std), len(dists_2std), len(dists_2std) / len(dists_all))
    print('mean (1 std)', np.mean(dists_1std), len(dists_1std), len(dists_1std) / len(dists_all))

    print('scale', cur_scale)


if __name__ == '__main__':
    main()
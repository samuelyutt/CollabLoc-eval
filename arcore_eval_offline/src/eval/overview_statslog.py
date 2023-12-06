import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt

import numpy as np

def parse_log(log_line):
    tokens = log_line.split()
    return {
        'n': int(tokens[1]),
        'median': float(tokens[3]),
        'mean': float(tokens[5]),
        # 'min': float(tokens[7]),
        # 'max': float(tokens[9]),
    }

def main():
    # methods = ['CollabLoc', 'HFNet']
    # n_clients = [1, 2, 5, 8, 12]
    methods = ['CollabLoc_new']
    methods = ['CollabLoc']
    model_name = 'nycu_hist_museum2_5'
    n_clients = [12]
    tracks = ['1', '23-1', '22-1', '23-3', '24-1', '3', '4', '5', '2', '6', '22-3', '23-2']
    # methods = ['CollabLoc']
    # n_clients = [1]
    # tracks = ['1']

    stats = {}
    for n_client in n_clients:
        stats[n_client] = {}

        for method in methods:
            stats[n_client][method] = {}

            observations = ['c_dist']
            # observations = ['c_dist', 'c_angle', 's_dist', 's_angle']
            # observations = ['c_dist', 'c_angle', 's_dist', 's_angle', 'f_dist', 'f_angle']
            # observations = ['c_dist', 'c_angle']

            for i, observation in enumerate(observations):
                stats[n_client][method][observation] = {}

                observe_line_idx = 3 + i * 2
                
                for track_idx in range(n_client):
                    track = tracks[track_idx]
                    statslog_path = f'./out_{method}/{model_name}/{n_client}_clients/{track}_statslog.txt'
                    with open(statslog_path, 'r') as f:
                        lines = f.readlines()

                    if observe_line_idx < len(lines):
                        for key, value in parse_log(lines[observe_line_idx]).items():
                            if key in stats[n_client][method][observation]:
                                stats[n_client][method][observation][key].append(value)
                            else:
                                stats[n_client][method][observation][key] = [value]
                
                for key in stats[n_client][method][observation]:
                    stats[n_client][method][observation][key] = np.array(stats[n_client][method][observation][key])

    print(stats)

    for n_client, n_client_val in stats.items():
        for method, method_val in n_client_val.items():
            for observation, observation_val in method_val.items():
                for key, value in observation_val.items():
                    value = np.array(value)
                    # print(n_client, method, observation, key, value, end=' ')

                    if key == 'n':
                        print(f'{method.ljust(10, " ")} ({n_client}c) {observation} {key.ljust(6, " ")}: {value}')
                    else:
                        avg = np.dot(observation_val['n'], value) / np.sum(observation_val['n'])
                        print(f'{method.ljust(10, " ")} ({n_client}c) {observation} {key.ljust(6, " ")}: {value} {avg}')
        print()

if __name__ == '__main__':
    main()
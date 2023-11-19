import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from datetime import datetime
import threading
import socket
import time
import cv2
import numpy as np
import argparse
from pathlib import Path

from utils import *

# Connection
SERVER_ADDRESS = '140.113.195.248'
SERVER_PORT = 9999

# Configuration
model_name = 'nycu_hist_museum'
focal_length = '  423.4869   422.5935'
display = False

# Path
model_path = '../../models'
arcore_log_path = '../../arcore_log'
gt_poses_path = '../../gt_poses'


parser = argparse.ArgumentParser()
parser.add_argument('--test_id', type=str)
parser.add_argument('--out', type=Path, default='out')
args = parser.parse_args()

def sendCurTrackConfThread():
    global s, curTrackConf, cur_arcore_sample, is_finished, cur_fused_pose
    global stats, logs, t0
    
    s.connect((SERVER_ADDRESS, SERVER_PORT))

    t_in = threading.Thread(target=socketThreadIn)
    t_in.start()

    while not is_finished:
        if not sendingQueryImage and cur_arcore_sample is not None:
            packetCurTrackConf = '{:010.4f}\0'.format(curTrackConf).encode()
            s.send(packetCurTrackConf)
            print(f'send curTrackConf {curTrackConf}')

            if cur_fused_pose is not None:
                # Penalty to confidence by tracking length
                prevAvailCheckPoint = checkPointsMap[latestAvailServerResultIdx]
                prevCkptServerPose = prevAvailCheckPoint['s_pose']
                distFused2PrevSSL = cur_fused_pose.dist_to(prevCkptServerPose) / curScale
                angleFused2PrevSSL = cur_fused_pose.dist_to(prevCkptServerPose)

                curTrackConf -= distFused2PrevSSL * 0.005
                curTrackConf -= angleFused2PrevSSL * 0.0003
                curTrackConf = min(max(curTrackConf, 0.0), 1.0)

        time.sleep(0.1)

    print('sendCurTrackConfThread ended')


def socketThreadIn():
    global s, curTrackConf, sendingQueryImage, curScale, imageIdx, cur_arcore_sample, curImage, diplay_serverPose, display_image, serverPose, cur_fused_pose
    global checkPointsMap, latestAvailServerResultIdx, isFusedPoseReady, is_finished
    global gt_poses, stats, logs, t0

    def ssl_filter(ckptServerPose, oldWorldPose):
        global curTrackConf
        
        if oldWorldPose is None:
            return True
        
        ServerResultIsAvailable = False
        distFused2SSL = oldWorldPose.dist_to(ckptServerPose) / curScale
        angleFused2SSL = oldWorldPose.angle_to(ckptServerPose)

        if (distFused2SSL <= 0.05 and angleFused2SSL <= 2):
            curTrackConf += 0.2
            ServerResultIsAvailable = True
        elif (distFused2SSL <= 0.1 and angleFused2SSL <= 4):
            curTrackConf += 0.1
            ServerResultIsAvailable = True
        elif (distFused2SSL <= 0.2 and angleFused2SSL <= 4):
            curTrackConf += 0.05
            ServerResultIsAvailable = True
        elif (distFused2SSL <= 0.3 and angleFused2SSL <= 6):
            curTrackConf += 0.0
            ServerResultIsAvailable = True
        elif (distFused2SSL <= 0.4 and angleFused2SSL <= 8):
            curTrackConf -= 0.1
        elif (distFused2SSL > 0.8 or angleFused2SSL > 10):
            curTrackConf -= 0.2
        elif (distFused2SSL > 0.6 or angleFused2SSL > 8):
            curTrackConf -= 0.2
        elif (distFused2SSL > 0.4 or angleFused2SSL > 6):
            curTrackConf -= 0.1

        curTrackConf = min(max(curTrackConf, 0.0), 1.0)

        return ServerResultIsAvailable

    while not is_finished:
        serverCmd = s.recv(10).decode()
        print('serverCmd', serverCmd)

        if cur_arcore_sample is None:
            continue

        if (serverCmd == 'send_query'):
            sendingQueryImage = True
            imageIdx += 1
            
            cur_image_idx_stamp = cur_arcore_sample["cur_image_idx_stamp"]
            curImage = cv2.imread(f'{arcore_log_path}/{args.test_id}/sampled_imgs/client-{cur_image_idx_stamp}.jpg')
            is_success, curImage_buf_arr = cv2.imencode(".jpg", curImage)
            packetImage = curImage_buf_arr.tobytes()
            display_image = True

            # cur_fused_pose = 'null'
            cur_fused_pose_str = cur_fused_pose.to_formatted_str('t:[x:%tx, y:%ty, z:%tz], q:[x:%qx, y:%qy, z:%qz, w:%qw]') if cur_fused_pose is not None else 'null'

            packetMetaData = f'img_size={len(packetImage)}|frame_idx={imageIdx}|focal_length={focal_length}|cur_fused_pose={cur_fused_pose_str}|cur_track_conf={curTrackConf}\0'.encode()
            packetMetaDataSize = '{:010d}\0'.format(len(packetMetaData)).encode()

            s.send('send_query\0'.encode())
            s.send(packetMetaDataSize)
            s.send(packetMetaData)
            s.send(packetImage)

            print(f'Query {imageIdx} sended {cur_image_idx_stamp}')

            checkpoint = {
                'imageIdx': imageIdx,
                'c_pose': pose_from_arpose_str(cur_arcore_sample['c_abs_pose']),
                'fused_pose': cur_fused_pose,
                'image_name': f'client-{cur_image_idx_stamp}.jpg',
                'timestamp': time.monotonic() - t0,
            }
            checkPointsMap[imageIdx] = checkpoint

            sendingQueryImage = False

        elif (serverCmd == 'result_mtx'):
            mtx = np.zeros(shape=(4, 4), dtype=float)
            for i in range(16):
                tmpStr = s.recv(18).decode()
                tmpStr = tmpStr[4:15]
                mtx[int(i / 4)][int(i % 4)] = float(tmpStr)

            serverPose = Pose(qvecFromRTMtx(mtx), tvecFromRTMtx(mtx), 'colmap')

            imageIdxStr = s.recv(10).decode()
            latestServerResultIdx = int(imageIdxStr)
            
            s_pose_is_available = False
            if not isFusedPoseReady or curTrackConf < 0.2:
                curTrackConf = 0.7
                s_pose_is_available = True
            elif ssl_filter(serverPose, checkPointsMap[latestServerResultIdx]['fused_pose']):
                s_pose_is_available = True
            # s_pose_is_available = True

            image_name = checkPointsMap[latestServerResultIdx]['image_name']
            
            if image_name in gt_poses:
                gt_pose = gt_poses[image_name]

                if s_pose_is_available:
                    checkPointsMap[latestServerResultIdx]['s_pose'] = serverPose
                    latestAvailServerResultIdx = latestServerResultIdx
                    isFusedPoseReady = True

                    timestamp = checkPointsMap[latestServerResultIdx]['timestamp']
                    dist = gt_pose.dist_to(serverPose)
                    angle = gt_pose.angle_to(serverPose)
                    stats['server']['time'].append(timestamp)
                    stats['server']['dist'].append(dist)
                    stats['server']['angle'].append(angle)
                    diplay_serverPose = 'red'
                    logs.append(f'server|{serverPose.to_formatted_str("%qw %qx %qy %qz %tx %ty %tz")}|{timestamp}|{dist}|{angle}|{image_name}')
                    logs.append(f'gt|{gt_pose.to_formatted_str("%qw %qx %qy %qz %tx %ty %tz")}|{timestamp}|||{image_name}')
                else:
                    timestamp = checkPointsMap[latestServerResultIdx]['timestamp']
                    dist = gt_pose.dist_to(serverPose)
                    angle = gt_pose.angle_to(serverPose)
                    stats['filtered']['time'].append(timestamp)
                    stats['filtered']['dist'].append(dist)
                    stats['filtered']['angle'].append(angle)
                    diplay_serverPose = 'black'
                    logs.append(f'filtered|{serverPose.to_formatted_str("%qw %qx %qy %qz %tx %ty %tz")}|{timestamp}|{dist}|{angle}|{image_name}')
                    logs.append(f'gt|{gt_pose.to_formatted_str("%qw %qx %qy %qz %tx %ty %tz")}|{timestamp}|||{image_name}')

        elif (serverCmd == 'curr_scale'):
            tmpStr = s.recv(20).decode()
            curScale = float(tmpStr)
    
    print('socketThreadIn ended')


if __name__ == '__main__':
    global cur_arcore_sample, curImage, diplay_serverPose, display_image, serverPose, curScale, cur_fused_pose
    global checkPointsMap, latestAvailServerResultIdx, isFusedPoseReady, is_finished
    global gt_poses, stats, logs, t0

    curImage = None
    diplay_serverPose = False
    display_image = False
    samples = read_sampled_imgs_log(f'{arcore_log_path}/{args.test_id}/sampled_imgs_log.txt')
    sendingQueryImage = False
    imageIdx = 0
    curTrackConf = 0.0
    curScale = 1.0
    cur_fused_pose = None
    cur_arcore_sample = None
    checkPointsMap = {}
    isFusedPoseReady = False
    is_finished = False
    stats = {
        'server': {'time': [], 'dist': [], 'angle': []},
        'client': {'time': [], 'dist': [], 'angle': []},
        'filtered': {'time': [], 'dist': [], 'angle': []},
    }
    logs = []

    # Socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Threads
    t_conf = threading.Thread(target=sendCurTrackConfThread)
    t_conf.start()
    
    # Visualize
    if display:
        fig, ax = initial_plot(f'{model_path}/{model_name}/points3D.txt')
        ax.view_init(elev=-5, azim=-90, roll=15)
        plt.draw()
        plt.pause(0.2)

    gt_poses = load_gt_poses(f'{gt_poses_path}/{model_name}/{args.test_id}/gt_poses.txt')

    # Main process
    t0 = time.monotonic()

    # for sample in samples:
    for sample in samples:
        t1 = time.monotonic()
        cur_arcore_sample = sample
        
        # if display:
        #     if display_image and curImage is not None:
        #         display_image = False
        #         cv2.imshow('curImage', curImage)

        # tmp_image = cv2.imread(f'{arcore_log_path}/{args.test_id}/sampled_imgs/client-{cur_arcore_sample["cur_image_idx_stamp"]}.jpg')
        # cv2.imshow('tmp_image', tmp_image)

        if display:
            if diplay_serverPose != False:
                ax = display_pose(ax, serverPose, diplay_serverPose)
                plt.draw()
                # plt.pause(0.05)
                diplay_serverPose = False

        if isFusedPoseReady and cur_arcore_sample is not None:
            cur_client_pose = pose_from_arpose_str(cur_arcore_sample['c_abs_pose'])
            latestAvailServerResult = checkPointsMap[latestAvailServerResultIdx]
            cur_fused_pose = calc_fused_pose(cur_client_pose, latestAvailServerResult['c_pose'], latestAvailServerResult['s_pose'], curScale)
            if display:
                ax = display_pose(ax, cur_fused_pose, 'blue', only_position=True)
                # plt.draw()
                # plt.pause(0.05)

            image_name = f'client-{cur_arcore_sample["cur_image_idx_stamp"]}.jpg'
            
            if image_name in gt_poses:
                gt_pose = gt_poses[image_name]
                dist = gt_pose.dist_to(cur_fused_pose)
                angle = gt_pose.angle_to(cur_fused_pose)
                timestamp = time.monotonic() - t0
                stats['client']['time'].append(timestamp)
                stats['client']['dist'].append(dist)
                stats['client']['angle'].append(angle)
                logs.append(f'client|{cur_fused_pose.to_formatted_str("%qw %qx %qy %qz %tx %ty %tz")}|{timestamp}|{dist}|{angle}|{image_name}')
                logs.append(f'gt|{gt_pose.to_formatted_str("%qw %qx %qy %qz %tx %ty %tz")}|{timestamp}|||{image_name}')

        t2 = time.monotonic()
        
        wait_time = 0.199 - (t2 - t1)
        if wait_time > 0:
            plt.pause(wait_time)
        
        t3 = time.monotonic()
        print(t3 - t1, t3 - t1 > 0.21)
    
    cur_arcore_sample = None
    is_finished = True

    print('__main__ ended')

    time.sleep(1)

    if display:
        plt.draw()
        plt.savefig(f'{args.out}/{args.test_id}_trackvis.png')

    # Statistics
    print('===========================')
    flag = datetime.strftime(datetime.now(), "%y-%m-%d %H:%M:%S")

    with open(f'{args.out}/{args.test_id}_logs.txt', 'w') as f:
        f.writelines('\n'.join([f'========\n{flag}\ncurScale={curScale}'] + logs))

    stats_log = [f'========\n{flag}']
    for key in ['client', 'server', 'filtered']:
        stats[key]['time'] = np.array(stats[key]['time'])
        stats[key]['dist'] = np.array(stats[key]['dist']) / curScale
        stats[key]['angle'] = np.array(stats[key]['angle'])

        for observation in ['dist', 'angle']:
            stat = stats[key][observation]
            if len(stat):
                stats_log.append(f'{key} {observation} error')
                stats_log.append(f'    n: {len(stat)} Median: {np.round(np.median(stat), 4)} Mean: {np.round(np.mean(stat), 4)} Min: {np.round(np.min(stat), 4)} Max: {np.round(np.max(stat), 4)}')
                print(f'{key} {observation} error')
                print(f'    n: {len(stat)} Median: {np.round(np.median(stat), 4)} Mean: {np.round(np.mean(stat), 4)} Min: {np.round(np.min(stat), 4)} Max: {np.round(np.max(stat), 4)}')

    with open(f'{args.out}/{args.test_id}_statslog.txt', 'w') as f:
        f.writelines('\n'.join(stats_log))

    if display:
        fig_stats, ax_stats = plt.subplots(2)
        ax_stats[0].set_ylim([0, 0.2])
        # ax_stats[0].set_title('Distance error')
        # ax_stats[0].set_xlabel('Time (sec)')
        ax_stats[0].set_ylabel('Error (m)')
        ax_stats[0].plot(stats['client']['time'], stats['client']['dist'], 'b-', label='Client')
        ax_stats[0].plot(stats['server']['time'], stats['server']['dist'], 'r.', label='Server')
        ax_stats[0].plot(stats['filtered']['time'], stats['filtered']['dist'], 'k.', label='Filtered')
        ax_stats[0].legend()
        
        ax_stats[1].set_ylim([0, 2.5])
        # ax_stats[1].set_title('Angle error')
        ax_stats[1].set_xlabel('Time (sec)')
        ax_stats[1].set_ylabel('Error (ang)')
        ax_stats[1].plot(stats['client']['time'], stats['client']['angle'], 'b-', label='Client')
        ax_stats[1].plot(stats['server']['time'], stats['server']['angle'], 'r.', label='Server')
        ax_stats[0].plot(stats['filtered']['time'], stats['filtered']['angle'], 'k.', label='Filtered')
        ax_stats[1].legend()
        plt.draw()
        plt.savefig(f'{args.out}/{args.test_id}_stats.png')

    s.close()
    
    if display:
        plt.show()
        # plt.savefig(f'args.{args.out}/{args.test_id}_trackvis.png')

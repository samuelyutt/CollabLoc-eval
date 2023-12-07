import matplotlib.style as mplstyle
mplstyle.use('fast')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection\

import numpy as np
import quaternion


class Point():
    def __init__(self, point3d_txt):
        #   point3D_txt: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
        _tmp = point3d_txt.split(' ')
        self.id = int(_tmp[0])
        self.x = float(_tmp[1])
        self.y = float(_tmp[2])
        self.z = float(_tmp[3])
        self.r = int(_tmp[4])
        self.g = int(_tmp[5])
        self.b = int(_tmp[6])

    def pos(self):
        return np.array([self.x, self.y, self.z])

    def color(self):
        return np.array([self.r, self.g, self.b])


class Pose():
    def __init__(self, q, t, ptype='normal'):
        self.type = ptype
        if ptype == 'colmap':
            colmap_quat = np.quaternion(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            colmap_pos = np.array([float(t[0]), float(t[1]), float(t[2])])
            quat = colmap_quat.inverse()
            quat_arr = quaternion.as_float_array(quat)
            pos_arr = np.matmul(-quaternion.as_rotation_matrix(quat), colmap_pos)
            self.qw = quat_arr[0]
            self.qx = quat_arr[1]
            self.qy = quat_arr[2]
            self.qz = quat_arr[3]
            self.x = pos_arr[0]
            self.y = pos_arr[1]
            self.z = pos_arr[2]
        elif ptype == 'arcore':
            ar_quat = np.quaternion(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            quat = ar_quat * np.quaternion(0, -1, 0, 0)
            quat_arr = quaternion.as_float_array(quat)
            self.qw = quat_arr[0]
            self.qx = quat_arr[1]
            self.qy = quat_arr[2]
            self.qz = quat_arr[3]
            self.x = float(t[0])
            self.y = float(t[1])
            self.z = float(t[2])
        else:
            self.qw = float(q[0])
            self.qx = float(q[1])
            self.qy = float(q[2])
            self.qz = float(q[3])
            self.x = float(t[0])
            self.y = float(t[1])
            self.z = float(t[2])

    def pos(self):
        return np.array([self.x, self.y, self.z])

    def quat(self):
        return np.quaternion(self.qw, self.qx, self.qy, self.qz)

    def rot_mtx(self):
        return quaternion.as_rotation_matrix(self.quat())

    def __str__(self):
        return f'{self.quat()} {self.pos()}'
    
    def to_arcore_pose_str(self):
        return f'{{{self.x}f, {self.y}f, {self.z}f}} {{{self.qx}f, {self.qy}f, {self.qz}f, {self.qw}f}}'


class CheckPoints():
    def __init__(self):
        self.ckpts = {}
        self.latest_ckpt_idx = None

    def __len__(self):
        return len(self.ckpts)

    def add(self, ckpt_idx, ckpt_c_pose, ckpt_s_pose, ckpt_scale, confidence):
        assert int(ckpt_idx) >= 0
        # confidence = min(max(confidence, 0.0), 1.0)
        self.ckpts[int(ckpt_idx)] = (ckpt_c_pose, ckpt_s_pose, ckpt_scale, confidence)
        if self.latest_ckpt_idx is None or int(ckpt_idx) > self.latest_ckpt_idx:
            self.latest_ckpt_idx = int(ckpt_idx)
    
    def get(self, ckpt_idx, accept_earlier=False, min_confidence=0.0, default=(None, None, None, None)):
        if accept_earlier and int(ckpt_idx) >= 0:
            ret = self.ckpts.get(int(ckpt_idx), None)
            if ret is not None and ret[3] >= min_confidence:
                return ret
            return self.get(int(ckpt_idx) - 1, True, min_confidence, default)
        else:
            return self.ckpts.get(int(ckpt_idx), default)

    def get_latest(self, accept_earlier=False, min_confidence=0.8, default=(None, None, None, None)):
        if self.latest_ckpt_idx is not None:
            return self.get(self.latest_ckpt_idx, accept_earlier, min_confidence, default)
        return default


class GTPoses():
    def __init__(self):
        self.gt_poses = {}

    def __len__(self):
        return len(self.gt_poses)
    
    def add(self, key, gt_pose):
        self.gt_poses[key.replace('.jpg', '').replace('.png', '')] = gt_pose

    def get(self, key, default=None):
        return self.gt_poses.get(key, default)
    
    def compare(self, key, rhs, default=(None, None)):
        gt_pose = self.get(key)
        if gt_pose is not None:
            dist = calc_dist(rhs.pos(), gt_pose.pos())
            ang = calc_angle(rhs.quat(), gt_pose.quat())
            return dist, ang
        else:
            return default


def read_file_lines(fname):
    f = open(fname, 'r')
    lines = f.readlines()
    return lines


def load_gt_poses(gtposes_fname):
    gt_poses = GTPoses()
    gtposes_lines = read_file_lines(gtposes_fname)

    for line_idx, line in enumerate(gtposes_lines):
        # Tokens format
        # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        tokens = line.split()
        if len(tokens) < 1:
            continue
        q = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4])])
        t = np.array([float(tokens[5]), float(tokens[6]), float(tokens[7])])
        gt_pose = Pose(q, t, 'colmap')
        gt_poses.add(tokens[9], gt_pose)

    return gt_poses


def parse_log_line(line):
    # 0:Received|1:ckpt_image_idx|2:ckpt_c_pose|3:ckpt_c_abs_pose|4:ckpt_s_pose|5:ckpt_s_world_pose|6:cur_scale|7:ckpt_fused_world_pose|8:old_fused_world_pose|9:is_ckpt_avail|10:cur_track_conf|11:focal_length
    # 0:ARPose|1:ckpt_image_idx|2:ckpt_c_pose|3:ckpt_c_abs_pose|4:ckpt_s_pose|5:ckpt_s_world_pose|6:cur_c_pose|7:cur_c_abs_pose|8:cur_fused_world_pose|9:cur_image_idx
    # 0:SampledPose|1:cur_image_idx|2:cur_image_idx_stamp|3:c_abs_pose|4:fused_world_pose|5:latest_avail_ckpt_idx

    fields = {
        'Received': ['ckpt_image_idx', 'ckpt_c_pose', 'ckpt_c_abs_pose', 'ckpt_s_pose', 'ckpt_s_world_pose', 'cur_scale', 'ckpt_fused_world_pose', 'old_fused_world_pose', 'is_ckpt_avail', 'cur_track_conf', 'focal_length'],
        'ARPose': ['ckpt_image_idx', 'ckpt_c_pose', 'ckpt_c_abs_pose', 'ckpt_s_pose', 'ckpt_s_world_pose', 'cur_c_pose', 'cur_c_abs_pose', 'cur_fused_world_pose', 'cur_image_idx'],
        'SampledPose': ['cur_image_idx', 'cur_image_idx_stamp', 'c_abs_pose', 'fused_world_pose', 'latest_avail_ckpt_idx'],
    }

    tokens = {}
    spl = line.replace('\n', '').split('|')
    if spl[0] in fields:
        tokens['log_type'] = spl[0]
        for i, field in enumerate(fields[spl[0]]):
            if i <= len(spl):
                tokens[field] = spl[i + 1]
            else:
                tokens[field] = '?'
        return tokens
    else:
        return spl


def calc_acc_img_cnts(sampledimgs_lines):
    acc_img_cnts = {}
    max_key = -1
    for line_idx, line in enumerate(sampledimgs_lines):
        tokens = parse_log_line(line)
        if len(tokens) < 1:
            continue
        sub_idx = int(tokens['cur_image_idx_stamp'].split('-')[1])
        if int(tokens['cur_image_idx']) not in acc_img_cnts:
            acc_img_cnts[int(tokens['cur_image_idx'])] = sub_idx
            if int(tokens['cur_image_idx']) > max_key:
                max_key = int(tokens['cur_image_idx'])
        else:
            if acc_img_cnts[int(tokens['cur_image_idx'])] < sub_idx:
                acc_img_cnts[int(tokens['cur_image_idx'])] = sub_idx

    for i in range(max_key):
        if i not in acc_img_cnts:
            acc_img_cnts[i] = 0
    tmp_list = sorted([(key, values) for key, values in acc_img_cnts.items()])

    acc_img_cnts = {}
    for key, values in tmp_list:
        acc_img_cnts[key] = values

    print(acc_img_cnts)
    acc_val = 0
    for key, value in acc_img_cnts.items():
        acc_val += value
        acc_img_cnts[key] = acc_val
    print(acc_img_cnts)
    return acc_img_cnts


def parse_arpose(arpose_str):
    trashes = ['q:', '[', 'x:', 'y:', 'z:', 'w:', ']', 't:', ',', '\n']
    for trash in trashes:
        arpose_str = arpose_str.replace(trash, "")
    tok = arpose_str.split(' ')
    # x y z qx qy qz qw -> id qw qx qy qz x y z
    q = np.array([tok[6], tok[3], tok[4], tok[5]])
    t = np.array([tok[0], tok[1], tok[2]])
    return q, t


def skip(test_id, tokens):
    ret = False
    return ret


def get_color_by_dist(dist):
    return 'lime' if dist <= 0.1 else 'orange' if dist <= 0.2 else 'red' if dist <= 0.3 else 'maroon'


def calc_dist(pos1, pos2):
    # Calculate distance between two positions
    return np.linalg.norm(pos1 - pos2)


def calc_angle(q1, q2):
    # Calculate angle between two quaternions
    # Referenced to function `rotation_error` in TrianFlow:
    # https://github.com/B1ueber2y/TrianFlow/blob/7234224ed73ac724a2cce9da41f712f5446c5ea6/core/evaluation/eval_odom.py#L123
    pose_error = quaternion.as_rotation_matrix(q1.inverse() * q2)
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5 * (a + b + c - 1.0)
    rot_error = np.arccos(max(min(d, 1.0), -1.0))
    rot_error_deg = np.degrees(rot_error)
    return rot_error_deg


def calc_fused_pose(cur_c_pose, ckpt_c_pose, ckpt_s_pose, scale):
    rel_q = ckpt_s_pose.quat() * ckpt_c_pose.quat().inverse()
    r_diff = quaternion.as_rotation_matrix(rel_q)

    fused_q = quaternion.from_rotation_matrix(np.matmul(
        r_diff,
        quaternion.as_rotation_matrix(cur_c_pose.quat()),
    ))
    fused_t = np.matmul(
        r_diff,
        cur_c_pose.pos() - ckpt_c_pose.pos(),
    ) * scale + ckpt_s_pose.pos()

    return Pose(quaternion.as_float_array(fused_q), fused_t)


def load_point_cloud(points_fname):
    f = open(points_fname, 'r')
    lines = f.readlines()
    pc = []
    for i, line in enumerate(lines):
        if line[0] == '#':
            continue
        p = Point(line)
        pc.append(p.pos())
    return np.array(pc)


def display_pose(cam_poses, pc=None, only_pos=False, axis_on=False):
    # Initial plot
    # plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.axis('on' if axis_on else 'off')
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-10, 10])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Display point cloud
    if pc is not None:
        ax.plot(pc[:, 0], pc[:, 1], pc[:, 2], marker='.', c='gray', markersize=0.3, lw=0)
        plt.draw()

    # Display traj
    for i, values in enumerate(cam_poses):
        # plt.draw() 
        # plt.pause(0.001) #is necessary for the plot to update for some reason

        if len(values) == 2:
            p, c = values
            markersize = 2
        else:
            p, c, markersize = values

        if only_pos:
            t = p.pos()
            ax.plot([t[0]], [t[1]], [t[2]], marker='o', c=c, markersize=markersize, lw=0)
        else:
            p_scale = 0.1
            rot_mtx = p.rot_mtx()
            t = p.pos()

            p1 = np.dot(rot_mtx, [2, 1, 1.5]) * p_scale + t
            p2 = np.dot(rot_mtx, [-2, 1, 1.5]) * p_scale + t
            p3 = np.dot(rot_mtx, [-2, -1, 1.5]) * p_scale + t
            p4 = np.dot(rot_mtx, [2, -1, 1.5]) * p_scale + t

            x = [p1[0], p2[0], p3[0], p4[0]]
            y = [p1[1], p2[1], p3[1], p4[1]]
            z = [p1[2], p2[2], p3[2], p4[2]]
            vertices = [list(zip(x, y, z))]
            poly = Poly3DCollection(vertices, alpha=0.3, color=c)
            ax.add_collection3d(poly)

            ax.plot([t[0], p1[0]], [t[1], p1[1]], [t[2], p1[2]], marker='.', c=c, markersize=0.5, lw=0.5)
            ax.plot([t[0], p2[0]], [t[1], p2[1]], [t[2], p2[2]], marker='.', c=c, markersize=0.5, lw=0.5)
            ax.plot([t[0], p3[0]], [t[1], p3[1]], [t[2], p3[2]], marker='o', c=c, markersize=1, lw=0.5)
            ax.plot([t[0], p4[0]], [t[1], p4[1]], [t[2], p4[2]], marker='.', c=c, markersize=0.5, lw=0.5)

    plt.draw() 

    # Display plot
    plt.show()

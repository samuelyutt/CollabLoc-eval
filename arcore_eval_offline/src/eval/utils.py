import math
import numpy as np
import quaternion

import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

    def __str__(self):
        return f'q:{self.t()}, t:{self.q()}'

    def __repr__(self):
        return self.__str__()

    def t(self):
        return np.array([self.x, self.y, self.z])

    def q(self):
        return np.quaternion(self.qw, self.qx, self.qy, self.qz)

    def rot_mtx(self):
        return quaternion.as_rotation_matrix(self.q())

    def dist_to(self, rhs):
        # Calculate distance between two positions
        return np.linalg.norm(self.t() - rhs.t())

    def angle_to(self, rhs):
        # Calculate angle between two quaternions
        # Referenced to function `rotation_error` in TrianFlow:
        # https://github.com/B1ueber2y/TrianFlow/blob/7234224ed73ac724a2cce9da41f712f5446c5ea6/core/evaluation/eval_odom.py#L123
        pose_error = quaternion.as_rotation_matrix(self.q().inverse() * rhs.q())
        a = pose_error[0, 0]
        b = pose_error[1, 1]
        c = pose_error[2, 2]
        d = 0.5 * (a + b + c - 1.0)
        rot_error = np.arccos(max(min(d, 1.0), -1.0))
        rot_error_deg = np.degrees(rot_error)
        return rot_error_deg

    def to_formatted_str(self, format_str: str) -> str:
        return format_str.replace(
            '%qw', str(self.qw)).replace('%qx', str(self.qx)).replace('%qy', str(self.qy)).replace('%qz', str(self.qz)).replace(
            '%tx', str(self.x)).replace('%ty', str(self.y)).replace('%tz', str(self.z))

    def to_arcore_pose_str(self) -> str:
        return self.to_formatted_str('{%txf, %tyf, %tzf} {%qxf, %qyf, %qzf, %qwf}')


def parse_arcore_pose_str(pose_str: str) -> list:
    for c in ['q', 't', 'x', 'y', 'z', 'w', ':', '[', ']', ',']:
        pose_str = pose_str.replace(c, '')
    return pose_str.split()


def pose_from_arpose_str(pose_str: str):
    if pose_str == 'null':
        return None
    tok = parse_arcore_pose_str(pose_str)
    q = [tok[6], tok[3], tok[4], tok[5]]
    t = [tok[0], tok[1], tok[2]]
    return Pose(q, t)


def read_sampled_imgs_log(sampled_imgs_log_path):
    with open(sampled_imgs_log_path) as f:
        lines = f.readlines()
    ret = []
    for line in lines:
        tokens = parse_log_line(line)
        ret.append(tokens)
    return ret


def read_file_lines(fname):
    f = open(fname, 'r')
    lines = f.readlines()
    return lines


def load_gt_poses(gt_poses_path):
    with open(gt_poses_path) as f:
        lines = f.readlines()
    gt_poses = {}
    for line in lines:
        tokens = line.split()
        if len(tokens) < 1:
            continue
        q = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4])])
        t = np.array([float(tokens[5]), float(tokens[6]), float(tokens[7])])
        gt_pose = Pose(q, t, 'colmap')
        gt_poses[tokens[9]] = gt_pose
    return gt_poses  


def load_point_cloud(points3d_path):
    f = open(points3d_path, 'r')
    lines = f.readlines()
    pc = []
    for i, line in enumerate(lines):
        if line[0] == '#':
            continue
        p = Point(line)
        pc.append(p.pos())
    return np.array(pc)


def initial_plot(pc_path=None):
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(projection='3d')
    ax.axis('off')
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-10, 10])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if pc_path is not None:
        pc = load_point_cloud(pc_path)
        ax.plot(pc[:, 0], pc[:, 1], pc[:, 2], marker='.', c='gray', markersize=0.3, lw=0)

    return fig, ax


def display_pose(ax, p, c, only_position=False, pose_size=0.05):
    if only_position:
        t = p.t()
        ax.plot([t[0]], [t[1]], [t[2]], marker='.', c=c, markersize=2, lw=0)
    else:
        rot_mtx = p.rot_mtx()
        t = p.t()

        p1 = np.dot(rot_mtx, [2, 1, 1.5]) * pose_size + t
        p2 = np.dot(rot_mtx, [-2, 1, 1.5]) * pose_size + t
        p3 = np.dot(rot_mtx, [-2, -1, 1.5]) * pose_size + t
        p4 = np.dot(rot_mtx, [2, -1, 1.5]) * pose_size + t

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

    return ax


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
    

def qvecFromRTMtx(mtx):
    # Returns qvec in qw, qx, qy, qz
    qw = math.sqrt(1.0 + mtx[0][0] + mtx[1][1] + mtx[2][2]) / 2.0
    w4 = (4.0 * qw)
    qx = -(mtx[2][1] - mtx[1][2]) / w4
    qy = -(mtx[0][2] - mtx[2][0]) / w4
    qz = -(mtx[1][0] - mtx[0][1]) / w4
    return qw, qx, qy, qz


def tvecFromRTMtx(mtx):
    # Returns tvec in (tx, ty, tz)
    # tvec = -R^T * T
    tx = -(mtx[0][0] * mtx[0][3] + mtx[1][0] * mtx[1][3] + mtx[2][0] * mtx[2][3])
    ty = -(mtx[0][1] * mtx[0][3] + mtx[1][1] * mtx[1][3] + mtx[2][1] * mtx[2][3])
    tz = -(mtx[0][2] * mtx[0][3] + mtx[1][2] * mtx[1][3] + mtx[2][2] * mtx[2][3])
    return tx, ty, tz


def calc_fused_pose(cur_c_pose, ckpt_c_pose, ckpt_s_pose, scale):
    rel_q = ckpt_s_pose.q() * ckpt_c_pose.q().inverse()
    r_diff = quaternion.as_rotation_matrix(rel_q)

    fused_q = quaternion.from_rotation_matrix(np.matmul(
        r_diff,
        quaternion.as_rotation_matrix(cur_c_pose.q()),
    ))
    fused_t = np.matmul(
        r_diff,
        cur_c_pose.t() - ckpt_c_pose.t(),
    ) * scale + ckpt_s_pose.t()

    return Pose(quaternion.as_float_array(fused_q), fused_t)

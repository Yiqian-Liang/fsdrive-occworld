#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multi_view_bev.py
- 聚合 nuScenes LIDAR 多帧到当前 sample，生成 BEV 网格
- 计算每个相机的 FOV 在地面 z=0 的投影多边形（近似）
- 可保存可视化 PNG

用法示例:
python multi_view_bev.py \
  --nusc_root /data/sets/nuscenes \
  --version v1.0-trainval \
  --scene_idx 0 \
  --out_dir bev_vis \
  --nsweeps 10 \
  --range_x -50 50 \
  --range_y -50 50 \
  --res 0.5
"""
import os, math, argparse, json
import numpy as np
import cv2
import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion


def get_sample_by_scene_idx(nusc: NuScenes, scene_idx: int, sample_offset: int):
    scene = nusc.scene[scene_idx]
    sample_token = scene["first_sample_token"]
    # 跳到第 sample_offset 帧
    for _ in range(sample_offset):
        sample = nusc.get("sample", sample_token)
        if sample["next"] == "":
            break
        sample_token = sample["next"]
    return nusc.get("sample", sample_token)


def load_lidar_points_in_lidar_frame(nusc: NuScenes, sample: dict, nsweeps=10):
    """
    返回当前 sample 的 LIDAR_TOP 多帧叠加点云（在当前帧 LIDAR 坐标系）
    """
    chan = "LIDAR_TOP"
    pc, _times = LidarPointCloud.from_file_multisweep(
        nusc, sample, chan, chan, nsweeps=nsweeps
    )
    # pc.points: shape (4, N) 或 (5, N)，前3行为 xyz
    pts = pc.points[:3, :].T  # [N,3]
    return pts  # LIDAR坐标（等价于“当前ego系的前/左/上”方向）


def lidar_to_ego_matrix(nusc: NuScenes, sample: dict):
    sd_rec = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    cs = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    R = Quaternion(cs["rotation"]).rotation_matrix  # lidar->ego
    t = np.array(cs["translation"]).reshape(3, 1)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3:] = t
    return T


def ego_to_global_matrix(nusc: NuScenes, sample_data_token: str):
    sd_rec = nusc.get("sample_data", sample_data_token)
    ego = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    R = Quaternion(ego["rotation"]).rotation_matrix  # ego->global
    t = np.array(ego["translation"]).reshape(3, 1)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3:] = t
    return T


def get_cam_extrinsic_intrinsic(nusc: NuScenes, cam_sd_token: str):
    sd = nusc.get("sample_data", cam_sd_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    # cam->ego
    R = Quaternion(cs["rotation"]).rotation_matrix
    t = np.array(cs["translation"]).reshape(3, 1)
    T_cam2ego = np.eye(4, dtype=np.float32)
    T_cam2ego[:3, :3] = R
    T_cam2ego[:3, 3:] = t
    K = np.array(cs["camera_intrinsic"], dtype=np.float32)
    return T_cam2ego, K, sd["width"], sd["height"]


def compute_cam_fov_ground_polygon(nusc: NuScenes, sample: dict, cam_key: str,
                                   plane_z=0.0, img_border=10):
    """
    取图像四角（可微调边界），把四条视线与 z=plane_z 的平面相交，得到多边形（在ego坐标系）
    简化：使用当前 sample 下相机自己的 ego 位姿，无跨时刻对齐（做FOV覆盖足够）
    """
    cam_sd = nusc.get("sample_data", sample["data"][cam_key])
    T_cam2ego, K, W, H = get_cam_extrinsic_intrinsic(nusc, cam_sd["token"])
    # 相机内参逆
    Kinv = np.linalg.inv(K)

    # 选图像四角
    corners = np.array([
        [img_border, img_border, 1.0],
        [W - img_border, img_border, 1.0],
        [W - img_border, H - img_border, 1.0],
        [img_border, H - img_border, 1.0]
    ], dtype=np.float32).T  # [3,4]

    # cam坐标系下方向向量
    rays_cam = Kinv @ corners  # [3,4]
    rays_cam = rays_cam / np.linalg.norm(rays_cam, axis=0, keepdims=True)

    # 变到 ego坐标 (点在相机原点 + s * 方向)
    R = T_cam2ego[:3, :3]
    t = T_cam2ego[:3, 3:4]
    dirs_ego = R @ rays_cam  # [3,4]
    cam_pos_ego = t          # [3,1]

    # 与 z=plane_z 相交： cam_pos_ego.z + s * dir.z = plane_z
    zs = dirs_ego[2, :] + 1e-9
    s = (plane_z - cam_pos_ego[2, 0]) / zs
    # 只要 s>0 的交点（在相机前方）
    s = np.maximum(s, 0.0)

    p_ego = cam_pos_ego + dirs_ego * s  # [3,4]
    poly = p_ego[:2, :].T               # [4,2], (x,y) in ego
    return poly  # 顺序：左上-右上-右下-左下（近似）


def make_bev_grid(xy, x_range=(-50, 50), y_range=(-50, 50), res=0.5, binary=False):
    """
    xy: [N,2] in ego, x 前/ y 左
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    W = int((x_max - x_min) / res)
    H = int((y_max - y_min) / res)

    # 将 (x,y) 映射到像素：行= y, 列= x。（习惯：x向上，y向右，最终图像再旋转/翻转看习惯）
    ix = ((xy[:, 0] - x_min) / res).astype(np.int32)  # 列
    iy = ((xy[:, 1] - y_min) / res).astype(np.int32)  # 行
    mask = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    ix, iy = ix[mask], iy[mask]

    bev = np.zeros((H, W), np.uint16)
    if binary:
        bev[iy, ix] = 1
    else:
        # 计数图（可后续归一化/阈值）
        np.add.at(bev, (iy, ix), 1)
    return bev  # [H,W], 0在上（对应 y最小）


def draw_cam_fov_on_bev(bev_rgb, polys_xy, x_range, y_range, res, color=(0, 255, 255), alpha=0.2):
    """
    在 BEV 上把 FOV 多边形涂出来（半透明）
    bev_rgb: [H,W,3] uint8
    polys_xy: list of [K,2] 顶点（ego坐标）
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    H, W = bev_rgb.shape[:2]

    overlay = bev_rgb.copy()
    for poly in polys_xy:
        # 到像素
        pts = []
        for x, y in poly:
            u = int((x - x_min) / res)  # 列
            v = int((y - y_min) / res)  # 行
            pts.append([u, v])
        pts = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(overlay, [pts], True, (0, 0, 0), 1)

    cv2.addWeighted(overlay, alpha, bev_rgb, 1 - alpha, 0, dst=bev_rgb)
    return bev_rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nusc_root", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--scene_idx", type=int, default=0)
    ap.add_argument("--sample_offset", type=int, default=0)
    ap.add_argument("--nsweeps", type=int, default=10)
    ap.add_argument("--range_x", nargs=2, type=float, default=[-50, 50])
    ap.add_argument("--range_y", nargs=2, type=float, default=[-50, 50])
    ap.add_argument("--res", type=float, default=0.5)
    ap.add_argument("--binary", action="store_true")
    ap.add_argument("--out_dir", default="bev_vis")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    nusc = NuScenes(version=args.version, dataroot=args.nusc_root, verbose=True)

    sample = get_sample_by_scene_idx(nusc, args.scene_idx, args.sample_offset)
    print("[*] scene:", nusc.scene[args.scene_idx]["name"], "| sample_token:", sample["token"])

    # 1) 聚合多帧点云（已做跨时刻位姿对齐）
    pts_lidar = load_lidar_points_in_lidar_frame(nusc, sample, nsweeps=args.nsweeps)  # [N,3]
    xy = pts_lidar[:, :2]  # ego/LiDAR坐标，x前y左

    # 2) 网格化
    bev = make_bev_grid(
        xy,
        x_range=tuple(args.range_x),
        y_range=tuple(args.range_y),
        res=args.res,
        binary=args.binary
    )  # [H,W]
    # 归一化可视化
    vis = bev.astype(np.float32)
    vis = vis / (vis.max() + 1e-6)
    vis = (vis * 255).clip(0, 255).astype(np.uint8)
    bev_rgb = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    # 3) 画6相机的FOV在地面z=0的投影（帮助理解多视角覆盖）
    cam_keys = [k for k in sample["data"].keys() if k.startswith("CAM_")]
    polys = []
    for k in cam_keys:
        poly = compute_cam_fov_ground_polygon(nusc, sample, k, plane_z=0.0, img_border=10)
        polys.append(poly)
    bev_rgb = draw_cam_fov_on_bev(
        bev_rgb, polys, x_range=tuple(args.range_x), y_range=tuple(args.range_y), res=args.res,
        color=(0, 255, 255), alpha=0.25
    )

    # 4) 画车体原点/坐标轴
    cx = int((-args.range_x[0]) / args.res)
    cy = int((-args.range_y[0]) / args.res)
    cv2.circle(bev_rgb, (cx, cy), 3, (0, 0, 255), -1)           # ego原点
    cv2.arrowedLine(bev_rgb, (cx, cy), (cx, cy - 40), (0, 0, 255), 2)  # x前(上)
    cv2.arrowedLine(bev_rgb, (cx, cy), (cx + 40, cy), (0, 255, 0), 2)  # y左(右)

    # 5) 保存
    out_png = os.path.join(
        args.out_dir,
        f"scene{args.scene_idx:03d}_s{args.sample_offset:04d}_bev.png"
    )
    cv2.imwrite(out_png, bev_rgb)
    print("[saved]", out_png)


if __name__ == "__main__":
    main()

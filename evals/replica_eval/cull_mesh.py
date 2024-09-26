# adapted from https://github.com/cvg/nice-slam
# 对输入的三维网格模型进行剪裁，根据相机位姿和内参信息
# 该文件cull mesh裁剪方式是： 筛出对于所有views都满足不在视锥内或者在图像外的点，然后删除这些点
# 注意：Gtmesh在world系，而learned mesh在scaleed中心坐标系 apply scale mat to mesh
import os
import subprocess

import numpy as np
import argparse
import pickle
import os
import glob
import open3d as o3d
import matplotlib.pyplot as plt
import torch
import trimesh


def load_poses(path):
    poses = []
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1
        c2w = torch.from_numpy(c2w).float()
        poses.append(c2w)
    return poses


parser = argparse.ArgumentParser(
    description='Arguments to cull the mesh.'
)

parser.add_argument('--input_mesh', default="/data/projects/monosdf/runs/replica_grids_6/2023_11_12_02_15_23/plots/surface_1999.ply",type=str, help='path to the mesh to be culled')
parser.add_argument('--input_scalemat', default="../data/Replica/scan6/cameras.npz", type=str, help='path to the scale mat')
parser.add_argument('--traj', default="../data/Replica/scan6/traj.txt",type=str, help='path to the trajectory')
parser.add_argument('--output_mesh', default="./replica_scan6_s3im_3views.ply",type=str, help='path to the output mesh')
args = parser.parse_args()

num_views=-1
fewshot_idx=[9,13,16]
H = 680
W = 1200
fx = 600.0
fy = 600.0
fx = 600.0
cx = 599.5
cy = 339.5
scale = 6553.5
poses = load_poses(args.traj)
n_imgs = len(poses)
mesh = trimesh.load(args.input_mesh, process=False)

# transform to original coordinate system with scale mat
scalemat = np.load(args.input_scalemat)['scale_mat_0']
mesh.vertices = mesh.vertices @ scalemat[:3, :3].T + scalemat[:3, 3]

pc = mesh.vertices
faces = mesh.faces

# delete mesh vertices that are not inside any camera's viewing frustum
whole_mask = np.ones(pc.shape[0]).astype(bool)

for i in range(0, n_imgs, 1):
    if num_views>=0 and not i in fewshot_idx:
        continue
    c2w = poses[i]
    points = pc.copy()
    points = torch.from_numpy(points).cuda()
    w2c = np.linalg.inv(c2w)
    w2c = torch.from_numpy(w2c).cuda().float()
    K = torch.from_numpy(
        np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)).cuda()
    ones = torch.ones_like(points[:, 0]).reshape(-1, 1).cuda()
    homo_points = torch.cat(
        [points, ones], dim=1).reshape(-1, 4, 1).cuda().float()
    cam_cord_homo = w2c @ homo_points
    cam_cord = cam_cord_homo[:, :3]

    cam_cord[:, 0] *= -1
    uv = K.float() @ cam_cord.float()
    z = uv[:, -1:] + 1e-5
    uv = uv[:, :2] / z
    uv = uv.float().squeeze(-1).cpu().numpy()
    edge = 0
    # ~mask: 在视锥外或在图像外，即所有视图都找不到的点。
    mask = (0 <= -z[:, 0, 0].cpu().numpy()) & (uv[:, 0] < W -
                                               edge) & (uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (uv[:, 1] > edge)
    whole_mask &= ~mask
pc = mesh.vertices
faces = mesh.faces
face_mask = whole_mask[mesh.faces].all(axis=1)
mesh.update_faces(~face_mask)
mesh.export(args.output_mesh)

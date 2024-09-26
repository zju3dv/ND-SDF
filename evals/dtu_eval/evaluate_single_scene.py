import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import glob
from skimage.morphology import binary_dilation, disk
import argparse

import trimesh
from pathlib import Path
import subprocess

import sys
sys.path.append("../code")
from utils import rend_util


def cull_scan(scan, mesh_path, result_mesh_file):

    # load poses
    instance_dir = os.path.join('/data/monosdf/DTU', 'scan{0}'.format(scan))
    image_dir = '{0}/image'.format(instance_dir)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    n_images = len(image_paths)

    cam_file = '{0}/cameras.npz'.format(instance_dir)
    camera_dict = np.load(cam_file)
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

    intrinsics_all = []
    pose_all = []
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
        intrinsics_all.append(torch.from_numpy(intrinsics).float())
        pose_all.append(torch.from_numpy(pose).float())

    # load mask
    mask_dir = '{0}/mask'.format(instance_dir)
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    masks = []
    for p in mask_paths:
        mask = cv2.imread(p)
        masks.append(mask)

    # hard-coded image shape
    W, H = 1600, 1200

    # load mesh
    mesh = trimesh.load(mesh_path) # mesh在物体中心坐标系
    vertices = mesh.vertices

    # project and filter
    vertices = torch.from_numpy(vertices).cuda()
    vertices = torch.cat((vertices, torch.ones_like(vertices[:, :1])), dim=-1)
    vertices = vertices.permute(1, 0)
    vertices = vertices.float()

    sampled_masks = []
    invalid_masks = []
    for i in range(n_images):
        pose = pose_all[i]
        w2c = torch.inverse(pose).cuda()
        intrinsic = intrinsics_all[i].cuda()

        with torch.no_grad():
            # transform and project
            cam_points = intrinsic @ w2c @ vertices
            pix_coords = cam_points[:2, :] / (cam_points[2, :].unsqueeze(0) + 1e-6)
            pix_coords = pix_coords.permute(1, 0)
            pix_coords[..., 0] /= W - 1
            pix_coords[..., 1] /= H - 1
            pix_coords = (pix_coords - 0.5) * 2
            valid = ((pix_coords > -1. ) & (pix_coords < 1.)).all(dim=-1).float() # 1: in image, 0: out image
            invalid= 1-valid
            # dialate mask similar to unisurf
            maski = masks[i][:, :, 0].astype(np.float32) / 256. # 0: background, 1: foreground

            maski = torch.from_numpy(binary_dilation(maski, disk(12))).float()[None, None].cuda() # 12 is the dilation radius

            # pix_coords: [N, 2], in [-1, 1]
            # grid_sample: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            # sampled_mask: 1: in object, 0: out object
            sampled_mask = F.grid_sample(maski, pix_coords[None, None], mode='nearest', padding_mode='zeros', align_corners=True)[0, -1, 0]

            # 三维主体mesh中的点，投影在各个视图上，要么投影在主体内即mask内，要么投影在图像外，表示某些视角下看不到该主体点。
            # 但这样有一个坏处：考虑了所有不在图像内即invalid的点，我自己增添一个异或操作去除所有不在图像内的点，保证只有主体点+主体点或者主体点+invalid主体点构成主体mesh。
            sampled_mask = sampled_mask+1-valid

            sampled_masks.append(sampled_mask)
            invalid_masks.append(invalid)
    invalid_masks = torch.stack(invalid_masks, -1)
    sampled_masks = torch.stack(sampled_masks, -1)

    # filter
    mask = (sampled_masks > 0.).all(dim=-1).cpu().numpy() ^ (invalid_masks > 0.).all(dim=-1).cpu().numpy()
    face_mask = mask[mesh.faces].all(axis=1)

    # update_vertices and update_faces: https://trimsh.org/trimesh.html#trimesh.Trimesh.update_vertices
    mesh.update_vertices(mask)
    mesh.update_faces(face_mask)

    # transform vertices to world
    scale_mat = scale_mats[0]
    mesh.vertices = mesh.vertices * scale_mat[0, 0] + scale_mat[:3, 3][None]
    mesh.export(result_mesh_file)
    del mesh


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Arguments to evaluate the mesh.'
    )

    parser.add_argument('--input_mesh', type=str, default='/data/projects/implicit_reconstruction/runs_dtu/dtu_nbg_24/2024-05-11_23-59-45/plots/mesh_409.ply', help='path to the mesh to be evaluated')
    parser.add_argument('--scan_id', type=str, default='24', help='scan id of the input mesh')
    parser.add_argument('--output_dir', type=str, default='evaluation_results_single', help='path to the output folder')
    parser.add_argument('--DTU', type=str,  default='/data/DTU', help='path to the GT DTU point clouds')
    args = parser.parse_args()


    Offical_DTU_Dataset = args.DTU
    out_dir = args.output_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    scan = args.scan_id
    ply_file = args.input_mesh

    result_mesh_file = os.path.join(out_dir, "culled_mesh.ply")

    cull_scan(scan, ply_file, result_mesh_file)

    cmd = f"python eval.py --data {result_mesh_file} --scan {scan} --mode mesh --dataset_dir {Offical_DTU_Dataset} --vis_out_dir {out_dir}"
    os.system(cmd)
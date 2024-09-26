import argparse
import json

import cv2
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import trimesh
import torch
import glob
import os
import pyrender
import os
from tqdm import tqdm
from pathlib import Path

os.environ['PYOPENGL_PLATFORM'] = 'egl'

class Renderer():
    def __init__(self, height=480, width=640):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()
        # self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES

    def __call__(self, height, width, intrinsics, pose, mesh):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                        fx=intrinsics[0, 0], fy=intrinsics[1, 1])
        self.scene.add(cam, pose=self.fix_pose(pose))
        return self.renderer.render(self.scene)  # , self.render_flags)

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh)

    def delete(self):
        self.renderer.delete()

def update_mesh(mesh, mask):
    # 完全删除mask为False的部分
    verts, colors = mesh.vertices, mesh.visual.vertex_colors
    # update
    verts, colors= verts[mask], colors[mask]
    if getattr(mesh, 'faces', None) is not None:
        index_mapping = np.zeros_like(mask, dtype=np.int64)
        index_mapping[mask] = np.arange(mask.sum())  # old idx -> new idx
        faces = mesh.faces
        face_mask = mask[faces].all(axis=1)
        faces = index_mapping[faces[face_mask]]
        mesh = trimesh.Trimesh(vertices=verts, vertex_colors=colors, faces=faces)
    else:
        mesh = trimesh.PointCloud(vertices=verts, colors=colors)
    return mesh

def cull_mesh(mesh, poses, K, h, w, apply_mask=False, mask_paths=None):
    scene_mask = np.ones(shape=mesh.vertices.shape[0], dtype=bool)
    invalids = np.ones(shape=mesh.vertices.shape[0], dtype=bool)
    for i,c2w in tqdm(enumerate(poses),total=len(poses)):
        w2c = np.linalg.inv(c2w)
        verts = mesh.vertices.T
        homo_verts = np.concatenate([verts,np.ones(shape=(1,verts.shape[1]))],axis=0)
        pix_verts = K@(w2c[:3]@homo_verts)
        pix = pix_verts[:2,:] / (pix_verts[2:,:]+1e-6)
        # 三维主体mesh中的点，投影在各个视图上，要么投影在主体内即mask内，要么投影在图像外，表示某些视角下看不到该主体点。
        # 但这样有一个坏处：考虑了所有不在图像内即invalid的点，最后增添一个异或操作去除所有不在图像内的点，保证只有主体点+主体点或者主体点+invalid主体点构成主体mesh。
        # FIXME：之前的out_scene mask没有考虑多视图一致性，比如空中错误漂浮物会被某个视图直接筛为主体点。
        mask = (pix_verts[-1,:]>=0)& ((pix[0,:]>=0)&(pix[0,:]<=w)&(pix[1,:]>=0)&(pix[1,:]<=h))
        invalid = ~mask  # 投影在图像外
        invalids = invalids&invalid
        if apply_mask and mask_paths is not None: # 如果有背景mask
            bg_mask = np.load(mask_paths[i])
            object_mask = cv2.resize((~bg_mask).astype(np.uint8), (w, h), cv2.INTER_NEAREST)
            object_mask = cv2.dilate(object_mask, np.ones((12,12),dtype=np.uint8), iterations=1).astype(bool)
            pix=pix.astype(np.int64)
            pix[0,:] = np.clip(pix[0,:],0, w-1)
            pix[1,:] = np.clip(pix[1,:], 0, h-1)
            mask = (mask)& (object_mask[pix[1,:],pix[0,:]])
        scene_mask = scene_mask&(mask|invalid)

    return update_mesh(mesh,scene_mask^invalids)

def refuse(mesh, poses, K, h ,w):
    renderer = Renderer()
    mesh_opengl = renderer.mesh_opengl(mesh)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.002,
        sdf_trunc=3 * 0.01,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    for pose in tqdm(poses):
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = K

        rgb = np.ones((h, w, 3))
        rgb = (rgb * 255).astype(np.uint8)
        rgb = o3d.geometry.Image(rgb)
        _, depth_pred = renderer(h, w, intrinsic, pose, mesh_opengl)
        depth_pred = o3d.geometry.Image(depth_pred)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth_pred, depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
        )
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)
        extrinsic = np.linalg.inv(pose)
        volume.integrate(rgbd, intrinsic, extrinsic)

    return volume.extract_triangle_mesh()


def main(args):
    # load mesh
    mesh = trimesh.load(args.mesh)

    idx=args.scan_id
    with open(os.path.join(args.data_dir,f'{idx}','meta_data.json'),'r') as f:
        data=json.load(f)

    # transform to world coordinate
    scale_mat = data['worldtogt']

    if args.gt_space:
        mesh.apply_transform(np.linalg.inv(scale_mat))
    poses=[]
    mask_paths = []
    for frame in data['frames']:
        pose = np.array(frame["camtoworld"], dtype=np.float32)
        poses.append(pose)
        # mask_paths.append(os.path.join(args.data_dir,f'scene{idx}',frame["mask_path"]))
    K = np.array(data['frames'][0]['intrinsics'], dtype=np.float32)[:3, :3]

    # intrinsic_path = os.path.join(f'../data/scannet/scan{idx}/intrinsic/intrinsic_color.txt')
    # K = np.loadtxt(intrinsic_path)[:3, :3]

    mesh = cull_mesh(mesh, poses, K,data['height'],data['width'], args.apply_mask, mask_paths)
    mesh.apply_transform(scale_mat)

    # save mesh
    os.makedirs(os.path.join(args.out_dir,args.exp_name+f"_{idx}"),exist_ok=True)
    out_mesh_path = os.path.join(args.out_dir,args.exp_name+f"_{idx}", 'culled_'+os.path.basename(args.mesh))
    mesh.export(out_mesh_path)

if __name__=='__main__':

    args=argparse.ArgumentParser()
    args.add_argument('--mesh', type=str, default='/data/projects/implicit_reconstruction/scripts/scannetpp_meshes_version1/scannetpp7-1_7f4d173c9c_cues_1024_latest.ply')
    args.add_argument('--data_dir',type=str, default='/data/scannetpp')
    args.add_argument('--scan_id', type=str, default='7f4d173c9c')
    args.add_argument('--apply_mask',action='store_true', help='是否仅保留mask内的mesh')
    args.add_argument('--gt_space',action='store_true',help='输入mesh是否在gt space, 该脚本文件保证输出为gt_space')
    args.add_argument('--exp_name', type=str, default='tmp')
    args.add_argument('--out_dir', type=str, default='culled_mesh')
    args.set_defaults(gt_space=False, apply_mask=False)

    args=args.parse_args()

    main(args)

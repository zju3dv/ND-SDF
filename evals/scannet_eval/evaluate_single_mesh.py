# adapted from https://github.com/zju3dv/manhattan_sdf
import argparse
import json

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
import matplotlib.pyplot as plt
import cv2
import subprocess

os.environ['PYOPENGL_PLATFORM'] = 'egl'


def visualize_point_cloud(verts, dist, vis_dist=0.05):
    # dist越大越红
    data_alpha = np.clip(dist / vis_dist, 0, 1)
    data_color = plt.cm.coolwarm(data_alpha)

    pcd = trimesh.points.PointCloud(verts, colors=data_color[:, :3])
    return pcd


def nn_correspondance(verts1, verts2):
    # distance from verts2 to verts1
    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    kdtree = KDTree(verts1)
    distances, indices = kdtree.query(verts2)
    distances = distances.reshape(-1)

    return distances


def evaluate(mesh_pred, mesh_trgt, threshold=.05, down_sample=.02):
    pcd_trgt = o3d.geometry.PointCloud()
    pcd_pred = o3d.geometry.PointCloud()

    pcd_trgt.points = o3d.utility.Vector3dVector(mesh_trgt.vertices[:, :3])
    pcd_pred.points = o3d.utility.Vector3dVector(mesh_pred.vertices[:, :3])

    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)

    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    dist1 = nn_correspondance(verts_pred, verts_trgt)
    dist2 = nn_correspondance(verts_trgt, verts_pred)

    precision = np.mean((dist2 < threshold).astype('float'))
    recal = np.mean((dist1 < threshold).astype('float'))
    fscore = 2 * precision * recal / (precision + recal)
    metrics = {
        'Acc': np.mean(dist2),
        'Comp': np.mean(dist1),
        'Chamfer': np.mean(dist1)*0.5 + np.mean(dist2)*0.5,
        'Prec': precision,
        'Recal': recal,
        'F-score': fscore,
    }
    vis_pred_pcd = visualize_point_cloud(verts_pred, dist2)
    vis_gt_pcd = visualize_point_cloud(verts_trgt, dist1)
    return metrics, vis_pred_pcd, vis_gt_pcd

# hard-coded image size
H, W = 968, 1296

# load pose
def load_poses(scan_id, data_dir):
    pose_path = os.path.join(f'{data_dir}/scan{scan_id}', 'pose')
    poses = []
    pose_paths = sorted(glob.glob(os.path.join(pose_path, '*.txt')),
                        key=lambda x: int(os.path.basename(x)[:-4]))
    for pose_path in pose_paths[::10]:
        c2w = np.loadtxt(pose_path)
        if np.isfinite(c2w).any():
            poses.append(c2w)
    poses = np.array(poses)
    return poses

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


def refuse(mesh, poses, K):
    renderer = Renderer()
    mesh_opengl = renderer.mesh_opengl(mesh)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01,
        sdf_trunc=3 * 0.01,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    for pose in tqdm(poses):
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = K

        rgb = np.ones((H, W, 3))
        rgb = (rgb * 255).astype(np.uint8)
        rgb = o3d.geometry.Image(rgb)
        _, depth_pred = renderer(H, W, intrinsic, pose, mesh_opengl)
        depth_pred = o3d.geometry.Image(depth_pred)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth_pred, depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
        )
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=W, height=H, fx=fx, fy=fy, cx=cx, cy=cy)
        extrinsic = np.linalg.inv(pose)
        volume.integrate(rgbd, intrinsic, extrinsic)

    return volume.extract_triangle_mesh()

parser=argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str,default='../../data/scannet')
parser.add_argument('--scan_id',type=str,default='2')
parser.add_argument('--mesh_dir',type=str,default='/home/dawn/mesh_2449.ply')
parser.add_argument('--gt_space',action='store_true', help='mesh是否已经是gt space')
parser.add_argument('--output_dir',type=str,default='evaluation_results_single',help='path to the output folder')
parser.set_defaults(gt_space=False)
args=parser.parse_args()

# 目前是对实验中途跑出的mesh_epoch.ply进行evaluation。
scan_id=args.scan_id
data_dir = args.data_dir
mesh_dir = args.mesh_dir
gt_space = args.gt_space

all_results=[]
os.makedirs(args.output_dir,exist_ok=True)
print("Evaluating ", mesh_dir)

mesh = trimesh.load(mesh_dir)

# transform to world coordinate
cam_file = f"{data_dir}/scan{scan_id}/cameras.npz"
scale_mat = np.load(cam_file)['scale_mat_0']
if not gt_space:
    mesh.vertices = (scale_mat[:3, :3] @ mesh.vertices.T + scale_mat[:3, 3:]).T

# load pose and intrinsic for render depth
poses = load_poses(scan_id, data_dir)

intrinsic_path = os.path.join(f'{data_dir}/scan{scan_id}/intrinsic/intrinsic_color.txt')
K = np.loadtxt(intrinsic_path)[:3, :3]

mesh = refuse(mesh, poses, K)

o3d.io.write_triangle_mesh(os.path.join(args.output_dir,'eval_mesh.ply'), mesh)
mesh=trimesh.load(os.path.join(args.output_dir,'eval_mesh.ply'))

id2sceneid = {'1':'0050_00', '2':'0084_00','3':'0580_00','4':'0616_00'}

# gt_mesh = trimesh.load(f"../../data/scannet/GTmesh/scene{id2sceneid[scan_id]}_vh_clean_2.ply")
gt_mesh = os.path.join(f"{data_dir}/GTmesh_lowres", f"{id2sceneid[scan_id]}.obj")

# gt_mesh = trimesh.load(gt_mesh)
# metrics, vis_pred_pcd, vis_gt_pcd= evaluate(mesh, gt_mesh)
# mesh.export(os.path.join(args.output_dir,'eval_mesh.ply'))
# gt_mesh.export(os.path.join(args.output_dir,'gt_mesh.ply'))
# vis_pred_pcd.export(os.path.join(args.output_dir,'vis_pred_pcd.ply'))
# vis_gt_pcd.export(os.path.join(args.output_dir,'vis_gt_pcd.ply'))
# print(metrics)
# all_results.append(metrics)

cmd = f"python eval_recon.py --rec_mesh {os.path.join(args.output_dir,'eval_mesh.ply')} --gt_mesh {gt_mesh}"
print(cmd)
output = subprocess.check_output(cmd, shell=True).decode("utf-8")
output = output.replace(" ", "\t")
print(output)
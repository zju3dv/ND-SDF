# adapted from https://github.com/zju3dv/manhattan_sdf
import argparse

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
import cv2
import subprocess

os.environ['PYOPENGL_PLATFORM'] = 'egl'


def nn_correspondance(verts1, verts2):
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
        'Prec': precision,
        'Recal': recal,
        'F-score': fscore,
    }
    return metrics


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

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default='../../runs')
parser.add_argument('--data_dir', type=str, default='../../data/scannet')
parser.add_argument('--exp_name', type=str, default='scannet')
parser.add_argument('--out_dir', type=str, default='evaluation')
args = parser.parse_args()
data_dir = args.data_dir
root_dir = args.root_dir
exp_name = args.exp_name
out_dir = os.path.join(args.out_dir, exp_name)
Path(out_dir).mkdir(parents=True, exist_ok=True)
evaluation_txt_file = f"{args.out_dir}/{exp_name}.csv"
evaluation_txt_file = open(evaluation_txt_file, 'w')

scenes = ["scene0050_00", "scene0084_00", "scene0580_00", "scene0616_00"]
all_results = []
for idx, scan in enumerate(scenes):
    idx = idx + 1
    cur_exp = f"{exp_name}_{idx}"
    cur_root = os.path.join(root_dir, cur_exp)
    # use first timestamps
    dirs = sorted(os.listdir(cur_root))
    cur_root = os.path.join(cur_root, dirs[0])
    files = list(filter(os.path.isfile, glob.glob(os.path.join(cur_root, "plots/*.ply"))))

    # evalute the latest mesh
    # mesh naming format mesh_<epoch>.ply
    files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
    ply_file = files[-1]
    print(ply_file)

    mesh = trimesh.load(ply_file)

    # transform to world coordinate
    cam_file = f"{data_dir}/scan{idx}/cameras.npz"
    scale_mat = np.load(cam_file)['scale_mat_0']
    mesh.vertices = (scale_mat[:3, :3] @ mesh.vertices.T + scale_mat[:3, 3:]).T

    # load pose and intrinsic for render depth
    poses = load_poses(idx, data_dir)

    intrinsic_path = os.path.join(f'{data_dir}/scan{idx}/intrinsic/intrinsic_color.txt')
    K = np.loadtxt(intrinsic_path)[:3, :3]

    mesh = refuse(mesh, poses, K)

    # save mesh
    out_mesh_path = os.path.join(out_dir, f"{exp_name}_scan_{idx}_{scan}.ply")
    o3d.io.write_triangle_mesh(out_mesh_path, mesh)
    mesh = trimesh.load(out_mesh_path)

    # gt_mesh = os.path.join("../data/scannet/GTmesh", f"{scan}_vh_clean_2.ply")
    gt_mesh = os.path.join("{data_dir}/GTmesh_lowres", f"{scan[5:]}.obj")

    cmd = f"python eval_recon.py --rec_mesh {out_mesh_path} --gt_mesh {gt_mesh}"
    print(cmd)
    # accuracy_rec, completion_rec, precision_ratio_rec, completion_ratio_rec, fscore, normal_acc, normal_comp, normal_avg
    output = subprocess.check_output(cmd, shell=True).decode("utf-8")
    output = output.replace(" ", "\t")
    print(output)
    evaluation_txt_file.write(f"Scan{idx}\t{output}")
    evaluation_txt_file.flush()

# print all results
for scan, metric in zip(scenes, all_results):
    values = [scan] + [str(metric[k]) for k in metric.keys()]
    out = ",".join(values)
    print(out)
# data format:
# input_dir
# ├── sparse
# │   └── 0
# │       ├── cameras.bin
# │       ├── images.bin
# │       ├── points.bin
# │       ├── points3D.ply(可选)
# │       └── ...
# └── images
#     ├── image1
#     ├── image2
#     └── ...

# output: meta_data.json, resized rgb, normalized_pcd.ply, [directory of mono_depth, mono_normal, mask, uncertainty]

import json

import os,shutil,sys
import numpy as np
import trimesh
import argparse
import tqdm
import open3d as o3d
from PIL import Image
from torchvision import transforms
from plyfile import PlyElement, PlyData
from colmap.utils import read_model,qvec2rotmat
sys.path.append('../..')
from utils.visual_utils import visual_radius

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, required=False, default='/data/colmap/dorm',help='colmap root path')
    parser.add_argument('-o', '--output_dir', type=str, default='', help='output folder')
    parser.add_argument('--resize', action='store_true', help='resize images')
    parser.add_argument('--resize_h', type=int, default=384, help='resize height')
    parser.add_argument('--resize_w', type=int, default=384, help='resize width')
    parser.add_argument('--radius', type=float, default=1.0,help='radius of the scene, or scene box bound of the scene')
    parser.add_argument('--if_interactive', action='store_true', help='interactive adjust bounding box')
    parser.add_argument("--has_mono_depth", action='store_true', help="monocular depth prior ")
    parser.add_argument("--has_mono_normal", action='store_true', help="monocular normal prior")
    parser.add_argument("--has_mask", action='store_true', help="2d mask")
    parser.add_argument("--has_uncertainty", action='store_true', help="2d uncertainty")
    parser.set_defaults(resize=True, has_mono_depth=True, has_mono_normal=True, has_mask=False, has_uncertainty=False, if_interactive=False)
    args = parser.parse_args()
    args.output_dir = args.output_dir if args.output_dir else args.input_dir
    return args

def error_to_confidence(error):
    # Here smaller_beta means slower transition from 0 to 1.
    # Increasing beta raises steepness of the curve.
    beta = 1
    # confidence = 1 / np.exp(beta*error)
    confidence = 1 / (1 + np.exp(beta*error))
    return confidence

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


def generate_bbox_ply(center, radius):
    # 生成一个bounding box的点云
    # center: 中心坐标
    # radius: 半径
    # 返回：点云
    x = np.linspace(center[0] - radius, center[0] + radius, 2)
    y = np.linspace(center[1] - radius, center[1] + radius, 2)
    z = np.linspace(center[2] - radius, center[2] + radius, 2)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    x, y, z = x.flatten(), y.flatten(), z.flatten()
    p = np.vstack([x, y, z]).T

    p_r = p+np.random.randn(*p.shape)*0.01

    verts = np.vstack([p, p_r])
    trimesh.Trimesh()
    return


def get_scale_mat(pcd, poses, bound=1.0, method='points', if_align=False, if_interactive=False):
    """

    Args:
        pcd: colmap sparse 3d points: trimesh.PointCloud
        poses: 所有相机位姿c2ws
        bound: normalized sphere的半径或cube：[-bound, bound]^3
        method: 'points' or 'poses'
        if_align: 是否将bounding box对齐到坐标轴，用trimesh.bounds.oriented_bounds实现
        if_interactive: 是否交互式调整bounding box
    Returns:
        scale_mat: new-w->gt, 物体中心坐标系到世界坐标系的变换，带缩放的非欧式变换。
        normalized_pcd: 去除离群点后的物体中心系点云
        poses: c->new-w
        mask: 非离群点mask
    """
    # 去除离群点
    # TODO： 改进离群点去除方法，这里仅考虑了距离异常点。
    verts = pcd.vertices
    colors = pcd.colors
    bbox_center = np.sum(verts, axis=0) / verts.shape[0]
    dist = np.linalg.norm(verts - bbox_center, axis=1)
    threshold = np.percentile(dist, 99.5)  # 距离过滤阈值
    print(f'scene threshold: {threshold}')
    mask = dist < threshold
    to_align = np.eye(4) # to_align: 定义的世界坐标的对齐变换（场景平移到中心且与坐标轴对齐），∈SE(3)，即是欧式变换没有缩放。默认to_align=I。
    # 获得to_align，以及align后的世界场景scale和shift
    if method == 'points': # 根据3d points
        if if_align:
            filtered_pcd = trimesh.PointCloud(vertices=verts[mask], colors=colors[mask])
            to_align, extents = trimesh.bounds.oriented_bounds(filtered_pcd) # to_align: 世界坐标的对齐变换，∈SE(3)，即是欧式变换没有缩放。
            radius = np.linalg.norm(extents)/2
            shift = np.zeros(3)
        else: # 不对齐：即简单的平移和缩放
            bbox_center = np.sum(verts[mask], axis=0) / verts[mask].shape[0]
            # bbox_center = np.zeros(3)
            radius = np.max(np.linalg.norm(verts[mask] - bbox_center[None], axis=1)) * 1.1
            shift = bbox_center
    elif method == 'poses': # 根据相机位姿
        if if_align:
            centers_pcd = trimesh.PointCloud(vertices=poses[:, :3, -1])
            to_align, extents = trimesh.bounds.oriented_bounds(centers_pcd)
            radius = np.linalg.norm(extents) / 2 * 1.5# 1.5是为了保证包围盒比点云大
            shift = np.zeros(3)
        else:
            centers = poses[:, :3, -1]
            bbox_min = np.min(centers, axis=0)
            bbox_max = np.max(centers, axis=0)
            bbox_center = (bbox_min + bbox_max) / 2
            radius = np.linalg.norm(bbox_max - bbox_center) * 1.3
            shift = bbox_center
    # Optional: 交互式调整bounding box
    if if_interactive:
        shift, radius = visual_radius((to_align[:3,:3]@verts[mask].T+to_align[:3,-1:]).T, colors[mask], 1, shift)
    print(f'align:{if_align}, align transform: {to_align}\nshift: {shift}, radius: {radius}')
    # 将c2w转换为c2alignedw   = w->alignedw @ c2w
    poses = np.matmul(to_align[None].repeat(poses.shape[0], axis=0), poses)
    tmp = (poses[:,:3,:3]**2).reshape(-1,9).sum(1)
    # 对align后的相机center进行缩放平移，这里变换得到的就是json中的camtoworld了。
    poses[:, :3, -1] -= shift
    poses[:, :3, -1] = poses[:, :3, -1]/radius*bound # 新的c2w, world坐标∈[-bound, bound]^3
    # 接下来构建scale_mat，即对齐缩放平移后物体中心坐标系到GT原世界坐标系的变换。 ！！！注意：不∈SE(3)！！！
    # 很简单，缩放回去再做对齐逆变换即可
    scale_mat = np.eye(4)
    scale_mat[range(3), range(3)] = radius/bound
    scale_mat[:3, -1] = shift
    scale_mat = np.linalg.inv(to_align) @ scale_mat
    # 将离群点即~mask的坐标和颜色设置为0
    normalized_pcd = pcd.copy()
    normalized_pcd.apply_transform(np.linalg.inv(scale_mat))
    normalized_pcd = update_mesh(normalized_pcd, mask)
    print(f'after scaling&shifting, max normalized_pcd dist: {np.max(np.linalg.norm(normalized_pcd.vertices, axis=1))}')

    return scale_mat, normalized_pcd, poses, mask

def plyfile_write(verts,colors,normals,errors,confidence,out_file):
    # verts:(n,3), colors:(n,3), normals:(n,3), errors:(n,), confidence:(n,)
    # use plyfile to write
    # 将数据合并为一个结构化数组, 用于自定义ply文件vertex属性，包括基本的verts、colors、normals
    # 读入方法：
    # plydata = PlyData.read('example.ply')
    # pts = np.vstack([plydata['vertex'][name] for name in ['x', 'y', 'z']]).T
    # colors = np.vstack([plydata['vertex'][name] for name in ['red', 'green', 'blue']]).T
    # normals = np.vstack([plydata['vertex'][name] for name in ['nx', 'ny', 'nz']]).T
    # errors = np.array(plydata['vertex']['error'])
    # confidence = np.array(plydata['vertex']['confidence'])

    data = np.empty(verts.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                              ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                                              ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                                              ('error', 'f4'),('confidence', 'f4')])
    data['x'] = verts[:, 0]
    data['y'] = verts[:, 1]
    data['z'] = verts[:, 2]
    data['red'] = colors[:, 0]
    data['green'] = colors[:, 1]
    data['blue'] = colors[:, 2]
    data['nx'] = normals[:, 0]
    data['ny'] = normals[:, 1]
    data['nz'] = normals[:, 2]
    data['error'] = errors[:]
    data['confidence'] = confidence[:]

    # 使用结构化数组创建一个PlyElement实例
    element = PlyElement.describe(data, 'vertex')

    # 将PlyElement写入一个.ply文件
    PlyData([element], text=True).write(out_file)

if __name__=='__main__':

    args = get_args()

    # 1. read cameras, images, points
    cameras, images, points = read_model(os.path.join(args.input_dir, 'sparse/0'), ext=".bin")
    camera_id = list(cameras.keys())[0]
    # 获取内参
    fx, fy, cx, cy = cameras[camera_id].params
    intrinsic = np.eye(4)
    intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2] = fx, fy, cx, cy

    # crop and resize使得resize前长宽比一致
    H, W = cameras[camera_id].height, cameras[camera_id].width
    if args.resize:
        h, w= args.resize_h, args.resize_w # resize to h,w
        min_ratio = min(H / h, W / w)
        crop_size = (int(h * min_ratio), int(w * min_ratio))
        resize_trans = transforms.Compose(
            [
                transforms.CenterCrop(crop_size),
                transforms.Resize((h, w), interpolation=Image.LANCZOS),
            ]
        )
        # 调整resize后的内参
        offset_x = int((W - crop_size[1]) * 0.5)
        offset_y = int((H - crop_size[0]) * 0.5)
        intrinsic[0, 2] -= offset_x
        intrinsic[1, 2] -= offset_y
        intrinsic[0, :] /= crop_size[1]/w
        intrinsic[1, :] /= crop_size[0]/h
    else:
        h,w=H,W

    # 2. get poses
    poses = []  # c2w
    intrinsics = []
    filenames = []
    for i, image in enumerate(sorted(images.values(), key=lambda x:x.id)):
        # get pose
        qvec=image.qvec
        tvec=image.tvec
        R = qvec2rotmat(qvec)
        extrinsic=np.concatenate([R, tvec.reshape(3, 1)], 1)
        c2w = np.linalg.inv(np.concatenate([extrinsic, np.array([0, 0, 0, 1])[None]], 0))
        poses.append(c2w)

        intrinsics.append(intrinsic)
        filenames.append(image.name)
    poses = np.array(poses)

    # 3. load points, get scale_mat
    # ply_file = os.path.join(args.input_dir,'sparse','0', 'points3D.ply') # sparse点云
    # pcd = trimesh.load(ply_file)

    verts = np.array([point.xyz for point in sorted(points.values(), key=lambda x: x.id)]) # 3d points from colmap
    colors = np.array([point.rgb for point in sorted(points.values(), key=lambda x: x.id)])
    errors = np.array([point.error for point in sorted(points.values(), key=lambda x: x.id)]) # 重投影误差
    pcd = trimesh.PointCloud(vertices=verts, colors=colors) # sparse点云
    scale_mat, normalized_pcd, poses, mask = get_scale_mat(pcd, poses, bound=args.radius, method='points', if_align=True, if_interactive=args.if_interactive)
    # verts_before = verts[mask]
    # verts_after = normalized_pcd.apply_transform(scale_mat).vertices
    # normalized_pcd 添加属性error_to_confidence(error)和normal
    errors = errors[mask]
    confidence = error_to_confidence(errors)
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(normalized_pcd.vertices)
    o3d_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    normals = np.asarray(o3d_pcd.normals)
    plyfile_write(normalized_pcd.vertices, normalized_pcd.colors, normals, errors, confidence, os.path.join(args.input_dir, 'sparse/0', 'points3D_normalized.ply'))

    # TODO: 1. 读入图片然后trans进行crop和resize,; 2. 保存json文件
    scene_box = { # near、intersection_type由confs配置文件确定, far由运行时具体光线与cube/sphere的交点确定
        "aabb": [[-args.radius, -args.radius, -args.radius], [args.radius, args.radius, args.radius]],
        "near": 0.0,
        "far": 2.5,
        "radius": args.radius,
        "collider_type": "box",
    }
    data= {
        "camera_model": "OPENCV",
        "height": h,
        "width": w,
        "scene_box": scene_box,
        "has_mono_depth": args.has_mono_depth,
        "has_mono_normal": args.has_mono_normal,
        "has_mask": args.has_mask,
        "has_uncertainty": args.has_uncertainty,
        "pts_path": "sparse/0/points3D_normalized.ply", # TODO: 当前仅为sparse点云，后续可考虑dense点云(vis-MVS-net生成
        "worldtogt": scale_mat.tolist(),
    }
    # frames
    frames = []
    out_index = 0
    # 创建rgb、mono_depth、mono_normal、mask、uncertainty等文件夹
    rgb_path = os.path.join(args.output_dir, "rgb")
    os.makedirs(rgb_path, exist_ok=True)
    mono_depth_path = os.path.join(args.output_dir, "mono_depth")
    os.makedirs(mono_depth_path, exist_ok=True)
    mono_normal_path = os.path.join(args.output_dir, "mono_normal")
    os.makedirs(mono_normal_path, exist_ok=True)
    mask_path = os.path.join(args.output_dir, "mask")
    os.makedirs(mask_path, exist_ok=True)
    uncertainty_path = os.path.join(args.output_dir, "uncertainty")
    os.makedirs(uncertainty_path, exist_ok=True)

    frames = []
    for i, (pose,intrinsic,filename) in tqdm.tqdm(enumerate(zip(poses, intrinsics, filenames))):
        img = Image.open(os.path.join(args.input_dir,'images',filename))
        if args.resize:
            img = resize_trans(img)
            img.save(os.path.join(rgb_path, filename))
        else: # move
            shutil.copy(os.path.join(args.input_dir,'images',filename), os.path.join(rgb_path, filename))

        frame = {
            "rgb_path": os.path.join("rgb", filename),
            "camtoworld": pose.tolist(),
            "intrinsics": intrinsic.tolist(),
            "mono_depth_path": os.path.join("mono_depth", 'res', filename[:-4] + ".npy"),
            "mono_normal_path": os.path.join("mono_normal", 'res', filename[:-4] + ".npy"),
            # "sensor_depth_path": rgb_path.replace("_rgb.png", "_sensor_depth.npy"),
            "mask_path": os.path.join("mask", 'res', filename[:-4] + ".npy"),
            "uncertainty_path": os.path.join("uncertainty", 'res', filename[:-4] + ".npy"),
        }

        frames.append(frame)
    data["frames"] = frames

    # 保存meta_data.json
    with open(os.path.join(args.output_dir, 'meta_data.json'), 'w') as f:
        json.dump(data, f, indent=4)
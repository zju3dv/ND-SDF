'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''
# This file is largely adapted from neural-angelo, convert COLMAP results of official training split of TNT dataset to meta_data.json.
# need colmap installed
import os, tqdm, shutil
import numpy as np
import json
import sys
from PIL import Image
from pathlib import Path
from argparse import ArgumentParser
import trimesh
from torchvision import transforms

from colmap.utils import read_model,rotmat2qvec, qvec2rotmat
from colmap.database import COLMAPDatabase
sys.path.append('../..')

def create_init_files(pinhole_dict_file, db_file, out_dir):
    # Partially adapted from https://github.com/Kai-46/nerfplusplus/blob/master/colmap_runner/run_colmap_posed.py

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # create template
    with open(pinhole_dict_file) as fp:
        pinhole_dict = json.load(fp)

    template = {}
    cameras_line_template = '{camera_id} RADIAL {width} {height} {f} {cx} {cy} {k1} {k2}\n'
    images_line_template = '{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_name}\n\n'

    for img_name in pinhole_dict:
        # w, h, fx, fy, cx, cy, qvec, t
        params = pinhole_dict[img_name]
        w = params[0]
        h = params[1]
        fx = params[2]
        # fy = params[3]
        cx = params[4]
        cy = params[5]
        qvec = params[6:10]
        tvec = params[10:13]

        cam_line = cameras_line_template.format(
            camera_id="{camera_id}", width=w, height=h, f=fx, cx=cx, cy=cy, k1=0, k2=0)
        img_line = images_line_template.format(image_id="{image_id}", qw=qvec[0], qx=qvec[1], qy=qvec[2], qz=qvec[3],
                                               tx=tvec[0], ty=tvec[1], tz=tvec[2], camera_id="{camera_id}",
                                               image_name=img_name)
        template[img_name] = (cam_line, img_line)

    # read database
    db = COLMAPDatabase.connect(db_file)
    table_images = db.execute("SELECT * FROM images")
    img_name2id_dict = {}
    for row in table_images:
        img_name2id_dict[row[1]] = row[0]

    cameras_txt_lines = [template[img_name][0].format(camera_id=1)]
    images_txt_lines = []
    for img_name, img_id in img_name2id_dict.items():
        image_line = template[img_name][1].format(image_id=img_id, camera_id=1)
        images_txt_lines.append(image_line)

    with open(os.path.join(out_dir, 'cameras.txt'), 'w') as fp:
        fp.writelines(cameras_txt_lines)

    with open(os.path.join(out_dir, 'images.txt'), 'w') as fp:
        fp.writelines(images_txt_lines)
        fp.write('\n')

    # create an empty points3D.txt
    fp = open(os.path.join(out_dir, 'points3D.txt'), 'w')
    fp.close()


def convert_cam_dict_to_pinhole_dict(cam_dict, pinhole_dict_file):
    # Partially adapted from https://github.com/Kai-46/nerfplusplus/blob/master/colmap_runner/run_colmap_posed.py

    print('Writing pinhole_dict to: ', pinhole_dict_file)
    h = 1080
    w = 1920

    pinhole_dict = {}
    for img_name in cam_dict:
        W2C = cam_dict[img_name]

        # params
        fx = 0.6 * w
        fy = 0.6 * w
        cx = w / 2.0
        cy = h / 2.0

        qvec = rotmat2qvec(W2C[:3, :3])
        tvec = W2C[:3, 3]

        params = [w, h, fx, fy, cx, cy,
                  qvec[0], qvec[1], qvec[2], qvec[3],
                  tvec[0], tvec[1], tvec[2]]
        pinhole_dict[img_name] = params

    with open(pinhole_dict_file, 'w') as fp:
        json.dump(pinhole_dict, fp, indent=2, sort_keys=True)


def load_COLMAP_poses(cam_file, img_dir, tf='w2c'):
    # load img_dir namges
    names = sorted(os.listdir(img_dir))

    with open(cam_file) as f:
        lines = f.readlines()

    # C2W
    poses = {}
    for idx, line in enumerate(lines):
        if idx % 5 == 0:  # header
            img_idx, valid, _ = line.split(' ')
            if valid != '-1':
                poses[int(img_idx)] = np.eye(4)
                poses[int(img_idx)]
        else:
            if int(img_idx) in poses:
                num = np.array([float(n) for n in line.split(' ')])
                poses[int(img_idx)][idx % 5-1, :] = num

    if tf == 'c2w':
        return poses
    else:
        # convert to W2C (follow nerf convention)
        poses_w2c = {}
        for k, v in poses.items():
            poses_w2c[names[k]] = np.linalg.inv(v)
        return poses_w2c


def load_transformation(trans_file):
    with open(trans_file) as f:
        lines = f.readlines()

    trans = np.eye(4)
    for idx, line in enumerate(lines):
        num = np.array([float(n) for n in line.split(' ')])
        trans[idx, :] = num

    return trans


def align_gt_with_cam(pts, trans):
    trans_inv = np.linalg.inv(trans)
    pts_aligned = pts @ trans_inv[:3, :3].transpose(-1, -2) + trans_inv[:3, -1]
    return pts_aligned


def compute_bound(pts):
    bounding_box = np.array([pts.min(axis=0), pts.max(axis=0)])
    center = bounding_box.mean(axis=0)
    radius = np.max(np.linalg.norm(pts - center, axis=-1)) * 1.01
    return center, radius, bounding_box.T.tolist()

def export_to_my_json(args, trans, cameras, images, bounding_box, center, radius, scene_path, output_path):
    '''
    Args:
        trans: colmap2wolrd, cameras、images：colmap格式
        bounding_box: [[min_x, min_y, min_z], [max_x, max_y, max_z]], center: [center_x, center_y, center_z], radius: float [all in colmap coordinate]
    '''

    # 1. get poses and intrinsics
    intrinsic_param = np.array([camera.params for camera in cameras.values()])
    H, W = cameras[1].height, cameras[1].width
    fl_x = intrinsic_param[0][0]
    fl_y = intrinsic_param[0][1]
    cx = intrinsic_param[0][2]
    cy = intrinsic_param[0][3]
    intrinsic = np.eye(4)
    intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2] = fl_x, fl_y, cx, cy

    poses = []  # c2w
    intrinsics = []
    filenames = []
    for i, image in enumerate(sorted(images.values(), key=lambda x: x.id)):
        # get pose
        qvec = image.qvec
        tvec = image.tvec
        R = qvec2rotmat(qvec)
        extrinsic = np.concatenate([R, tvec.reshape(3, 1)], 1)
        c2w = np.linalg.inv(np.concatenate([extrinsic, np.array([0, 0, 0, 1])[None]], 0))
        poses.append(c2w)

        intrinsics.append(intrinsic)
        filenames.append(image.name)
    poses = np.array(poses)
    intrinsics = np.array(intrinsics)

    # 2. adjust poses and get scale_mat
    radius = radius/args.radius # radius normalized to 1, then scale to args.radius
    poses[:,:3, -1] = poses[:,:3, -1] - center
    poses[:,:3, -1] = poses[:,:3, -1] / radius
    scale_mat = np.eye(4)
    scale_mat[range(3), range(3)] = radius
    scale_mat[:3, -1] = center
    scale_mat = trans @ scale_mat # FIXME：是否trans转换到gt world坐标系

    # 3. resize and adjust intrinsics
    if args.resize:
        h, w = args.h, args.w  # resize to h,w
        min_ratio = min(H / h, W / w)
        crop_size = (int(h * min_ratio), int(w * min_ratio))
        resize_trans = transforms.Compose(
            [
                transforms.CenterCrop(crop_size),
                transforms.Resize((h, w), interpolation=Image.LANCZOS),
            ]
        )
    else:
        min_ratio = 1
        h, w = H, W
    offset_x = (W - int(w * min_ratio)) * 0.5
    offset_y = (H - int(h * min_ratio)) * 0.5
    intrinsics[:, 0, 2] -= offset_x
    intrinsics[:, 1, 2] -= offset_y
    # resize
    intrinsics[:, 0, :] /= int(w * min_ratio) / w
    intrinsics[:, 1, :] /= int(h * min_ratio) / h

    # 4. write to json
    scene_box = {  # near、intersection_type由confs配置文件确定, far由运行时具体光线与cube/sphere的交点确定
        "aabb": [[-args.radius, -args.radius, -args.radius], [args.radius, args.radius, args.radius]],
        "near": 0.0,
        "far": 2.5,
        "radius": args.radius,
        "collider_type": "box",
    }
    data = {
        "camera_model": "OPENCV",
        "height": h,
        "width": w,
        "scene_box": scene_box,
        "has_mono_depth": args.has_mono_depth,
        "has_mono_normal": args.has_mono_normal,
        "has_mask": args.has_mask,
        "has_uncertainty": args.has_uncertainty,
        "pts_path": "",  # TODO: 当前仅为sparse点云，后续可考虑dense点云(vis-MVS-net生成
        "worldtogt": scale_mat.tolist(),
    }
    # 创建rgb、mono_depth、mono_normal、mask、uncertainty等文件夹
    rgb_path = os.path.join(output_path, "rgb")
    os.makedirs(rgb_path, exist_ok=True)
    mono_depth_path = os.path.join(output_path, "mono_depth")
    os.makedirs(mono_depth_path, exist_ok=True)
    mono_normal_path = os.path.join(output_path, "mono_normal")
    os.makedirs(mono_normal_path, exist_ok=True)
    mask_path = os.path.join(output_path, "mask")
    os.makedirs(mask_path, exist_ok=True)
    uncertainty_path = os.path.join(output_path, "uncertainty")
    os.makedirs(uncertainty_path, exist_ok=True)

    frames = []
    for i, (pose, intrinsic, filename) in tqdm.tqdm(enumerate(zip(poses, intrinsics, filenames))):
        img = Image.open(os.path.join(scene_path, 'images', filename))
        if args.resize:
            img = resize_trans(img)
            img.save(os.path.join(rgb_path, filename))
        else:  # move
            shutil.copy(os.path.join(scene_path, 'images', filename), os.path.join(rgb_path, filename))

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
    with open(os.path.join(output_path, 'meta_data.json'), 'w') as f:
        json.dump(data, f, indent=4)

# largely adapted from neural-angelo
def init_colmap(args):
    assert args.tnt_path, "Provide path to Tanks and Temples dataset"
    scene_list = os.listdir(args.tnt_path)

    for scene in scene_list:
        if scene!=args.scene and scene != 'All':
            continue
        scene_path = os.path.join(args.tnt_path, scene)

        if not os.path.exists(f"{scene_path}/images_raw"):
            raise Exception(f"'images_raw` folder cannot be found in {scene_path}."
                            "Please check the expected folder structure in DATA_PREPROCESSING.md")

        # extract features
        os.system(f"colmap feature_extractor --database_path {scene_path}/database.db \
                --image_path {scene_path}/images_raw \
                --ImageReader.camera_model=RADIAL \
                --SiftExtraction.use_gpu=true \
                --SiftExtraction.num_threads=32 \
                --ImageReader.single_camera=true"
                  )

        # match features
        os.system(f"colmap sequential_matcher \
                --database_path {scene_path}/database.db \
                --SiftMatching.use_gpu=true"
                  )

        # read poses
        poses = load_COLMAP_poses(os.path.join(scene_path, f'{scene}_COLMAP_SfM.log'),
                                  os.path.join(scene_path, 'images_raw'))

        # convert to colmap files
        pinhole_dict_file = os.path.join(scene_path, 'pinhole_dict.json')
        convert_cam_dict_to_pinhole_dict(poses, pinhole_dict_file)

        db_file = os.path.join(scene_path, 'database.db')
        sfm_dir = os.path.join(scene_path, 'sparse')
        create_init_files(pinhole_dict_file, db_file, sfm_dir)

        # bundle adjustment
        os.system(f"colmap point_triangulator \
                --database_path {scene_path}/database.db \
                --image_path {scene_path}/images_raw \
                --input_path {scene_path}/sparse \
                --output_path {scene_path}/sparse \
                --Mapper.tri_ignore_two_view_tracks=true"
                  )
        os.system(f"colmap bundle_adjuster \
                --input_path {scene_path}/sparse \
                --output_path {scene_path}/sparse \
                --BundleAdjustment.refine_extrinsics=false"
                  )

        # undistortion
        os.system(f"colmap image_undistorter \
            --image_path {scene_path}/images_raw \
            --input_path {scene_path}/sparse \
            --output_path {scene_path} \
            --output_type COLMAP")

        # read for bounding information
        trans = load_transformation(os.path.join(scene_path, f'{scene}_trans.txt'))
        pts = trimesh.load(os.path.join(scene_path, f'{scene}.ply'))
        pts, colors = pts.vertices, pts.visual.vertex_colors
        pts_aligned = align_gt_with_cam(pts, trans)
        center, radius, bounding_box = compute_bound(pts_aligned[::100])
        # aligned_pcd = trimesh.PointCloud(pts_aligned, colors=colors)
        # aligned_pcd.export(os.path.join(scene_path, f'{scene}_aligned.ply'))
        # colmap to json
        cameras, images, points3D = read_model(os.path.join(args.tnt_path, scene, 'sparse'), ext='.bin')
        output_path = args.output_path if args.output_path else scene_path
        export_to_my_json(args, trans, cameras, images, bounding_box, list(center), radius, scene_path, output_path)
        print('Writing data to json file: ', os.path.join(scene_path, 'meta_data.json'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tnt_path', type=str, default='/data/tankandtemples/training', help='Path to tanks and temples dataset')
    parser.add_argument('--scene', type=str, default='Truck',
                        choices=['Barn', 'Caterpillar', 'Church', 'Courthouse', 'Ignatius', 'MeetingRoom','Truck', 'All'])  # training
    parser.add_argument('--split', type=str, default='train', choices=['train'])
    # 1. test-split: advanced, intermediate,
    # choices of advanced: ['Auditorium', 'Ballroom', 'Courtroom', 'Museum', 'Palace', 'Temple']
    # choices of intermediate: ['Barn', 'Cafe', 'DiningRoom', 'LivingRoom', 'Office', 'Restaurant']
    # 2. train-split：training
    # choices of training：['Barn', 'Caterpillar', 'Church', 'Courthouse', 'Ignatius', 'MeetingRoom', 'Truck']
    parser.add_argument("--output_path", type=str, default='', help="path to output")
    parser.add_argument('--resize', action='store_true', help='resize images')
    parser.add_argument("--h", type=int, default=540, help="resize height")
    parser.add_argument("--w", type=int, default=960, help="resize width")
    parser.add_argument("--radius", type=float, default=1, help="radius of the scene, or scene box bound of the scene")
    parser.add_argument("--has_mono_depth", action='store_true', help="monocular depth prior ")
    parser.add_argument("--has_mono_normal", action='store_true', help="monocular normal prior")
    parser.add_argument("--has_mask", action='store_true', help="2d mask")
    parser.add_argument("--has_uncertainty", action='store_true', help="2d uncertainty")
    parser.set_defaults(resize=True, has_mono_depth=True, has_mono_normal=True, has_mask=False, has_uncertainty=False)
    args = parser.parse_args()

    init_colmap(args)
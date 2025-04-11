import argparse
import glob
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tqdm
import trimesh
from PIL import Image
from torchvision import transforms

# copy from vis-mvsnet
def load_cam(file: str):
    """ read camera txt file """
    cam = np.zeros((2, 4, 4))
    with open(file) as f:
        words = f.read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]

    return cam

# This is for test split of TNT dataset, training split is adapted from neural-angelo
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str,default='/mnt/xishu/ND-SDF/data/tankandtemples/advanced')
parser.add_argument('--scene',type=str,default='Ballroom', choices=['Auditorium', 'Ballroom', 'Courtroom', 'Museum', 'Palace', 'Temple', # advanced
                                                                    'Family', 'Francis', 'Horse', 'Lighthouse', 'M60', 'Panther', 'Playground', 'Train']) # intermediate
parser.add_argument('--split',type=str,default='test', choices=['test'])
# 1. test-split: advanced, intermediate,
# choices of advanced: ['Auditorium', 'Ballroom', 'Courtroom', 'Museum', 'Palace', 'Temple']
# choices of intermediate: ['Family', 'Francis', 'Horse', 'Lighthouse', 'M60', 'Panther', 'Playground', 'Train']
# 2. train-split：training
# choices of training：['Barn', 'Caterpillar', 'Church', 'Courthouse', 'Ignatius', 'MeetingRoom', 'Truck']
parser.add_argument("--output_path", type=str, default='', help="path to output")
parser.add_argument('--resize', action='store_true', help='if resize images')
parser.add_argument("--h",  type=int, default=384, help="resize height")
parser.add_argument("--w",  type=int, default=384, help="resize width")
# radius: 'Auditorium': 4, 'Ballroom': 2.5, 'Courtroom': 1.5, 'Museum': 2, 'Palace': ?, 'Temple': ?
parser.add_argument("--radius", type=float, default=2.5, help="radius of the scene, or scene box bound of the scene")
parser.add_argument("--has_mono_depth", action='store_true', help="monocular depth prior ")
parser.add_argument("--has_mono_normal", action='store_true', help="monocular normal prior")
parser.add_argument("--has_mask", action='store_true', help="2d mask")
parser.add_argument("--has_uncertainty", action='store_true', help="2d uncertainty")
parser.set_defaults(resize=False, has_mono_depth=True, has_mono_normal=True, has_mask=False, has_uncertainty=False)

args = parser.parse_args()

H, W = 1080, 1920
if args.resize:
    h, w= args.h, args.w  # resize to h,w
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

input_path = os.path.join(args.data_dir,f'{args.scene}')
output_path = args.output_path if args.output_path else input_path

os.makedirs(output_path, exist_ok=True)

# load scannet数据的sens结构color
color_path = os.path.join(input_path, "images_raw")
color_paths = sorted(glob.glob(os.path.join(color_path, "*.jpg")), key=lambda x: int(os.path.basename(x)[:-4]))

# load pose, intrinsic
cam_path = os.path.join(input_path, "cams")
cams = sorted(glob.glob(os.path.join(cam_path, "*.txt")), key=lambda x: int(os.path.basename(x)[:-8]))
poses = []
intrinsics = []
for cam in cams:
    extrinsic = load_cam(cam)[0]
    intrinsic = load_cam(cam)[1]
    intrinsics.append(intrinsic)
    pose = np.linalg.inv(extrinsic)
    poses.append(pose)
poses = np.array(poses)
certers = poses[:, :3, -1]
certers_norm = np.linalg.norm(certers, axis=1)
print('max center norm:', np.max(certers_norm))
poses[:, :3, -1] = poses[:, :3, -1] / args.radius
intrinsics = np.array(intrinsics)
scale_mat = np.eye(4).astype(np.float32) * args.radius

# center crop by 2 * image_size
offset_x = (W - int(w * min_ratio)) * 0.5
offset_y = (H - int(h * min_ratio)) * 0.5
intrinsics[:, 0, 2] -= offset_x
intrinsics[:, 1, 2] -= offset_y
# resize from 384*2 to 384
intrinsics[:, 0, :] /= int(w * min_ratio) / w
intrinsics[:, 1, :] /= int(h * min_ratio) / h


frames = []
out_index = 0
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
for idx, (pose, image_path) in tqdm.tqdm(enumerate(zip(poses, color_paths)),total=len(color_paths)):
    filename = os.path.basename(image_path)
    target_image = os.path.join(rgb_path, filename)
    img = Image.open(image_path)
    if args.resize:
        img_tensor = resize_trans(img)
        img_tensor.save(target_image)
    else:
        import shutil
        shutil.copy(image_path, target_image)

    frame = {
        "rgb_path": os.path.join("rgb", filename),
        "camtoworld": pose.tolist(),
        "intrinsics": intrinsics[idx].tolist(),
        "mono_depth_path": os.path.join("mono_depth", 'res', filename[:-4]+ ".npy"),
        "mono_normal_path": os.path.join("mono_normal", 'res', filename[:-4]+ ".npy"),
        # "sensor_depth_path": rgb_path.replace("_rgb.png", "_sensor_depth.npy"),
        "mask_path": os.path.join("mask", 'res', filename[:-4]+ ".npy"),
        "uncertainty_path": os.path.join("uncertainty", 'res', filename[:-4]+ ".npy"),
    }

    frames.append(frame)
    out_index += 1

# scene bbox for the scannet scene
scene_box = {
    "aabb": [[-args.radius, -args.radius, -args.radius], [args.radius, args.radius, args.radius]],
    "near": 0.0,
    "far": 2.5,
    "radius": args.radius,
    "collider_type": "box",
}

# meta data
output_data = {
    "camera_model": "OPENCV",
    "height": h,
    "width": w,
    "has_mono_depth": args.has_mono_depth,
    # "has_sensor_depth": True,
    "has_mono_normal": args.has_mono_normal,
    "has_mask": args.has_mask,
    "has_uncertainty": args.has_uncertainty,
    "pairs": None,
    "worldtogt": scale_mat.tolist(),
    "scene_box": scene_box,
}

output_data["frames"] = frames

# save as json
with open(os.path.join(output_path, "meta_data.json"), "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4)

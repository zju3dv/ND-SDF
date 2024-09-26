# dtu数据集整理自neural-angelo仓库。
# curl -L -o data.zip https://www.dropbox.com/sh/w0y8bbdmxzik3uk/AAAaZffBiJevxQzRskoOYcyja?dl=1
import argparse
import glob
import json
import os

import cv2
import shutil
import numpy as np
import PIL
import tqdm
import trimesh
from PIL import Image
from torchvision import transforms

def load_K_Rt_from_P(filename, P=None):
    # This function is borrowed from IDR: https://github.com/lioryariv/idr
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str,default='/data/angelo/dtu')
parser.add_argument('--id',type=str,default='122')
parser.add_argument("--output_path", type=str, default='', help="path to output")
parser.add_argument("--h",  type=int, default=600, help="resize height")
parser.add_argument("--w",  type=int, default=800, help="resize width")
parser.add_argument('--resize', action='store_true', help='resize images')
parser.add_argument("--radius", type=float, default=1.0, help="radius of the scene, or scene box bound of the scene")
parser.add_argument("--has_mono_depth", action='store_true', help="monocular depth prior ")
parser.add_argument("--has_mono_normal", action='store_true', help="monocular normal prior")
parser.add_argument("--has_mask", action='store_true', help="2d mask")
parser.add_argument("--has_uncertainty", action='store_true', help="2d uncertainty")
parser.set_defaults(resize=True, has_mono_depth=True, has_mono_normal=True, has_mask=False, has_uncertainty=False)

args = parser.parse_args()

H, W = 1200, 1600
if args.resize:
    h, w= args.h, args.w
else:
    h, w = H, W
min_ratio = int(min(H/h, W/w))
crop_size = (int(h*min_ratio), int(w*min_ratio))
trans = transforms.Compose(
    [
        transforms.CenterCrop(crop_size),
        transforms.Resize((h, w), interpolation=PIL.Image.LANCZOS),
    ]
)

input_path = os.path.join(args.data_dir, f'scene{args.id}')
output_path = args.output_path if args.output_path else input_path
os.makedirs(output_path, exist_ok=True)

# camera_param: scale_mati, world_mati, camera_mati
camera_param = dict(np.load(os.path.join(input_path, 'cameras_sphere.npz')))



color_path = os.path.join(input_path, "image")
color_paths = sorted(glob.glob(os.path.join(color_path, "*.png")), key=lambda x: int(os.path.basename(x)[:-4]))


# load intrinsic and pose
poses = []
intrinsics = []
scale_mat = None
for idx, filename in enumerate(color_paths):
    world_mat = camera_param['world_mat_%d' % idx]
    scale_mat = camera_param['scale_mat_%d' % idx]
    # TODO: 这里只改变了scale_mat应该是没有问题的，线暂时都以bound=2来运行，待测试。
    scale_mat[range(3),range(3)] = scale_mat[range(3),range(3)] / args.radius

    # scale and decompose
    P = world_mat @ scale_mat
    P = P[:3, :4]
    camera_intrinsic, c2w = load_K_Rt_from_P(None, P)
    intrinsics.append(camera_intrinsic)
    poses.append(c2w)
poses = np.array(poses)

# 调整resize后的camera_intrinsic
for i in range(len(intrinsics)):
    # center crop
    offset_x = (W - crop_size[1]) * 0.5
    offset_y = (H - crop_size[0]) * 0.5
    intrinsics[i][0, 2] -= offset_x
    intrinsics[i][1, 2] -= offset_y

    # resize_factor = 1.0 / min_ratio
    intrinsics[i][0, :] /= (crop_size[1]/w)
    intrinsics[i][1, :] /= (crop_size[0]/h)


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
    if not args.resize:
        shutil.copy(image_path, target_image)
    else:
        img = Image.open(image_path)
        img_tensor = trans(img)
        img_tensor.save(target_image)

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

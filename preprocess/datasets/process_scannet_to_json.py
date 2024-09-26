# adapted from https://github.com/autonomousvision/sdfstudio/blob/master/scripts/datasets/process_scannet_to_sdfstudio.py
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

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str,default='/mnt/xishu/snu+da')
parser.add_argument('--id',type=str,default='0721_00')
parser.add_argument("--output_path", type=str, default='', help="path to output")
parser.add_argument("--h",  type=int, default=480, help="resize height")
parser.add_argument("--w",  type=int, default=640, help="resize width")
parser.add_argument("--radius", type=float, default=1.0, help="radius of the scene, or scene box bound of the scene")
parser.add_argument("--has_mono_depth", action='store_true', help="monocular depth prior ")
parser.add_argument("--has_mono_normal", action='store_true', help="monocular normal prior")
parser.add_argument("--has_mask", action='store_true', help="2d mask")
parser.add_argument("--has_uncertainty", action='store_true', help="2d uncertainty")
parser.set_defaults(has_mono_depth=True, has_mono_normal=True, has_mask=True, has_uncertainty=True)

args = parser.parse_args()

H, W = 968, 1296
h, w= args.h, args.w
min_ratio = int(min(H/h, W/w))
crop_size = (h*min_ratio, w*min_ratio)
trans = transforms.Compose(
    [
        transforms.CenterCrop(crop_size),
        transforms.Resize((h, w), interpolation=PIL.Image.LANCZOS),
    ]
)

# depth_trans_totensor = transforms.Compose(
#     [
#         transforms.Resize([968, 1296], interpolation=PIL.Image.NEAREST),
#         transforms.CenterCrop(image_size * 2),
#         transforms.Resize(image_size, interpolation=PIL.Image.NEAREST),
#     ]
# )
input_path = os.path.join(args.data_dir,f'scene{args.id}')
output_path = args.output_path if args.output_path else input_path

os.makedirs(output_path, exist_ok=True)

# load scannet数据的sens结构color
color_path = os.path.join(input_path, "color")
color_paths = sorted(glob.glob(os.path.join(color_path, "*.jpg")), key=lambda x: int(os.path.basename(x)[:-4]))

# # load sensor depth
# depth_path = input_path / "depth"
# depth_paths = sorted(glob.glob(os.path.join(depth_path, "*.png")), key=lambda x: int(os.path.basename(x)[:-4]))


# load intrinsic
intrinsic_path = os.path.join(input_path, "intrinsic", "intrinsic_color.txt")
camera_intrinsic = np.loadtxt(intrinsic_path)

# load pose
pose_path = os.path.join(input_path, "pose")
poses = []
pose_paths = sorted(glob.glob(os.path.join(pose_path, "*.txt")), key=lambda x: int(os.path.basename(x)[:-4]))
for pose_path in pose_paths:
    c2w = np.loadtxt(pose_path)
    poses.append(c2w)
poses = np.array(poses)

# deal with invalid poses
valid_poses = np.isfinite(poses).all(axis=2).all(axis=1)
min_vertices = poses[:, :3, 3][valid_poses].min(axis=0)
max_vertices = poses[:, :3, 3][valid_poses].max(axis=0)

center = (min_vertices + max_vertices) / 2.0
scale = 2.0*args.radius / (np.max(max_vertices - min_vertices) + 3.0)
print(center, scale)

# we should normalize pose to unit cube
poses[:, :3, 3] -= center
poses[:, :3, 3] *= scale

# inverse normalization
scale_mat = np.eye(4).astype(np.float32)
scale_mat[:3, 3] -= center
scale_mat[:3] *= scale
scale_mat = np.linalg.inv(scale_mat)

# gt_mesh = trimesh.load("/data/scannet/scans/scene0616_00/scene0616_00_vh_clean_2.ply")
# gt_mesh.apply_transform(np.linalg.inv(scale_mat))
# verts=gt_mesh.vertices
# max_vert = np.max(verts, axis=0)
# min_vert = np.min(verts, axis=0)

# center crop by 2 * image_size
offset_x = (W - w * min_ratio) * 0.5
offset_y = (H - h * min_ratio) * 0.5
camera_intrinsic[0, 2] -= offset_x
camera_intrinsic[1, 2] -= offset_y
# resize from 384*2 to 384
resize_factor = 1.0 / min_ratio
camera_intrinsic[:2, :] *= resize_factor

K = camera_intrinsic

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
for idx, (valid, pose, image_path) in tqdm.tqdm(enumerate(zip(valid_poses, poses, color_paths)),total=len(color_paths)):

    if idx % 10 != 0:
        continue
    if not valid:
        continue
    filename = f"{out_index:06d}.png"
    target_image = os.path.join(rgb_path, filename)
    img = Image.open(image_path)
    img_tensor = trans(img)
    img_tensor.save(target_image)

    # # 这里sensor depth暂时不处理
    # # load depth
    # target_depth_image = output_path / f"{out_index:06d}_sensor_depth.png"
    # depth = cv2.imread(depth_path, -1).astype(np.float32) / 1000.0
    #
    # depth_PIL = Image.fromarray(depth)
    # new_depth = depth_trans_totensor(depth_PIL)
    # new_depth = np.asarray(new_depth)
    # # scale depth as we normalize the scene to unit box
    # new_depth *= scale
    # plt.imsave(target_depth_image, new_depth, cmap="viridis")
    # np.save(str(target_depth_image).replace(".png", ".npy"), new_depth)

    frame = {
        "rgb_path": os.path.join("rgb", filename),
        "camtoworld": pose.tolist(),
        "intrinsics": K.tolist(),
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

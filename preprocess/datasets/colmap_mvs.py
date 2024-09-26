import os, json
import numpy as np
import subprocess
import shutil
from pathlib import Path

# 项目目录
project_dir = Path('/data/scannetpp/036bce3393')
colmap_workspace = project_dir / 'colmap'
image_dir = colmap_workspace / 'input'
database_path = colmap_workspace / 'database.db'
sparse_dir = colmap_workspace / 'sparse'
dense_dir = colmap_workspace / 'dense'

# 确保工作目录存在
colmap_workspace.mkdir(parents=True, exist_ok=True)

# 清空之前的结果
if database_path.exists():
    os.remove(database_path)
if sparse_dir.exists():
    shutil.rmtree(sparse_dir)
if dense_dir.exists():
    shutil.rmtree(dense_dir)

# 确保必要的目录存在
sparse_dir.mkdir(parents=True, exist_ok=True)
dense_dir.mkdir(parents=True, exist_ok=True)

# 加载相机内参和位姿
data_json = project_dir / 'meta_data.json'
with open(data_json, 'r') as f:
    data = json.load(f)

# poses = np.load(project_dir / 'scene' / 'poses.npy')
# intrinsics = np.load(project_dir / 'scene' / 'intrinsics.npy')
poses = []
intrinsics = []
for idx, frame in enumerate(data['frames']):
    pose = np.array(frame["camtoworld"], dtype=np.float32)
    intrinsic = np.array(frame["intrinsics"], dtype=np.float32)
    poses.append(pose)
    intrinsics= intrinsic
poses = np.array(poses)
intrinsics = np.array(intrinsics)
# 步骤1：创建COLMAP数据库并导入图像
subprocess.run([
    'colmap', 'database_creator',
    '--database_path', str(database_path)
])

subprocess.run([
    'colmap', 'image_registrator',
    '--database_path', str(database_path),
    '--image_path', str(image_dir)
])

from colmap.utils import rotmat2qvec
# 步骤2：导入位姿和内参
def import_cameras_and_images(database_path, image_dir, intrinsics, poses):
    import sqlite3

    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    # 插入相机参数
    cursor.execute(
        "INSERT INTO cameras (camera_id, model, width, height, params, prior_focal_length) VALUES (?, ?, ?, ?, ?, ?)",
        (1, 'PINHOLE', int(intrinsics[0, 2] * 2), int(intrinsics[1, 2] * 2), ','.join(map(str, intrinsics[:2, :3].flatten())), 1)
    )

    # 插入图像和位姿
    for i, image_path in enumerate(sorted(image_dir.iterdir())):
        extrinsic = np.linalg.inv(poses[i])
        q = rotmat2qvec(extrinsic[:3, :3])
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        tx,ty,tz = extrinsic[0,3],extrinsic[1,3],extrinsic[2,3]

        cursor.execute(
            "INSERT INTO images (image_id, name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (i + 1, str(image_path.name), 1, qw, qx, qy, qz, tx, ty, tz)
        )

    connection.commit()
    connection.close()


import_cameras_and_images(database_path, image_dir, intrinsics, poses)

# 步骤3：稀疏重建
subprocess.run([
    'colmap', 'mapper',
    '--database_path', str(database_path),
    '--image_path', str(image_dir),
    '--output_path', str(sparse_dir)
])

# 步骤4：稠密重建
# 将图像从colmap/input复制到colmap/dense/images
dense_image_dir = dense_dir / 'images'
shutil.copytree(image_dir, dense_image_dir)

subprocess.run([
    'colmap', 'image_undistorter',
    '--image_path', str(dense_image_dir),
    '--input_path', str(sparse_dir / '0'),
    '--output_path', str(dense_dir)
])

subprocess.run([
    'colmap', 'patch_match_stereo',
    '--workspace_path', str(dense_dir),
    '--workspace_format', 'COLMAP',
    '--PatchMatchStereo.geom_consistency', 'false'
])

subprocess.run([
    'colmap', 'stereo_fusion',
    '--workspace_path', str(dense_dir),
    '--workspace_format', 'COLMAP',
    '--input_type', 'geometric',
    '--output_path', str(dense_dir / 'fused.ply')
])

print(f"Dense reconstruction saved to {dense_dir / 'fused.ply'}")

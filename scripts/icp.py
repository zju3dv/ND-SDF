# icp迭代最近点算法对齐两个mesh
M1 = "/home/dawn/1/scannet/helixsurf_meshes/0050_00_world.ply"

# M2 = "/home/dawn/1/scannet/monosdf/mlp/scan1_gt.ply"
M2 = "/home/dawn/1/scannet/ours/scannet_omni_384-21-2_1_512_latest.ply"

import trimesh
import numpy as np
import open3d as o3d

def mesh_to_point_cloud(mesh, sample_points=50000):
    # 从网格中采样点云
    points = mesh.sample(sample_points)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud

def icp_align(pcd1, pcd2, threshold=10):
    # ICP
    trans_init = np.identity(4)  # 初始变换矩阵

    # 执行ICP算法
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd1, pcd2, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    T = reg_p2p.transformation
    return T

import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.spatial.distance import directed_hausdorff

mesh1 = trimesh.load(M1)
mesh2 = trimesh.load(M2)
# 将网格转换为点云并下采样
pcd1 = mesh_to_point_cloud(mesh1)
pcd2 = mesh_to_point_cloud(mesh2)
print("Point Cloud 1 has {} points".format(len(pcd1.points)))
print("Point Cloud 2 has {} points".format(len(pcd2.points)))
print('max distance:', directed_hausdorff(np.asarray(pcd1.points), np.asarray(pcd2.points))[0])
# 可视化原始点云
o3d.visualization.draw_geometries([pcd1, pcd2], window_name="Original Point Clouds")

times= 3
threshold = 4
T_end = np.identity(4)
for i in range(times):
    T = icp_align(pcd1, pcd2, threshold)
    pcd1 = pcd1.transform(T)
    print(f'iteration {i+1}:')
    print(f'error: {directed_hausdorff(np.asarray(pcd1.points), np.asarray(pcd2.points))[0]}')
    threshold = threshold/2
    T_end = np.dot(T, T_end) # 累积变换
    o3d.visualization.draw_geometries([pcd1, pcd2], window_name=f"Aligned Point Clouds {i+1}")



# 输出变换矩阵T_end
print("Transformation Matrix:")
print(T_end)

mesh1.apply_transform(T_end)
aligned_pcd1 = mesh_to_point_cloud(mesh1)
o3d.visualization.draw_geometries([aligned_pcd1, pcd2], window_name="Aligned Point Clouds")
mesh1.export(M1.replace('.ply', '_aligned.ply'))
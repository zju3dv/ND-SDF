import json
import os
import sys
sys.path.append('../..')
from utils.visual_utils import *
import cv2
import imageio
import numpy as np
import pyrender
import trimesh
from tqdm import tqdm

# pyrender没办法渲染normal。
################################################################ Hyperparameters ################################################################
mesh_dir = '/data/projects/implicit_reconstruction/scripts/tnt8-1_1_768_latest.ply'
data_dir = '/data/monosdf/tnt_advanced/scan1'
out_dir = 'scan1'
gt_space = True # 输入的mesh是否已经是gt space
textured = False
pose_type ='original' # 'original' or 'circle'
if_interpolate = True
num_interpolations = 0
inter_type = 'uniform' # 'uniform' or 'sin'
fps = 60
width= 640
height= 480
merge_type = 'cat' # 'cat' or 'half', 其他表示输出rgb和depth合并的结果
################################################################ Hyperparameters ################################################################

mesh = trimesh.load_mesh(mesh_dir)
print('Loaded mesh from {}'.format(mesh_dir))
os.makedirs(out_dir,exist_ok=True)
# 加载json文件
poses = []
with open(os.path.join(data_dir, 'meta_data.json'), 'r') as f:
    data = json.load(f)
for frame in data["frames"]:
    pose = np.array(frame["camtoworld"], dtype=np.float32)
    poses.append(pose)
scale_mat = np.array(data["worldtogt"], dtype=np.float32)
poses = np.array(poses)
if pose_type=='circle': # FIXME：轴和scene不对齐，需要调整 [√，现可自定义相机x轴和z轴]
    if gt_space:
        normalized_mesh = mesh.apply_transform(np.linalg.inv(scale_mat))
        verts, colors = normalized_mesh.vertices, normalized_mesh.visual.vertex_colors
    else:
        verts, colors = mesh.vertices, mesh.visual.vertex_colors
    visual_radius(verts,colors,radius=1,center=np.array([0,0,0],dtype=np.float32)) # visualize pcd and bounding box
    poses = generate_circle_pose(center=np.array([0,0,0],dtype=np.float32),elevation=30,radius=1.5,num_poses=poses.shape[0], camera_x_axis=[0,1,0],camera_z_axis=[0,0,1])

bound = data['scene_box']['radius']
gray = 0.69999999999999996 if not textured else 1.
# gray = 1.
if if_interpolate:
    print('Will interpolate {} frames between each pair of poses'.format(num_interpolations))
    poses = interpolate_poses(poses, num_interpolations=num_interpolations, inter_type='uniform') # 实际上是c2n
    # Rs = poses[:,:3,:3]
    # Ts = poses[:,:3,3]
    # sum_R = np.sum((Rs**2).reshape(-1,9),axis=1)
    # sum_T = np.sum((Ts**2),axis=1)

if gt_space:
    poses = np.matmul(scale_mat[None].repeat(poses.shape[0],axis=0),poses) # c2n/c2w-> c2gt

# material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.69999999999999996, 0.69999999999999996, 0.69999999999999996, 1])
if textured:
    m = pyrender.Mesh.from_trimesh(mesh)
else:
    mesh.visual.vertex_colors = [178,178,178,255] # 灰色0.69999999999999996
    m = pyrender.Mesh.from_trimesh(mesh)


# 创建一个场景
scene = pyrender.Scene()
scene.add(m) # 添加网格

# 相机
pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414) # 创建一个透视相机
oc= pyrender.OrthographicCamera(xmag=1.0, ymag=1.0) # 创建一个正交相机

camera_node = pyrender.Node(camera=pc, matrix=np.eye(4)) # 创建一个相机节点
scene.add_node(camera_node) # 将相机节点添加到场景中

# 创建光
pl = pyrender.PointLight(color=[gray, gray, gray], intensity=4) # 创建点光源
dl = pyrender.DirectionalLight(color=[gray, gray, gray], intensity=4) # 创建平行直线光源

# 创建光节点
dl_node = pyrender.Node(light=dl, matrix=np.eye(4))
pl_pos1 = [0, 0, bound]
pl_pos2 = [0, 0, -bound]
pl_pos3 = [bound, bound, 0]
pl_pos4 = [-bound, -bound, 0]
def pack_light_pos(pos):
    return np.concatenate([np.concatenate([np.eye(3),np.array(pos)[:,None]],axis=1),np.array([[0,0,0,1]])],axis=0)
# pl_node0 = pyrender.Node(light=pl, matrix=pack_light_pos([0, 0, 0]))
pl_node1 = pyrender.Node(light=pl, matrix=pack_light_pos(pl_pos1))
pl_node2 = pyrender.Node(light=pl, matrix=pack_light_pos(pl_pos2))
pl_node3 = pyrender.Node(light=pl, matrix=pack_light_pos(pl_pos3))
pl_node4 = pyrender.Node(light=pl, matrix=pack_light_pos(pl_pos4))
# 将光源添加到场景中
scene.add_node(dl_node)
scene.add_node(pl_node1),scene.add_node(pl_node2),scene.add_node(pl_node3),scene.add_node(pl_node4)

# 创建offscreen渲染器
renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)



depths = []
colors=[]
render_dir = os.path.join(out_dir,'render_images')
os.makedirs(render_dir,exist_ok=True)
os.makedirs(render_dir+'/color',exist_ok=True)
os.makedirs(render_dir+'/depth',exist_ok=True)
# 遍历所有相机位姿，渲染每一帧并写入视频
for pose in tqdm(poses,total=len(poses),desc='Rendering rgb and depth',unit='frames',colour='blue'):
    # pose由OPENCV->OPENGL
    pose[:3,1:3] = -pose[:3,1:3]
    # 设置相机的位置和方向
    scene.set_pose(camera_node, pose=pose)
    scene.set_pose(dl_node, pose=pose)


    # 渲染场景
    color, depth = renderer.render(scene)
    cv2.imwrite(os.path.join(render_dir+'/color',f'{len(depths):05d}.png'),color)
    color=cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    depths.append(depth)

    # 将渲染的帧添加到视频
    colors.append(color)

# write out the depth and color rgb+depth
# 设置视频和帧率
fourcc = cv2.VideoWriter_fourcc(*'MPEG') # 保存视频的编码: XVID, MPEG
color_out = cv2.VideoWriter(os.path.join(out_dir,os.path.basename(mesh_dir).split('.')[0]+'_rgb.avi'), fourcc, fps, (width, height))
depth_out = cv2.VideoWriter(os.path.join(out_dir,os.path.basename(mesh_dir).split('.')[0]+'_depth.avi'), fourcc, fps, (width, height), isColor=True) # isColor=True: 彩色视频, isColor=False: 灰度视频
if merge_type=='cat' or merge_type=='half':
    cat_out = cv2.VideoWriter(os.path.join(out_dir,os.path.basename(mesh_dir).split('.')[0]+f'_{merge_type}.avi'), fourcc, fps, (width*(2 if merge_type == 'cat' else 1), height), isColor=True)

depth_min = np.min(np.array(depths))
depth_max = np.max(np.array(depths))
for i,depth in tqdm(enumerate(depths),total=len(depths),desc='write to video...',unit='frames',colour='green'):
    # depth: float[0,1]单位m -> 16位无符号整数[0,65535]单位mm
    # depth = (depth * 1000).astype(np.uint16)
    depth = (depth-depth_min)/(depth_max-depth_min)
    depth_image_color = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.imwrite(os.path.join(render_dir+'/depth',f'{i:05d}.png'),depth_image_color)

    depth_out.write(depth_image_color)
    color_out.write(colors[i])
    if merge_type=='cat':
        cat_image = cv2.hconcat([colors[i],depth_image_color])
        cat_out.write(cat_image)
    elif merge_type=='half':
        mid = int(width/2)
        cat_image = cv2.hconcat([colors[i][:,:mid],depth_image_color[:,mid:]])
        cat_out.write(cat_image)

# 完成视频写入
# writer.close()
color_out.release()
depth_out.release()

# 清理渲染器资源
renderer.delete()
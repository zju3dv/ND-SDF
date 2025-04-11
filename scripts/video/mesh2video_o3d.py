import json
import os
import sys
sys.path.append('../..')
from utils.visual_utils import *
import cv2
import numpy as np
import open3d.cuda.pybind.io

from scipy.linalg import expm
import open3d as o3d
from tqdm import tqdm

def get_poses(data_dir):
    intrinsics = []
    poses = []
    with open(os.path.join(data_dir, 'meta_data.json'), 'r') as f:
        data = json.load(f)
    for frame in data["frames"]:
        pose = np.array(frame["camtoworld"], dtype=np.float32)
        poses.append(pose)
        intrinsics.append(np.array(frame["intrinsics"], dtype=np.float32))
    scale_mat = np.array(data["worldtogt"], dtype=np.float32)
    return np.array(poses),scale_mat, np.array(intrinsics), data["width"], data["height"]

################################################################ Hyperparameters ################################################################
mesh_dir = '/home/dawn/1-ND-SDF/scannetpp/e0/e0_notop.ply'
data_dir = '/data/monosdf/scannet/scan4'
scene = 'e0_blender'
output_video_dir = f'{scene}/rgb+normal'
output_image_dir = f'{scene}/render_images'
gt_space = True # mesh是否已经是gt space
pose_type ='original' # 'original' or 'circle'
custom_pose = True # 自定义pose
custom_pose_path = '/home/dawn/poses_e0.npy'
custom_intrinsics_path = '/home/dawn/intrinsics_e0.npy'
if_interpolate = True
num_interpolations = 0
inter_type = 'uniform' # 'uniform' or 'sin'
fps = 24
use_image_wh = False
width= 1920
height= 1080
merge_type = 'half' # 'cat' or 'half'
rendered_output_names = ["rgb","normal"] # 可选择渲染rgb和normal，rgb是纯色的mesh。
################################################################ Hyperparameters ################################################################

# load poses... from data_dir/meta_data.json
poses,scale_mat,intrinsics, image_width, image_height= get_poses(data_dir) # from data_dir/meta_data.json
if use_image_wh:
    width, height = image_width, image_height
# align intrinsics from image_width and image_height to width and height
intrinsics[:,0] = intrinsics[:,0]/image_width*width
intrinsics[:,1] = intrinsics[:,1]/image_height*height
intrinsics = intrinsics[0]
if custom_pose:
    poses = np.load(custom_pose_path)
    if os.path.exists(custom_intrinsics_path):
        intrinsics = np.load(custom_intrinsics_path)

if pose_type=='circle': # FIXME：轴和scene不对齐，需要调整 [√，现可自定义相机x轴和z轴]
    mesh = trimesh.load_mesh(mesh_dir)
    if gt_space:
        normalized_mesh = mesh.apply_transform(np.linalg.inv(scale_mat))
        verts, colors = normalized_mesh.vertices, normalized_mesh.visual.vertex_colors
    else:
        verts, colors = mesh.vertices, mesh.visual.vertex_colors
    # visual_radius(verts,colors,radius=1,center=np.array([0,0,0],dtype=np.float32)) # visualize pcd and bounding box
    poses = generate_circle_pose(center=np.array([0,0,0],dtype=np.float32),elevation=25,radius=3,num_poses=poses.shape[0], camera_x_axis=[-0.2,0,1],camera_z_axis=[0,-1,0])  # TODO：调整ele、radius以及根据visual_radius的调整camera_x_axis和camera_z_axis
if if_interpolate: # 是否插值
    print('Will interpolate {} frames between each pair of poses'.format(num_interpolations))
    poses = interpolate_poses(poses, num_interpolations=num_interpolations, inter_type='uniform') # 实际上是c2n

if gt_space and not custom_pose:
    poses = np.matmul(scale_mat[None].repeat(poses.shape[0],axis=0),poses) # c2n/c2w-> c2gt

mesh = open3d.cuda.pybind.io.read_triangle_mesh(mesh_dir)
mesh.compute_vertex_normals() # 计算法向量
mesh.paint_uniform_color([1.0,1.0,1.0])

# open3d可视化
vis = open3d.visualization.VisualizerWithKeyCallback()
vis.create_window('rendering',width=width, height=height)

vis.add_geometry(mesh)
vis.get_render_option().load_from_json('./render.json')

index = -1
rendered_images=[]

def move_forward(vis):
    # This function is called within the o3d.visualization.Visualizer::run() loop
    # The run loop calls the function, then re-render
    # So the sequence in this function is to:
    # 1. Capture frame
    # 2. index++, check ending criteria
    # 3. Set camera
    # 4. (Re-render)
    ctr = vis.get_view_control()
    if_out_image = False
    global width
    global height
    global index
    global intrinsics
    global poses
    global rendered_images
    if index >= 0:
        images = []
        for render_name in rendered_output_names:
            output_image_dir_cur = output_image_dir +'/'+ render_name

            if render_name == "normal":
                vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Normal
            elif render_name == "rgb":
                vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Color
            if not os.path.exists(output_image_dir_cur):
                os.makedirs(output_image_dir_cur, exist_ok=True)
            vis.capture_screen_image(output_image_dir_cur +'/' f"{index:05d}.png", True)

            images.append(cv2.imread(output_image_dir_cur +'/' f"{index:05d}.png"))
        if merge_type == "cat":
            images = np.concatenate(images, axis=1)
        elif merge_type == "half":
            mask = np.zeros_like(images[0])
            mask[:, : mask.shape[1] // 2, :] = 1
            images = images[0] * mask + images[1] * (1 - mask)
        rendered_images.append(images)
    index = index + 1
    if index < len(poses):

        param = ctr.convert_to_pinhole_camera_parameters()

        # use original intrinsics from data_dir/meta_data.json。一般不用。
        camera = intrinsics
        fx = camera[0, 0]
        fy = camera[1, 1]
        cx = camera[0, 2]
        cy = camera[1, 2]

        param.intrinsic.set_intrinsics(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)

        param.extrinsic = np.linalg.inv(poses[index])

        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
    else:
        vis.register_animation_callback(None)
        vis.destroy_window()

    return False


vis.register_animation_callback(move_forward)
vis.run()

fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 保存视频的编码: MPEG-avi, mp4v-mp4
os.makedirs(output_video_dir, exist_ok=True)
writer_merge = cv2.VideoWriter(os.path.join(output_video_dir,os.path.basename(mesh_dir).split('.')[0]+f'_{merge_type}.mp4'), fourcc, fps, (width*(2 if merge_type == 'cat' else 1), height), isColor=True) # isColor=True: 彩色视频, isColor=False: 灰度视频
writer_rgb = cv2.VideoWriter(os.path.join(output_video_dir,os.path.basename(mesh_dir).split('.')[0]+'_rgb.mp4'), fourcc, fps, (width, height), isColor=True)
writer_normal = cv2.VideoWriter(os.path.join(output_video_dir,os.path.basename(mesh_dir).split('.')[0]+'_normal.mp4'), fourcc, fps, (width, height), isColor=True)
for i in tqdm(range(len(rendered_images)),desc='write to video...',unit='frames',colour='green'):
    writer_merge.write(rendered_images[i])
    rgb_i, normal_i = cv2.imread(output_image_dir + '/rgb/' f"{i:05d}.png"), cv2.imread(output_image_dir + '/normal/' f"{i:05d}.png")
    writer_rgb.write(rgb_i)
    writer_normal.write(normal_i)
writer_merge.release()
writer_rgb.release()
writer_normal.release()
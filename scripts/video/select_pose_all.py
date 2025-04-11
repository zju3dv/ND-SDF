import copy
import shutil

import open3d as o3d
import numpy as np
import threading
import os, json
import glob
from functools import partial

# 全局变量存储可视化窗口和相机参数
vis = None
camera_params = None
save_count = 0
poses = []

# 创建目录来保存图像
save_rtdir = './select'

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

def save_images(vis, mesh, mesh_name, save_count, color_dir='./select/color', normal_dir='./select/normal', textured=False):
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)
    global camera_params
    vis.clear_geometries()
    vis.add_geometry(mesh)

    # 恢复相机参数
    view_control = vis.get_view_control()
    view_control.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)

    # 渲染法线图像
    if not textured: # 有纹理的mesh不需要法线图像
        vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Normal
        normal_image_path = os.path.join(normal_dir, f'{mesh_name}_normal{save_count:04d}.png')
        vis.capture_screen_image(normal_image_path, True)

    # 渲染彩色图像
    vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Color
    color_image_path = os.path.join(color_dir, f'{mesh_name}_color{save_count:04d}.png')
    vis.capture_screen_image(color_image_path, True)

def visualize_mesh(main_mesh_idx, mesh_paths, meshes, width=1080, height=1080, images_dir = None, textured=False):
    global vis, camera_params, save_count, poses
    # 创建一个可视化窗口
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=width, height=height)
    vis.add_geometry(meshes[main_mesh_idx])

    if os.path.exists('./render.json'):  # 加载渲染配置
        vis.get_render_option().load_from_json('./render_lighter.json')

    # 初始化视点
    # vis.get_view_control().set_front([0, 0, -1])
    # vis.get_view_control().set_lookat([0, 0, 0])
    # vis.get_view_control().set_up([0, -1, 0])
    # vis.get_view_control().set_zoom(0.5)

    # 定义获取相机位姿和保存图像的回调函数
    save_rendering = True
    def capture_camera_pose(vis):
        global camera_params, save_count,mode
        view_control = vis.get_view_control()
        camera_params = view_control.convert_to_pinhole_camera_parameters()
        c2w = np.linalg.inv(camera_params.extrinsic)
        print("Camera-to-world pose (c2w):")
        poses.append(c2w)
        print(c2w)
        save_idx = save_count if mode == 'select' else change_view.view_idx
        if mode == 'view':
            print('Saving view {}...'.format(change_view.view_idx))
            # # 保存当前视角的图像, 可省略。
            # if images_dir is not None:
            #     images = glob.glob(images_dir + '/*.png')+glob.glob(images_dir + '/*.jpg')
            #     images.sort()
            #     filename = os.path.basename(images[change_view.view_idx])
            #     shutil.copy(images[change_view.view_idx], f'{save_rtdir}/color/{save_idx}/'+filename)


        # 保存所有meshes的图像
        if save_rendering:
            cur_color_dir = os.path.join(save_rtdir, 'color', f'{save_idx}')
            cur_normal_dir = os.path.join(save_rtdir, 'normal', f'{save_idx}')
            for i, (mesh_path, mesh) in enumerate(zip(mesh_paths, meshes)):
                mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
                if textured and i>=len(meshes)//2:
                    mesh_name += '_textured'
                save_images(vis, mesh, mesh_name, save_idx, cur_color_dir, cur_normal_dir, textured and i>=len(meshes)//2)
            # 保存当前视角的图像, 可省略。
            if images_dir is not None and mode=='view':
                images = glob.glob(images_dir + '/*.png') + glob.glob(images_dir + '/*.jpg')
                images.sort()
                filename = os.path.basename(images[change_view.view_idx])
                shutil.copy(images[change_view.view_idx], f'{cur_color_dir}/' + filename)
        # 切换回主mesh
        change_mesh(vis, ord(str(main_mesh_idx)))

        # 更新保存计数
        save_count += 1
        print(f"Saved color and normal images {save_count}")

    def change_mesh(vis, key):
        if ord('0') <= key <= ord('9'):
            idx = key - ord('0')
            if idx < 0 or idx >= len(meshes):
                print("Invalid index")
                return
            print(f"Switching to mesh {idx}")
            nonlocal main_mesh_idx
            main_mesh_idx = idx
            view_control = vis.get_view_control()
            camera_params = view_control.convert_to_pinhole_camera_parameters()
            vis.clear_geometries()
            vis.add_geometry(meshes[main_mesh_idx])
            view_control.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True) # newest open3d
            # view_control.convert_from_pinhole_camera_parameters(camera_params)


    def change_view(vis, key):
        global view_poses, intrinsics,width,height
        # 静态变量存储当前视角
        if not hasattr(change_view, "view_idx"):
            change_view.view_idx = 0
        if key == 263:
            change_view.view_idx -= 1
        elif key == 262:
            change_view.view_idx += 1
        change_view.view_idx = change_view.view_idx % len(view_poses)
        print(f"Switching to view {change_view.view_idx}")
        ctr = vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        param.extrinsic = np.linalg.inv(view_poses[change_view.view_idx])
        # use original intrinsics from data_dir/meta_data.json。一般不用。
        if intrinsics is not None:
            camera = intrinsics[0]
            fx = camera[0, 0]
            fy = camera[1, 1]
            cx = camera[0, 2]
            cy = camera[1, 2]
            param.intrinsic.set_intrinsics(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)

        ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)


    # 绑定空格键来获取相机位姿和保存图像, 绑定数字键来切换mesh
    vis.register_key_callback(ord(" "), capture_camera_pose)
    # 绑定'a'、'd'来切换预读pose的上一个和下一个视角
    if mode =='view':
        vis.register_key_callback(262, partial(change_view, key=262))
        vis.register_key_callback(263, partial(change_view, key=263))
    for key in range(ord('0'), ord('9') + 1):
        vis.register_key_callback(key, partial(change_mesh, key=key))
    # 渲染
    vis.run()
    vis.destroy_window()

# 一般dataset矫正到单位球/立方体内
# 0e75f3c4d9
# 036bce3393
# 108ec0b806
# e050c15a8d

# 355e5e32db
# 578511c8a9
# 09c1414f1b
# 7f4d173c9c
# ab11145646

# ---------------------------------------------- #
save_rtdir = './teaser_scannetpp_1'
mesh_paths = glob.glob('/data/projects/gaussian-opacity-fields/output/tnt4/test/ours_30000/fusion' + '/*.ply')
mesh_paths.sort()
# mesh_paths = ['/data/projects/implicit_reconstruction/runs_tnt3-5/tnt4_4_2048_latest.ply']

gt_space = False
mode = 'select' # select or view or read_pose
data_dir = '/data/scannetpp/036bce3393'
width = 1920
height = 1080
gt_images_dir = None #'/mnt/xishu/ND-SDF/data/tankandtemples/advanced/Ballroom/images_raw'
has_intrinsics = False
textured = False # 是否加载纹理
# ---------------------------------------------- #
intrinsics = None
if (has_intrinsics and data_dir) or mode == 'view':
    # load poses... from data_dir/meta_data.json
    view_poses, scale_mat, intrinsics, image_width, image_height = get_poses(data_dir)  # from data_dir/meta_data.json
    # align intrinsics from image_width and image_height to width and height
    intrinsics[:, 0] = intrinsics[:, 0] / image_width * width
    intrinsics[:, 1] = intrinsics[:, 1] / image_height * height
    # if gt_space:
    #     view_poses = np.matmul(scale_mat[None,...].repeat(len(view_poses), axis=0), view_poses)


os.makedirs(os.path.join(save_rtdir, 'color'), exist_ok=True)
os.makedirs(os.path.join(save_rtdir, 'normal'), exist_ok=True)
# 加载其他meshes
# trimesh_meshes = [trimesh.load_mesh(path) for path in mesh_paths]
meshes = [o3d.io.read_triangle_mesh(path) for path in mesh_paths if path]
meshes_textured = []
# 计算法线
for i,mesh in enumerate(meshes):
    if gt_space and mode == 'view':
        gt2world = np.linalg.inv(scale_mat)
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) @ gt2world[:3, :3].T + gt2world[:3, 3])
        # mesh.transform(np.linalg.inv(scale_mat)) # FIXME：o3d的transform函数有问题不起作用
    if not gt_space and os.path.basename(mesh_paths[i]) == 'mesh_aligned_0.05.ply': # scannet-pp gt->world
        gt2world = np.linalg.inv(scale_mat)
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) @ gt2world[:3, :3].T + gt2world[:3, 3])
        # mesh.transform(np.linalg.inv(scale_mat))
    if textured:
        meshes_textured.append(copy.deepcopy(mesh))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([1.0, 1.0, 1.0])
if textured:
    meshes.extend(meshes_textured)
    mesh_paths.extend(mesh_paths)
# 创建一个线程来运行可视化
main_mesh_idx = 0
for i, mesh_path in enumerate(mesh_paths):
    if textured and i>=len(meshes)//2:
        mesh_path = mesh_path.split('.')[0] + '_textured.ply'
    print(f"{i}: {mesh_path}")
vis_thread = threading.Thread(target=visualize_mesh, args=(main_mesh_idx, mesh_paths, meshes, width, height, gt_images_dir, textured))
vis_thread.start()

# 等待可视化线程结束
vis_thread.join()

poses = np.array(poses)
np.save(os.path.join(save_rtdir,"selected_poses.npy"), poses)



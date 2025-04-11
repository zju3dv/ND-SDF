import open3d as o3d
import numpy as np
import threading
import cv2
import os

# 全局变量存储可视化窗口和相机参数
vis = None
camera_params = None
save_count = 0
poses = []

# 创建目录来保存图像
os.makedirs('./select/color', exist_ok=True)
os.makedirs('./select/normal', exist_ok=True)


def visualize_mesh(mesh):
    global vis, camera_params, poses
    # 创建一个可视化窗口
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(mesh)
    if os.path.exists('./render.json'): # 加载渲染配置
        vis.get_render_option().load_from_json('./render.json')

    # 初始化视点
    vis.get_view_control().set_front([0, 0, -1])
    vis.get_view_control().set_lookat([0, 0, 0])
    vis.get_view_control().set_up([0, -1, 0])
    vis.get_view_control().set_zoom(0.5)

    # 定义获取相机位姿和保存图像的回调函数
    def capture_camera_pose(vis):
        global camera_params, save_count
        view_control = vis.get_view_control()
        camera_params = view_control.convert_to_pinhole_camera_parameters()
        c2w = np.linalg.inv(camera_params.extrinsic)
        print("Camera-to-world pose (c2w):")
        poses.append(c2w)
        print(c2w)


        # 渲染法线图像
        vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Normal
        # vis.poll_events()
        # vis.update_renderer()

        # 获取法线图像
        normal_image = vis.capture_screen_image(f'./select/normal/{save_count:04d}_normal.png', True)
        normal_image = np.asarray(normal_image)

        # 渲染彩色图像
        vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Color
        # vis.poll_events()
        # vis.update_renderer()

        # 获取彩色图像
        color_image = vis.capture_screen_image(f'./select/color/{save_count:04d}_color.png', True)
        color_image = np.asarray(color_image)

        # 更新保存计数和图像保存状态
        save_count += 1
        print(f"Saved color and normal images {save_count}")

    # 绑定空格键来获取相机位姿和保存图像
    vis.register_key_callback(ord(" "), capture_camera_pose)

    # 渲染
    vis.run()
    vis.destroy_window()


def main():
    global vis, camera_params, poses
    # 加载网格
    mesh = o3d.io.read_triangle_mesh("/data/projects/implicit_reconstruction/scripts/scannetpp_meshes_version1_2048/scannetpp7-1_036bce3393_cues_2048_latest.ply")
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([1.0, 1.0, 1.0])
    # 创建一个线程来运行可视化
    vis_thread = threading.Thread(target=visualize_mesh, args=(mesh,))
    vis_thread.start()

    # 等待可视化线程结束
    vis_thread.join()

    poses = np.array(poses)
    np.save("selected_poses.npy", poses)


if __name__ == "__main__":
    main()

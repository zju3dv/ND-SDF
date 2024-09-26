import numpy as np
import trimesh
import cv2
from scipy.linalg import expm
from tqdm import tqdm
def rot2quat(R):
    batch_size, _,_ = R.shape
    q = np.ones((batch_size, 4))

    R00 = R[:, 0,0]
    R01 = R[:, 0, 1]
    R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]
    R11 = R[:, 1, 1]
    R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]
    R21 = R[:, 2, 1]
    R22 = R[:, 2, 2]

    q[:,0]=np.sqrt(1.0+R00+R11+R22)/2
    q[:, 1]=(R21-R12)/(4*q[:,0])
    q[:, 2] = (R02 - R20) / (4 * q[:, 0])
    q[:, 3] = (R10 - R01) / (4 * q[:, 0])
    return q
def quat2rot(q):
    batch_size, _ = q.shape
    q = q/np.linalg.norm(q, axis=1, keepdims=True)
    R = np.ones((batch_size, 3,3))
    qr=q[:,0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0]=1-2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj *qi -qk*qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1-2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2*(qj*qk - qi*qr)
    R[:, 2, 0] = 2 * (qk * qi-qj * qr)
    R[:, 2, 1] = 2 * (qj*qk + qi*qr)
    R[:, 2, 2] = 1-2 * (qi**2 + qj**2)
    return R

def frames2video(frames, video_path, fps=30):
    '''
    frames: list of frames, each frame is a numpy array with shape (H,W,3)
    '''
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    for frame in frames:
        video.write(frame)
    video.release()

# 已知属性
def generate_bbox_ply(center, radius):
    # 生成以center为中心,radius为半径的bounding box ply
    # center: 中心坐标
    # radius: 半径
    # 返回：点云
    x = np.linspace(center[0] - radius, center[0] + radius, 2)
    y = np.linspace(center[1] - radius, center[1] + radius, 2)
    z = np.linspace(center[2] - radius, center[2] + radius, 2)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    x, y, z = x.flatten(), y.flatten(), z.flatten()
    p = np.vstack([x, y, z]).T

    p_r = p + np.random.randn(*p.shape) * 0.05
    p_r2 = p + np.random.randn(*p.shape) * 0.05

    verts = np.vstack([p, p_r, p_r2])
    faces = []
    face_colors = []
    for i in range(8):
        for d in range(3):
            j = i ^ (1 << d)
            # i,j,j+8,j+16组成的四面体
            id = [i, j, j+8, j+16]
            faces.extend([[id[0], id[1], id[2]], [id[0], id[2], id[3]], [id[0], id[3], id[1]], [id[1], id[2], id[2]] ])
            face_colors.extend([[0, 0, 0, 255]]*4)
    faces = np.array(faces)
    face_colors = np.array(face_colors)

    # 再以verts[0]为原点添加x轴、y轴、z轴,长度为4*radius，颜色分别为红、绿、蓝
    ofs = 24
    origin = center - radius
    p_axis = np.vstack([origin+np.array([4*radius, 0, 0]), origin+np.array([0, 4*radius, 0]), origin+np.array([0, 0, 4*radius])])
    p_axis_r = p_axis + np.random.randn(*p_axis.shape) * 0.1
    p_axis_r2 = p_axis + np.random.randn(*p_axis.shape) * 0.1
    verts = np.vstack([verts, p_axis, p_axis_r, p_axis_r2]) # 24+3+3+3=22
    axis_faces = []
    axis_face_colors = []
    for ax in range(3):
        # 0, ofs+ax, ofs+ax+3, ofs+ax+6组成的四面体
        id = [0, ofs+ax, ofs+ax+3, ofs+ax+6]
        axis_faces.extend([[id[0], id[1], id[2]], [id[0], id[2], id[3]], [id[0], id[3], id[1]], [id[1], id[2], id[2]] ])
        axis_color = [255,0,0,255] if ax==0 else [0,255,0,255] if ax==1 else [0,0,255,255]
        axis_face_colors.extend([axis_color]*4)
    axis_faces = np.array(axis_faces)
    axis_face_colors = np.array(axis_face_colors)

    faces = np.vstack([faces, axis_faces])
    face_colors = np.vstack([face_colors, axis_face_colors])
    bbox_ply = trimesh.Trimesh(vertices=verts, faces=faces, face_colors=face_colors)
    return bbox_ply


def visual_radius(verts, colors, radius, center):
    # visualize pcd and bounding box
    # 功能：从radius开始交互式调整（增大减小）以center为中心的bounding box，直到完全包含住感兴趣的点云区域，返回调整后的center和radius
    # 三维可视化：①点云 ②bounding box ③center
    # 调整：①按键盘上下左右键调整center ②按键盘+/-键调整radius
    # 返回：center, radius
    pcd = trimesh.PointCloud(vertices=verts, colors=colors)
    while True:
        # 1. 可视化pcd和bounding box
        bbox_ply = generate_bbox_ply(center, radius)
        trimesh.Scene([pcd, bbox_ply]).show()

        # key指令：center shift: '<x/y/z><+/-><float>'  radius: '<s><float>' 代表scale
        key = input('指令：①移动center: <r/g/b><+/-><float>，r、g、b分别对应可视化轴（x、y、z），float是cube边长（2*r）的倍数；②radius: <s><float> ，float代表scale倍率；③exit: <q>。请输入：')
        # 解析
        if key == 'q':
            break
        elif key[0] in ['r', 'g', 'b']: # r-x, g-y, b-z
            axis = key[0]
            sign = 1 if key[1] == '+' or '0'<=key[1]<='9' else -1
            shift = float(key[1:]) if '0'<=key[1]<='9' else float(key[2:])
            if axis == 'r':
                center[0] += sign * shift*radius*2
            elif axis == 'g':
                center[1] += sign * shift*radius*2
            elif axis == 'b':
                center[2] += sign * shift*radius*2
        elif key[0] == 's':
            radius *= float(key[1:])
        else:
            print('invalid key')

    return center, radius

def generate_circle_pose(center, elevation, radius, num_poses, camera_x_axis=[0,1,0], camera_z_axis=[-1,0,0]):
    # center: (3,), 物体中心
    # elevation: float, 仰角∈[-90,90]
    # radius: float, 半径
    # num_poses: int, 生成的位姿数量
    # camera_x_axis: (3,), 仰角等于0时的相机坐标系的x轴
    # camera_z_axis: (3,), 仰角等于0时的相机坐标系的z轴
    elevation = elevation / 180 * np.pi
    camera_x_axis = np.array(camera_x_axis,dtype=np.float32)[:,None] # c 0°x轴 (3,1)
    camera_z_axis = np.array(camera_z_axis,dtype=np.float32)[:,None] # c 0°z轴 (3,1)
    # 对z进行施密特正交化
    camera_z_axis = camera_z_axis - camera_z_axis.T@camera_x_axis*camera_x_axis
    camera_x_axis,camera_z_axis = camera_x_axis/np.linalg.norm(camera_x_axis),camera_z_axis/np.linalg.norm(camera_z_axis)
    camera_y_axis = np.cross(camera_z_axis, camera_x_axis, axis=0)   # c 0°y轴 (3,1)
    # 仰角elevation的相机z轴等于0°时绕camera_x_axis旋转-elevation
    camera_z_axis = cv2.Rodrigues(-elevation*camera_x_axis[:,0])[0]@camera_z_axis # c elevation°z轴 (3,1)

    start_pos = -camera_z_axis*(radius/np.cos(elevation))
    delta_angle = 2*np.pi/num_poses
    poses = []
    for i in range(num_poses):
        # rotate around z axis, just rotate x and z
        angle = i*delta_angle
        R = cv2.Rodrigues(angle*(-camera_y_axis)[:,0])[0]
        cur_x_axis = R@camera_x_axis
        cur_z_axis = R@camera_z_axis
        cur_y_axis = np.cross(cur_z_axis,cur_x_axis,axis=0)
        cur_pos = R@start_pos+center[:,None]
        cur_pose = np.eye(4,dtype=np.float32)
        cur_pose[:3,:3] = np.concatenate([cur_x_axis,cur_y_axis,cur_z_axis],axis=1)
        cur_pose[:3,3] = cur_pos[:,0]
        poses.append(cur_pose)
    return np.array(poses)

def normalize_rotation(R):
    # 使用奇异值分解（SVD）对矩阵进行标准化, inter_R=R1@expm(t*(R1.T@R2))非正交
    U, S, V_T = np.linalg.svd(R)
    normalized_R = U @ V_T

    return normalized_R

def pose_interpolation(p1, p2, t):
    # p1: (n, 4, 4), 批量左侧位姿
    # p2: (n, 4, 4), 批量右侧位姿
    # t: (m,), 线性插值参数

    # 改shape
    n,m=p1.shape[0],t.shape[0]
    p1=p1[:,None].repeat(m,axis=1).reshape(n*m,4,4)
    p2=p2[:,None].repeat(m,axis=1).reshape(n*m,4,4)
    t=t[None].repeat(n,axis=0).reshape(n*m)
    # 提取姿势中的平移向量和旋转矩阵
    T1, R1 = p1[:,:3, 3], p1[:,:3, :3]
    T2, R2 = p2[:,:3, 3], p2[:,:3, :3]

    # 线性插值平移向量和旋转矩阵
    inter_T = T1 * (1 - t[:,None]) + T2 * t[:,None]
    inter_R = np.array([normalize_rotation(R1[i]@expm(t[i]*(R1[i].T@R2[i]))) for i in tqdm(range(n*m),desc='Interpolating poses...')]) # 旋转矩阵的线性插值：R1@(R1.T@R2)^t

    # 构建插值后的姿势矩阵
    inter_poses = np.eye(4, dtype=np.float32)[None].repeat(n*m,axis=0)
    inter_poses[:,:3, :3] = inter_R
    inter_poses[:,:3, 3] = inter_T

    return inter_poses

def interpolate_poses(poses, num_interpolations=5, inter_type='uniform'):
    # Interpolate between the poses to get more frames
    # poses: (N, 4, 4), camera-to-world poses
    # num_interpolations: int, number of frames to interpolate between each pair of poses
    poses_l=poses[:-1]
    poses_r=poses[1:]
    # 1. 均匀间隔
    if inter_type=='uniform':
        ts = np.linspace(0, 1, num_interpolations + 2)[:-1]
    elif inter_type=='sin':
        ts = np.linspace(0, 1, num_interpolations + 2)[:-1]
        ts = np.sin(ts * np.pi / 2)
    else:
        raise NotImplementedError
    inter_poses=pose_interpolation(poses_l, poses_r, ts)
    inter_poses = np.concatenate([inter_poses,poses[-1:,...]],axis=0) # 最后一个位姿
    return inter_poses
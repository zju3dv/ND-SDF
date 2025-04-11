import os
import cv2

from PIL import Image

def clip_video(video_path, clipped_path, frame_rate, resolution, clip_seconds):
    # 打开视频文件
    video = cv2.VideoCapture(video_path)

    # 获取视频的总帧数和帧率
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = video.get(cv2.CAP_PROP_FPS)

    # 计算每个帧之间的间隔
    if frame_rate > original_fps:
        frame_interval = 1
    else:
        frame_interval = int(original_fps / frame_rate)
    print('frame_interval:',frame_interval)
    clip_frames = []

    clip_frames_idx=[int(clip_seconds[0]*original_fps),int(clip_seconds[1]*original_fps)]
    # 逐帧读取视频
    for frame_index in range(0, total_frames, frame_interval):
        if frame_index<clip_frames_idx[0] or frame_index>clip_frames_idx[1]:
            continue
        # 设置视频帧位置
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # 读取当前帧
        ret, frame = video.read()

        if ret:
            # 调整分辨率
            resized_frame = cv2.resize(frame, resolution)
            clip_frames.append(resized_frame)
    # 关闭视频文件
    video.release()
    # write
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # 保存视频的编码: XVID, MPEG
    writer = cv2.VideoWriter(clipped_path, fourcc, original_fps/frame_interval, resolution, isColor=True) # isColor=True: 彩色视频, isColor=False: 灰度视频
    for frame in clip_frames:
        writer.write(frame)
    writer.release()

# 设置视频路径、帧率和分辨率
video_path = '/scripts/video/cxd/rgb+normal'  # 目录或者文件
out_dir = 'clipped_videos'
frame_rate = 30  # GIF的帧率为每秒10帧
resolution = (640, 480)  # GIF的分辨率为640x480
clip_seconds = [24, 54]  # 截取视频的时间段
####################

os.makedirs(out_dir, exist_ok=True)

# video_path是目录的话
if os.path.isdir(video_path):
    files = os.listdir(video_path)
    for file in files:
        pth = os.path.join(video_path, file)
        clipped_path = os.path.join(out_dir, file.split('.')[0] + f'_{clip_seconds[0]}to{clip_seconds[1]}.avi')
        clip_video(pth, clipped_path, frame_rate, resolution, clip_seconds)
else:
    clipped_path = os.path.join(out_dir, os.path.basename(video_path).split('.')[0] + f'_{clip_seconds[0]}to{clip_seconds[1]}.avi')
    clip_video(video_path, clipped_path, frame_rate, resolution, clip_seconds)
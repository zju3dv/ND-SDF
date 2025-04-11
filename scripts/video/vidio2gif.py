import os
import cv2
from moviepy.editor import VideoFileClip

from PIL import Image

def convert_video_to_gif(video_path, gif_path, frame_rate, downscale, clip_seconds):
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

    # 创建一个空白的GIF图像
    gif_frames = []

    clip_frames=[int(clip_seconds[0]*original_fps),int(clip_seconds[1]*original_fps)]
    # 逐帧读取视频并将其转换为GIF
    for frame_index in range(0, total_frames, frame_interval):
        if frame_index<clip_frames[0] or frame_index>clip_frames[1]:
            continue
        # 设置视频帧位置
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # 读取当前帧
        ret, frame = video.read()

        if ret:
            # 转换为RGB格式
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 调整分辨率
            resolution_downscale = (int(rgb_frame.shape[1] / downscale), int(rgb_frame.shape[0] / downscale))
            resized_frame = cv2.resize(rgb_frame, resolution_downscale, interpolation=cv2.INTER_AREA)

            # 将帧转换为PIL图像
            pil_image = Image.fromarray(resized_frame)

            # 添加到GIF帧列表
            gif_frames.append(pil_image)

    # 保存GIF图像
    gif_frames[0].save(gif_path,
                       save_all=True,
                       append_images=gif_frames[1:],
                       optimize=True,
                       duration=int(1000 / frame_rate),
                       loop=0,
                       quality=1)
    print("GIF转换完成！")

    # 关闭视频文件
    video.release()

# 设置视频路径、GIF路径、帧率和分辨率
video_path = '/data/projects/implicit_reconstruction/scripts/video/0616_bishe/rgb+normal'
out_dir = 'gifs'
frame_rate = 15  # GIF的帧率为每秒10帧
downscale = 2.0
clip_seconds = [0, 99999999]  # 截取视频的时间段

# 调用函数进行视频转换为GIF
os.makedirs(out_dir, exist_ok=True)

# video_path是目录的话
if os.path.isdir(video_path):
    files = os.listdir(video_path)
    for file in files:
        pth = os.path.join(video_path, file)
        gif_path = os.path.join(out_dir, file.split('.')[0] + f'.gif')
        convert_video_to_gif(pth, gif_path, frame_rate, downscale, clip_seconds)
else:
    gif_path = os.path.join(out_dir, os.path.basename(video_path).split('.')[0] + '.gif')
    convert_video_to_gif(video_path, gif_path, frame_rate, downscale, clip_seconds)
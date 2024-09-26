import numpy as np
import torch
import os
import argparse
import PIL
from PIL import Image
import glob, cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

from torchvision import transforms
from data.transforms import get_transform
from modules.midas.dpt_depth import DPTDepthModel


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--input_dir', type=str, default='/data/projects/monosdf/data/scannet/scan4/rgb', help="输入图像分辨率宽高比等于1为正方形时...")
    args.add_argument('--task', choices=['normal', 'depth'])
    args.add_argument('--output_dir', type=str, default='/data/projects/monosdf/data/scannet/scan4')

    return args.parse_args()

def get_img_transform(task):

    if task == 'depth':
        trans_totensor = transforms.Compose(
            [
                transforms.Resize(384, interpolation=PIL.Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5),
            ]
        )
    elif task == 'normal':
        trans_totensor = transforms.Compose(
            [
                transforms.Resize(384, interpolation=PIL.Image.BILINEAR),
                get_transform('rgb', image_size=None),
            ]
        )

    return trans_totensor


def get_depth_output(img, model, trans_totensor, device):
    with torch.no_grad():
        input_tensor = trans_totensor(Image.fromarray(img))[None].to(device)
        output = model(input_tensor).clamp(min=0, max=1)
        output = torch.nn.functional.interpolate(output.unsqueeze(0), size=img.shape[0], mode='bilinear')
        depth = output.squeeze().cpu().numpy()
        return depth

def save_depth_output(img, depth, output_dir, img_name):
    # img: (h,w,3), np.array, rgb
    # depth: (h,w), np.array, depth∈[0,1]
    norm_depth = ((depth - depth.min()) / (depth.max() - depth.min())*255).astype(np.uint8)
    vis_depth = cv2.applyColorMap(norm_depth, cv2.COLORMAP_INFERNO)
    cat = np.concatenate([img, cv2.cvtColor(vis_depth, cv2.COLOR_BGR2RGB)], axis = 1)
    Image.fromarray(cat).save(os.path.join(output_dir, 'vis','{}.png'.format(img_name)))
    np.save(os.path.join(output_dir, 'res','{}.npy'.format(img_name)), depth)

def get_normal_output(img, model, trans_totensor, device):
    with torch.no_grad():
        input_tensor = trans_totensor(Image.fromarray(img))[None].to(device)
        output = model(input_tensor).clamp(min=0, max=1)
        output = torch.nn.functional.interpolate(output, size=img.shape[0], mode='bilinear')
        output = output * 2 - 1
        output = output / (output.norm(dim = 1, keepdim = True) + 1e-15)
        normal = output.float().detach().squeeze(0).cpu().numpy()
    return normal.transpose(1, 2, 0) # (h, w, 3)

def save_normal_output(img, normal, output_dir, img_name):
    normal = (normal + 1) / 2
    output_path = os.path.join(output_dir, 'vis','{}.png'.format(img_name))
    # print(output_path)
    normal_img = (normal * 255)
    normal_img = normal_img.astype(np.uint8)
    normal_img = np.concatenate([img, normal_img], axis = 1)
    Image.fromarray(normal_img).save(output_path)
    np.save(os.path.join(output_dir, 'res','{}.npy'.format(img_name)), normal)

def main(args):
    map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_size = 384
    pretrained_weights_path = os.path.join('./pretrained_models', f"omnidata_dpt_{args.task}_v2.ckpt")  # 'omnidata_dpt_depth_v1.ckpt'
    model = DPTDepthModel(backbone="vitb_rn50_384", num_channels=3 if args.task == 'normal' else 1)  # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    if "state_dict" in checkpoint:
        state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)

    trans_totensor = get_img_transform(args.task)

    img_paths = glob.glob(os.path.join(args.input_dir, '*.jpg'))+glob.glob(os.path.join(args.input_dir, '*.png'))
    img_paths.sort()

    output_dir = os.path.join(args.output_dir, 'mono_'+args.task)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'res'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'vis'), exist_ok=True)

    for idx, ipath in enumerate(tqdm(img_paths)):
        img = np.asarray(Image.open(ipath))
        img_name = os.path.basename(ipath).split('.')[0]
        if args.task == 'depth':
            depth = get_depth_output(img, model, trans_totensor, device)
            save_depth_output(img, depth, output_dir, img_name)
        elif args.task == 'normal':
            normal = get_normal_output(img, model, trans_totensor, device)
            save_normal_output(img, normal, output_dir, img_name)
        # quit()

if __name__ == "__main__":
    args = get_args()
    main(args)
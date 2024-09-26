import os.path
import sys

import cv2
import numpy as np
import torch
import argparse

from tqdm import tqdm

from models.system import ImplicitReconSystem
from utils.mesh import my_extract_mesh, texture_function
from omegaconf import OmegaConf
from functools import partial
from dataset.base_dataset import BaseDataset
import utils.utils as utils

# 添加cuda路径
env_list = os.environ['PATH'].split(':')
env_list.append('/usr/local/cuda/bin')
os.environ['PATH'] = ':'.join(env_list)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='/data/projects/implicit_reconstruction/runspp_end1/scannetpp7-1-5_036bce3393_cues/2024-05-14_12-51-27/conf.yaml')
    parser.add_argument('--checkpoint', type=str, default='/data/projects/implicit_reconstruction/runspp_end1/scannetpp7-1-5_036bce3393_cues/2024-05-14_12-51-27/checkpoints/latest.pth')
    parser.add_argument('--output_dir', type=str, default='./rendering_results')
    parser.add_argument('--downscale', type=int, default=1)
    parser.add_argument('--static', action='store_true', help='use static rendering')
    parser.set_defaults(static=True) # 即app
    args=parser.parse_args()

    # load model
    conf = OmegaConf.load(args.conf)
    bound = 1.0 if not hasattr(conf.model, 'bound') else conf.model.bound
    model = ImplicitReconSystem(conf, bound, device='cuda:0').cuda()
    ckpt=torch.load(args.checkpoint)
    cur_step = ckpt['step'] if ckpt.get('step') else 1e9
    # restore model state
    model.load_state_dict(ckpt['model'])
    if conf.model.object.sdf.enable_progressive:
        model.sdf.set_active_levels(cur_step)
        model.sdf.set_normal_epsilon()
    if conf.model.background.enabled and conf.model.background.type=='grid_nerf':
        model.bg_nerf.set_active_levels(cur_step)
    model.eval()
    if args.static:
        model.rgb_enable_app = False

    conf.dataset.scan_id = conf.dataset.scan_id.split('_')[0]
    conf.dataset.data_dir = '/data/scannetpp' # scannetpp

    # rendering
    valid_dataset = BaseDataset(conf.dataset, split='valid',num_rays=conf.train.num_rays, downscale=args.downscale, fewshot=getattr(conf.dataset, 'fewshots', False), fewshot_idx=getattr(conf.dataset, 'fewshot_idx', []))
    valid_dataset.set_loop_all()
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    h,w=valid_dataset.h,valid_dataset.w
    for i,sample in enumerate(valid_loader):
        if i<72:
            continue
        if i>72:
            break
        sample = {k: v.cuda() for k, v in sample.items()} # to gpu
        split_sample = utils.split_input(sample, valid_dataset.total_pixels, n_pixels=1024)
        outputs = []
        for s in tqdm(split_sample, total=len(split_sample), desc=f'rendering batch {i}...', file=sys.stdout, position=0, leave=True):
            output = model(s)
            d = {'rgb': output['rgb'].detach(), 'depth': output['depth'].detach(), 'normal': output['normal'].detach()}
            if conf.model.nbfield.enabled and conf.dataset.use_mono_normal:
                d['quat'] = output['quat'].detach()
                d['biased_normal'] = output['biased_normal'].detach()
                d['biased_mono_normal'] = output['biased_mono_normal'].detach()
            outputs.append(d)
        outputs = utils.merge_output(outputs) # outputs: {'rgb': (batch_size, h*w, 3), 'depth': (batch_size, h*w, 1), 'normal': (batch_size, h*w, 3)}
        plot_outputs = utils.get_plot_data(outputs, sample, valid_dataset.h, valid_dataset.w,monocular_depth=valid_dataset.has_mono_depth, with_single=True)
        for plot_output in plot_outputs:
            idx=plot_output['idx']
            filename=valid_dataset.rgb_paths[idx].split('/')[-1]
            for k,v in plot_output.items():
                if k != 'idx':
                    os.makedirs(os.path.join(args.output_dir, k), exist_ok=True)
                    v.save(os.path.join(args.output_dir, k, filename))
            print(f'{filename} rendered and saved to {args.output_dir}', file=sys.stdout)

        # # 可视化normal_mask rgb
        # rgb = sample['rgb'][0].cpu().numpy().reshape(h, w, 3)
        # mask = sample['mask'][0].cpu().numpy().reshape(h, w)
        # uncertainty = sample['uncertainty'][0].cpu().numpy().reshape(h, w)
        # uncertainty_grad = sample['uncertainty_grad'][0].cpu().numpy().reshape(h, w)
        # normal_mask = ((mask == 1) | ((mask == 0) & (uncertainty < 0.3) & (uncertainty_grad < 0.03)))
        # rgb[~normal_mask] = 0
        # rgb = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        # os.makedirs(os.path.join(args.output_dir, 'normal_mask'), exist_ok=True)
        # cv2.imwrite(os.path.join(args.output_dir,'normal_mask', f'{i}.png'), rgb)



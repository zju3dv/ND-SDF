import os.path
import sys

import trimesh

sys.path.append('/data/projects/implicit_reconstruction')

import numpy as np
import torch
import argparse
from models.system import ImplicitReconSystem
from utils.mesh import my_extract_mesh, texture_function
from omegaconf import OmegaConf
from functools import partial
from dataset.base_dataset import BaseDataset

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='/data/projects/implicit_reconstruction/runs_tnt_sum/tnt15_4/2024-05-20_02-38-25/conf.yaml')
    parser.add_argument('--checkpoint', type=str, default='/data/projects/implicit_reconstruction/runs_tnt_sum/tnt15_4/2024-05-20_02-38-25/checkpoints/latest.pth')
    parser.add_argument('--to_world',action='store_true',help='transform to ground truth world coordinates')
    parser.add_argument('--res', type=int, default=512)
    parser.add_argument('--block_res', type=int ,default=512)
    parser.add_argument('--textured',action='store_true', help='extract textured mesh')
    parser.add_argument('--output_dir', type=str, default='.')
    parser.set_defaults(to_world=False, textured=False)
    args=parser.parse_args()

    # load model
    conf = OmegaConf.load(args.conf)
    bound = 1.0 if not hasattr(conf.model, 'bound') else conf.model.bound
    model = ImplicitReconSystem(conf, bound, device='cuda:0').cuda()
    ckpt=torch.load(args.checkpoint)
    cur_step = ckpt['step'] if ckpt.get('step') else 1e9
    if conf.model.object.sdf.enable_progressive:
        model.sdf.set_active_levels(cur_step)
        model.sdf.set_normal_epsilon()
    model.load_state_dict(ckpt['model'])
    model.eval()

    # extract mesh
    bounds = np.array([[-bound, bound], [-bound, bound], [-bound, bound]])
    texture_func = None
    if args.textured:
        texture_func = partial(texture_function, neural_sdf=model.sdf, neural_rgb=model.rgb)

    mesh = my_extract_mesh(model.sdf.get_sdf,bounds,args.res,args.block_res,texture_func)
    # mesh = trimesh.load_mesh('/data/projects/implicit_reconstruction/scripts/tnt3_4_2048_latest.ply')
    if args.to_world:
        conf.dataset.data_dir = '/data/monosdf/tnt_advanced' # tnt
        #
        # conf.dataset.data_dir = '/data/scannetpp' # scannetpp
        # conf.dataset.scan_id = conf.dataset.scan_id.split('_')[0]

        # conf.dataset.data_dir = '/data/monosdf/scannet' # scannet
        dataset = BaseDataset(conf.dataset, split='valid',num_rays=conf.train.num_rays)
        mesh.apply_transform(dataset.scale_mat)
    filename = conf.train.exp_name if conf.dataset.scan_id == -1 else conf.train.exp_name + f'_{conf.dataset.scan_id}'+f'_{args.res}'+ f'_{os.path.basename(args.checkpoint).split(".")[0]}'+'.ply'
    mesh.export(os.path.join(args.output_dir,filename), 'ply')

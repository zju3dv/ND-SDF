import os.path
import sys

import trimesh

sys.path.append('/data/projects/implicit_reconstruction')

import numpy as np
import torch
import argparse
import os, glob, omegaconf

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--exps_dir', type=str, default='/home/dawn/projects/ND-SDF/runs_tnt_new')
    parser.add_argument('--to_world',action='store_true',help='transform to ground truth world coordinates')
    parser.add_argument('--res', type=int, default=512)
    parser.add_argument('--block_res', type=int ,default=512)
    parser.add_argument('--textured',action='store_true', help='extract textured mesh')
    parser.add_argument('--output_dir', type=str, default='.')
    parser.set_defaults(to_world=True, textured=False)
    args=parser.parse_args()

    exps_dir = args.exps_dir
    exps = os.listdir(exps_dir)
    exps.sort()

    for exp in exps:
        exp_name = exp.split('_')[0]
        # if exp_name!='tnt9':
        #     continue
        if not os.path.isdir(os.path.join(exps_dir, exp)):
            continue
        exp_dir = os.path.join(exps_dir, exp)
        timestamps_dir = glob.glob(os.path.join(exp_dir, '*'))
        timestamps_dir = [d for d in timestamps_dir if os.path.isdir(d)]  # 只保留文件夹
        # 按照时间戳排序
        timestamps_dir.sort()
        conf_path = os.path.join(timestamps_dir[-1], 'conf.yaml')
        latest_ckpt = timestamps_dir[-1]+"/checkpoints/latest.pth"

        cmd = f'python extract_mesh.py --conf {conf_path} --checkpoint {latest_ckpt} --res {args.res} --block_res {args.block_res} --output_dir {args.output_dir}'
        if args.to_world:
            cmd += ' --to_world'
        if args.textured:
            cmd += ' --textured'
        os.system(cmd)


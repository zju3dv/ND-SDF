# evaluate all the experiments in a directory
# mesh
import os
import sys
import omegaconf
import argparse
import glob

#
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/scannet/scans')
    parser.add_argument('--exps_dir', type=str, default='/home/dawn/runs')
    args = parser.parse_args()

    exps_dir = args.exps_dir
    exps = os.listdir(exps_dir)
    exps.sort()

    for exp in exps:
        if not os.path.isdir(os.path.join(exps_dir, exp)):
            continue
        exp_dir = os.path.join(exps_dir, exp)
        timestamps_dir = glob.glob(os.path.join(exp_dir, '*'))
        timestamps_dir = [d for d in timestamps_dir if os.path.isdir(d)] # 只保留文件夹
        # 按照时间戳排序
        timestamps_dir.sort()
        conf_path = os.path.join(timestamps_dir[-1], 'conf.yaml')
        conf = omegaconf.OmegaConf.load(conf_path)
        id = conf.dataset.scan_id
        id = id[:-5]
        plot_dir = os.path.join(timestamps_dir[-1], 'plots')
        meshs = glob.glob(os.path.join(plot_dir, 'mesh_*.ply'))
        if len(meshs) == 0:
            continue
        meshs.sort()
        latest_mesh_path = meshs[-1]

        print(f'Exp: {exp}')
        cmd = f'python eval_scannetpp.py --data_dir {args.data_dir} --id {id} --mesh {latest_mesh_path}'
        os.system(cmd)
        print('------------------------------------------------------------------------------------------------------------')
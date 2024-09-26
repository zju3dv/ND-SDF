import argparse
import os
import glob

import trimesh
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default='../../runs')
parser.add_argument('--data_dir', type=str, default='./../data/scannetpp')
parser.add_argument('--exp_name', type=str, default='scannetpp')
parser.add_argument('--out_dir', type=str, default='evaluation')
args = parser.parse_args()

scans = ["0e75f3c4d9","036bce3393","e050c15a8d"]
root_dir = args.root_dir
data_dir = args.data_dir
exp_name = args.exp_name
out_dir = os.path.join(args.out_dir, exp_name)
Path(out_dir).mkdir(parents=True, exist_ok=True)

evaluation_txt_file = f"{args.out_dir}/{exp_name}.csv"
evaluation_txt_file = open(evaluation_txt_file, 'w')

for idx, scan in enumerate(scans):
    idx = idx + 1
    cur_exp = f"{exp_name}_{idx}"
    cur_root = os.path.join(root_dir, cur_exp)
    if not os.path.exists(cur_root):
        continue
    # use first timestamps
    dirs = sorted(os.listdir(cur_root))
    cur_root = os.path.join(cur_root, dirs[0])
    files = list(filter(os.path.isfile, glob.glob(os.path.join(cur_root, "plots/*.ply"))))

    files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
    ply_file = files[-1]
    print(ply_file)

    # curmesh

    cmd = f"python evaluate_single_mesh.py --data_dir {data_dir} --mesh {ply_file} --eval_dir {os.path.join(args.out_dir, cur_exp)} --vis_err"
    print(cmd)
    # accuracy_rec, completion_rec, precision_ratio_rec, completion_ratio_rec, fscore, normal_acc, normal_comp, normal_avg
    output = subprocess.check_output(cmd, shell=True).decode("utf-8")
    output = output.replace(" ", ",")
    print(output)

    evaluation_txt_file.write(f"{scan},{Path(ply_file).name},{output}")
    evaluation_txt_file.flush()

import argparse
import os
import glob

import trimesh
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default='../../runs')
parser.add_argument('--data_dir', type=str, default='../../data/replica')
parser.add_argument('--exp_name', type=str, default='replica')
parser.add_argument('--out_dir', type=str, default='evaluation')
args = parser.parse_args()

scans = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]
exp_scans = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]
data_dir = args.data_dir
root_dir = args.root_dir
exp_name = args.exp_name
out_dir = os.path.join(args.out_dir, exp_name)
Path(out_dir).mkdir(parents=True, exist_ok=True)

evaluation_txt_file = f"{args.out_dir}/{exp_name}.csv"
evaluation_txt_file = open(evaluation_txt_file, 'w')

for idx, scan in enumerate(scans):
    idx = idx + 1
    # test set
    if not (scan in exp_scans):
       continue

    cur_exp = f"{exp_name}_{idx}"
    cur_root = os.path.join(root_dir, cur_exp)
    # use first timestamps
    dirs = sorted(os.listdir(cur_root))
    cur_root = os.path.join(cur_root, dirs[0])
    files = list(filter(os.path.isfile, glob.glob(os.path.join(cur_root, "plots/*.ply"))))

    files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
    ply_file = files[-1]
    print(ply_file)

    # curmesh
    cull_mesh_out = os.path.join(out_dir, f"{scan}.ply")
    cmd = f"python cull_mesh.py --input_mesh {ply_file} --input_scalemat {data_dir}/scan{idx}/cameras.npz --traj {data_dir}/scan{idx}/traj.txt --output_mesh {cull_mesh_out}"
    print(cmd)
    os.system(cmd)

    cmd = f"python eval_recon.py --rec_mesh {cull_mesh_out} --gt_mesh {data_dir}/cull_GTmesh/{scan}.ply"
    print(cmd)
    # accuracy_rec, completion_rec, precision_ratio_rec, completion_ratio_rec, fscore, normal_acc, normal_comp, normal_avg
    output = subprocess.check_output(cmd, shell=True).decode("utf-8")
    output = output.replace(" ", ",")
    print(output)

    evaluation_txt_file.write(f"{scan},{Path(ply_file).name},{output}")
    evaluation_txt_file.flush()

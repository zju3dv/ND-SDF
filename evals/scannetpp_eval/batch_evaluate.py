# evaluate all the experiments in a directory
# mesh
import os
import sys
import omegaconf
import argparse
import glob

# 036bce3393
# 0e75f3c4d9
# 108ec0b806
# 21d970d8de
# 47b37eb6f9
# e050c15a8d

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/scannetpp')
    parser.add_argument('--meshes_dir', type=str, default='/home/dawn/projects/ND-SDF/scripts')
    args = parser.parse_args()

    meshes_dir = args.meshes_dir
    meshes = os.listdir(meshes_dir)
    meshes = [mesh for mesh in meshes if mesh.endswith('.ply')]
    meshes.sort(key=lambda x: x.split('_')[-4]) # 按id
    meshes.sort(key=lambda x: x.split('mesh')[0]) # 按exp
    for mesh_name in meshes:

        print(f'Mesh: {mesh_name}')
        scan_id = mesh_name.split('_')[-4]
        exp_name = mesh_name.split('mesh')[0][:-1]

        # if scan_id != "108ec0b806":
        #     continue
        # if exp_name.split('_')[0] != "scannetpp7-1":
        #     continue

        cmd = f'python evaluate_single_mesh.py --data_dir {args.data_dir} --id {scan_id} --mesh {os.path.join(meshes_dir,mesh_name)}'
        os.system(cmd)
        print('------------------------------------------------------------------------------------------------------------')
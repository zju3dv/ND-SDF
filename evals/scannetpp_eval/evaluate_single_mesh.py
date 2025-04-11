import os
# import open3d as o3d
from pathlib import Path
import json
import trimesh
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import argparse
import matplotlib.pyplot as plt



# data_list = ['Caterpillar', 'Courthouse']

def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)

def nn_correspondance(verts1, verts2):
    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    kdtree = KDTree(verts1)
    distances, indices = kdtree.query(verts2)
    distances = distances.reshape(-1)

    return distances

def write_color_distances(path, pcd, distances, max_distance):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    # cmap = plt.get_cmap("afmhot")
    cmap = plt.get_cmap("hot_r")
    distances = np.array(distances)
    colors = cmap(np.minimum(distances, max_distance) / max_distance)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd)


def evaluate(mesh_pred, mesh_trgt, threshold=.05, down_sample=.02, scale=1, offset=[0., 0, 0], vis_err=False, eval_dir=None):
    pcd_trgt = o3d.geometry.PointCloud()
    pcd_pred = o3d.geometry.PointCloud()

    vertices = mesh_trgt.vertices[:, :3]
    vertices -= offset
    vertices *= scale
    mesh_trgt.vertices[:, :3] = vertices

    pcd_trgt.points = o3d.utility.Vector3dVector(mesh_trgt.vertices[:, :3])
    pcd_pred.points = o3d.utility.Vector3dVector(mesh_pred.vertices[:, :3])

    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)

    colors = np.random.rand(np.array(pcd_trgt.points).shape[0], 3)
    pcd_trgt.colors = o3d.utility.Vector3dVector(colors)
    colors = np.random.rand(np.array(pcd_pred.points).shape[0], 3)
    pcd_pred.colors = o3d.utility.Vector3dVector(colors)

    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    dist1 = nn_correspondance(verts_pred, verts_trgt)
    dist2 = nn_correspondance(verts_trgt, verts_pred)

    #########
    if vis_err:
        os.makedirs(eval_dir, exist_ok=True)
        source_n_fn =f"{eval_dir}/precision.ply"
        target_n_fn = f"{eval_dir}/recall.ply"

        # print("[ViewDistances] Add color coding to visualize error")
        # eval_str_viewDT = (
        #     OPEN3D_EXPERIMENTAL_BIN_PATH
        #     + "ViewDistances "
        #     + source_n_fn
        #     + " --max_distance "
        #     + str(threshold * 3)
        #     + " --write_color_back --without_gui"
        # )
        # os.system(eval_str_viewDT)
        write_color_distances(source_n_fn, pcd_pred, dist2, 3 * threshold)

        # print("[ViewDistances] Add color coding to visualize error")
        # eval_str_viewDT = (
        #     OPEN3D_EXPERIMENTAL_BIN_PATH
        #     + "ViewDistances "
        #     + target_n_fn
        #     + " --max_distance "
        #     + str(threshold * 3)
        #     + " --write_color_back --without_gui"
        # )
        # os.system(eval_str_viewDT)
        write_color_distances(target_n_fn, pcd_trgt, dist1, 3 * threshold)
        ###########

    # print(dist1.max(), dist1.min())
    # print(dist2.max(), dist2.min())
    precision = np.mean((dist2 < threshold).astype('float'))
    recal = np.mean((dist1 < threshold).astype('float'))
    fscore = 2 * precision * recal / (precision + recal)
    metrics = {
        'Acc': np.mean(dist2),
        'Comp': np.mean(dist1),
        'Chamfer': (np.mean(dist1) + np.mean(dist2)) / 2,
        'Prec': precision,
        'Recal': recal,
        'F-score': fscore,
    }
    return metrics


parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, default="1")
parser.add_argument('--data_dir', type=str, default="./../data/scannetpp")
parser.add_argument('--mesh', type=str, default='/data/projects/implicit_reconstruction/scripts/scannetpp_7-3-aa_0e75f3c4d9_cues_2048_latest.ply')
parser.add_argument('--eval_dir', type=str, default='./eval_results')
parser.add_argument("--vis_err", action="store_true")
parser.add_argument("--gt_space", action="store_true")
parser.set_defaults(gt_space=False, vis_err=False)
args = parser.parse_args()

mesh_path = args.mesh
id = args.id
# data = mesh_path.split('/')[-1].split('.')[0].split('-')[-1]
# data = '0e75f3c4d9'
mesh = trimesh.load_mesh(mesh_path)
meta = load_from_json(Path(f'{args.data_dir}/{id}/meta_data.json'))
worldtogt = np.array(meta['worldtogt'])
# mesh.vertices = mesh.vertices / 0.8
mesh.vertices = mesh.vertices
if not args.gt_space:
    mesh.vertices = (worldtogt[:3, :3] @ mesh.vertices.T).T + worldtogt[:3, 3][None, ...]
# o3d.io.write_point_cloud(f'/mesh/pure-neus-reproduce-{data}.ply', mesh)
# mesh.export(f'test.ply')

gt_mesh = trimesh.load_mesh(f'{args.data_dir}/{id}/mesh_aligned_0.05.ply')
bbox = gt_mesh.bounds
vertices = np.asarray(mesh.vertices)
mask = (vertices > bbox[0]).all(axis=1) & (vertices < bbox[1]).all(axis=1)

# Compute a modified mesh with only the vertices inside the bounding box
vertices_in_box = vertices[mask]

# Compute a list of faces after removing the ones outside the bounding box
# faces_in_box = mesh.faces[np.all(mask[mesh.faces], axis=1)]

# Create a new mesh
mesh_in_box = trimesh.Trimesh(vertices=vertices_in_box)

# mesh_in_box.export(f'test.ply')

metrics = evaluate(mesh_in_box, gt_mesh, threshold=0.025, vis_err=args.vis_err, eval_dir=args.eval_dir)
print(metrics)

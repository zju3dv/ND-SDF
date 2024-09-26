

import torch
import numpy as np

from tqdm import tqdm
import sys
import trimesh
import skimage.measure as meature
from torch.nn import Upsample,AvgPool3d
from skimage import measure


def filter_largest_cc(mesh):
    components = mesh.split(only_watertight=False)
    areas = np.array([c.area for c in components], dtype=float)
    if len(areas) > 0 and mesh.vertices.shape[0] > 0:
        new_mesh = components[areas.argmax()]
    else:
        new_mesh = trimesh.Trimesh()
    return new_mesh

def texture_function(pts, neural_sdf, neural_rgb):
    with torch.enable_grad():
        sdf, feat, gradient, _=neural_sdf.get_all(pts, if_cal_hessian_x=False)
    normals=torch.nn.functional.normalize(gradient,dim=1)
    rgb=neural_rgb.forward(pts,-normals,normals,feat,app=None)
    # if neural_rgb.color_encoder:
    #     rgb = neural_rgb.decode_color(rgb)
    return torch.hstack([normals,rgb])


@torch.no_grad()
def my_extract_mesh(sdf_func, bounds, res, block_res, texture_func=None, filter_lcc=False):
    '''
    This function is used to extract mesh from neural sdf field, plus texture if texture_func is set.

    Args:
        sdf_func: sdf function, takes a tensor of shape (N,3) as input and return a tensor of shape (N,1) as output.
        bounds: bounds of the mesh, shape (3,2).
        res: resolution of the mesh.
        block_res: resolution of the block, used to speed up the extraction.
        texture_func: texture function, takes a tensor of shape (N,3) as input and return a tensor of shape (N,4) as output.
    Returns:
        mesh: a trimesh mesh.
    '''
    print("extracting mesh...")
    print(f"res: {res}, block_res: {block_res}")
    levels = res // block_res
    block_a = (bounds[:, 1] - bounds[:, 0]) / levels
    meshes = []
    pbar = tqdm(total=levels ** 3, leave=False, file=sys.stdout)
    idx = 0
    for xmin in np.linspace(bounds[0, 0], bounds[0, 1], levels + 1)[:-1]:
        for ymin in np.linspace(bounds[1, 0], bounds[1, 1], levels + 1)[:-1]:
            for zmin in np.linspace(bounds[2, 0], bounds[2, 1], levels + 1)[:-1]:
                idx += 1
                pbar.set_description(f"extracting mesh at ({xmin:.2f},{ymin:.2f},{zmin:.2f}) to ({xmin + block_a[0]:.2f},{ymin + block_a[1]:.2f},{zmin + block_a[2]:.2f})")
                xmax, ymax, zmax = xmin + block_a[0], ymin + block_a[1], zmin + block_a[2]
                # indexing='ij': meshgrid顺序与坐标轴顺序一致
                grid_x, grid_y, grid_z = np.meshgrid(np.linspace(xmin, xmax, block_res),np.linspace(ymin, ymax, block_res),np.linspace(zmin, zmax, block_res), indexing='ij')
                grid_x, grid_y, grid_z = torch.from_numpy(grid_x).float(), torch.from_numpy(
                    grid_y).float(), torch.from_numpy(grid_z).float()

                xyz = torch.stack([grid_x, grid_y, grid_z], dim=0)
                res_s = [block_res]
                xyz_s = [xyz]
                # 构建coarse -> fine金字塔
                coarsest_res = 64
                while xyz.shape[-1] > coarsest_res:
                    xyz = AvgPool3d(kernel_size=2, stride=2)(xyz[None])[0]
                    res_s.append(xyz.shape[-1])
                    xyz_s.append(xyz)
                res_s, xyz_s = res_s[::-1], xyz_s[::-1]

                def chunk_func(pts, func, chunk=100000):
                    v_list = []
                    for chunk_pts in torch.split(pts, chunk):
                        chunk_pts = chunk_pts.cuda()
                        chunk_v = func(chunk_pts).cpu()
                        v_list.append(chunk_v)
                    return torch.vstack(v_list)

                # 由coarse -> fine 逐层提取mesh
                # grid_mask维护当前res下感兴趣的点, grid_sdf逐步更新对finer res感兴趣点的sdf值
                grid_mask = torch.ones(size=(coarsest_res, coarsest_res, coarsest_res), dtype=torch.bool)
                grid_sdf = torch.empty(size=(coarsest_res, coarsest_res, coarsest_res), dtype=torch.float32)
                for ri, xyz in zip(res_s, xyz_s):
                    # threshold of |sdf|
                    threshold = 2 * (xmax - xmin) / ri
                    xyz = xyz.permute(1, 2, 3, 0)
                    points = xyz[grid_mask].contiguous()
                    # !!!!注意: len(points)可能为0, 此时chunk_func会报错
                    if len(points) > 0:
                        points_sdf = chunk_func(points, sdf_func).ravel()
                        grid_sdf[grid_mask] = points_sdf  # update the grid sdf
                    grid_mask = abs(grid_sdf) < threshold  # update the grid mask

                    print(f"ratio: {grid_mask.float().sum() / grid_mask.numel():.2f}")  # ratio of interested points of this res
                    # upsample
                    if ri < block_res:
                        grid_mask = Upsample(scale_factor=2)(grid_mask[None, None].float())[0, 0].bool()
                        grid_sdf = Upsample(scale_factor=2)(grid_sdf[None, None])[0, 0]

                if grid_sdf.max() > 0 and grid_sdf.min() < 0:  # 有点在表面
                    # recommend using skimage marching_cubes
                    verts, faces, normals, values = meature.marching_cubes(volume=grid_sdf.numpy(),
                                                                           level=0,
                                                                           spacing=((xmax - xmin) / (block_res - 1),
                                                                                    (ymax - ymin) / (block_res - 1),
                                                                                    (zmax - zmin) / (block_res - 1)))
                    verts += np.array([xmin, ymin, zmin])
                    # verts,faces=mcubes.marching_cubes(grid_sdf.numpy(),0)
                    # verts=verts*2/res+np.array([xmin,ymin,zmin])
                    if texture_func is not None:
                        n_cat_c = chunk_func(torch.from_numpy(verts).float().contiguous(), texture_func)
                        normals, colors = n_cat_c[:, :3].numpy(), n_cat_c[:, 3:].numpy()
                        colors = (colors * 255).astype(np.uint8)
                        mesh = trimesh.Trimesh(verts, faces, vertex_normals=normals, vertex_colors=colors)
                        # mesh.export(f"/data/projects/neuralangelo/meshes/object/{idx}.ply")
                    else:
                        mesh = trimesh.Trimesh(verts, faces)
                    # mesh = filter_points_outside_bounding_sphere(mesh)  # filter points outside bounding sphere
                    mesh = filter_largest_cc(mesh) if filter_lcc else mesh  # filter largest connected component
                    meshes.append(mesh)
                pbar.update(1)
    mesh = trimesh.util.concatenate(meshes)
    pbar.close()
    return mesh


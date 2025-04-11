import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm, trange
import numpy as np

# import utils.general as utils
from glob import glob
import cv2
import random
import json
from PIL import Image
from plyfile import PlyData


def rotation_matrix(axis, angle):
    # 确保axis是单位向量
    axis = axis / np.linalg.norm(axis)

    # 计算旋转角度的三角函数值
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    # 计算反对称矩阵K
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])

    # 旋转矩阵R
    I = np.eye(3)  # 单位矩阵
    R = I + sin_angle * K + (1 - cos_angle) * np.dot(K, K)

    # # 旋转向量: 旋转轴 * 旋转角度
    # rotation_vector = axis * angle
    #
    # # 使用 cv2.Rodrigues 计算旋转矩阵
    # rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    return R

def rotation_matrix_batch(axis, angle):
    # input: axis:(3,n), angle:(1,n)
    # output: R:(3,3,n)
    axis = axis.numpy()  # (3, n)
    angle = angle.numpy()  # (1,n)

    # 确保axis每列是单位向量
    axis = axis / np.linalg.norm(axis, axis=0)  # 按列标准化

    zero = np.zeros_like(angle, dtype=np.float32)
    batch_K = np.stack([zero, -axis[2:3], axis[1:2],
                        axis[2:3], zero, -axis[0:1],
                        -axis[1:2], axis[0:1], zero], axis=0).reshape(3, 3, -1).transpose(2, 0, 1)  # (n,3,3)
    cos_angle = np.cos(angle)   # (1,n)
    sin_angle = np.sin(angle)
    I = np.eye(3)  # (3,3)
    R = I[:, :, None] + sin_angle[None] * (batch_K.transpose(1, 2, 0)) + (1 - cos_angle[None]) * (np.matmul(batch_K, batch_K).transpose(1, 2, 0))  # (3,3,n)


    return R


class BaseDataset(Dataset):
    def __init__(self, conf, split='train', num_rays=1024, downscale=1., preload=False, custom_sampling=False,
                 fewshot=False, fewshot_idx=[22, 25, 28], monosdf=True) -> None:
        super().__init__()
        # ----------------------- rebuttal specially bias normal priors -----------------------
        self.rebuttal_bias_prior = False
        self.randomly_bias = True
        self.randomly_sigma = 60
        self.axis = np.array([1., 1., 1.]) / np.sqrt(3)
        self.bias_angle = 5  # [3,5,10,15,30,45,60]
        # ---------------------------------------------------------------------------------------
        self.monosdf = False
        if self.monosdf:
            self.data_dir = conf.data_dir if str(conf.scan_id) == '-1' else os.path.join(conf.data_dir,f'{conf.scan_id}')
        else:
            self.data_dir = conf.data_dir if str(conf.scan_id) == '-1' else os.path.join(conf.data_dir,f'{conf.scan_id}')
        self.split = split
        self.num_rays = num_rays
        self.downscale = downscale
        self.preload = preload
        self.custom_sampling = custom_sampling
        self.fewshot = fewshot
        self.fewshot_idx = fewshot_idx

        data_json = os.path.join(self.data_dir, 'meta_data.json')
        with open(data_json, 'r') as f:
            data = json.load(f)

        self.h = int(data["height"] / downscale)  # downscale
        self.w = int(data["width"] / downscale)
        self.total_pixels = self.h * self.w
        self.img_res = [self.h, self.w]
        self.bound = data["scene_box"]["radius"]

        self.has_mono_depth = data["has_mono_depth"] if not hasattr(conf, "use_mono_depth") else conf.use_mono_depth
        self.has_mono_normal = data["has_mono_normal"] if not hasattr(conf, "use_mono_normal") else conf.use_mono_normal
        # TODO: 添加2d mask
        self.has_mask = data["has_mask"] if not hasattr(conf, "use_mask") else conf.use_mask
        # TODO: 添加2d uncertainty
        self.has_uncertainty = data["use_uncertainty"] if not hasattr(conf, "use_uncertainty") else conf.use_uncertainty

        self.pts_path = data["pts_path"] if (
                    "pts_path" in data and data['pts_path'] != 'null' and getattr(conf, "use_pts", False)) else None
        if self.pts_path is not None:
            plydata = PlyData.read(os.path.join(self.data_dir, self.pts_path))
            self.pts = torch.from_numpy(np.vstack([plydata['vertex'][name] for name in ['x', 'y', 'z']]).T).float()
            # colors = np.vstack([plydata['vertex'][name] for name in ['red', 'green', 'blue']]).T
            self.pts_normal = torch.from_numpy(
                np.vstack([plydata['vertex'][name] for name in ['nx', 'ny', 'nz']]).T).float()
            # errors = np.array(plydata['vertex']['error'])
            self.pts_confidence = torch.from_numpy(np.array(plydata['vertex']['confidence'])).float()[:, None]

        self.scale_mat = data["worldtogt"]
        frames = data["frames"]
        self.n_images = len(frames)
        self.rgb_paths = []
        self.mono_depth_paths = []
        self.mono_normal_paths = []
        self.mask_paths = []
        self.uncertainty_paths = []
        self.poses = []
        self.intrinsics = []

        self.preload_cache = []
        if self.preload:
            print(f"Preloading {self.data_dir} with {self.n_images} images...")
        for idx, frame in enumerate(frames):
            rgb_path = os.path.join(self.data_dir, frame["rgb_path"])
            pose = np.array(frame["camtoworld"], dtype=np.float32)
            intrinsic = np.array(frame["intrinsics"], dtype=np.float32)
            # 根据downscale进行缩放
            intrinsic[0] = intrinsic[0] / (data["width"] / self.w)
            intrinsic[1] = intrinsic[1] / (data["height"] / self.h)

            self.rgb_paths.append(rgb_path)
            self.poses.append(pose)
            self.intrinsics.append(intrinsic)

            if self.has_mono_depth:
                self.mono_depth_paths.append(os.path.join(self.data_dir, frame["mono_depth_path"]))
            if self.has_mono_normal:
                self.mono_normal_paths.append(os.path.join(self.data_dir, frame["mono_normal_path"]))
            if self.has_mask:
                if self.monosdf == True:  # monosdf数据的mask
                    self.mask_paths.append(
                        os.path.join(self.data_dir, frame["mono_normal_path"].replace('normal', 'mask')))
                else:
                    self.mask_paths.append(os.path.join(self.data_dir, frame["mask_path"]))
            if self.has_uncertainty:
                self.uncertainty_paths.append(os.path.join(self.data_dir, frame["uncertainty_path"]))

            if self.preload:
                self.preload_cache.append((self.load_data(idx)))
        print(f"Loaded {self.data_dir} with {self.n_images} images successfully, split: {split}")
        self.loop_all = True if split == 'train' else False

    def set_loop_all(self):
        self.loop_all = True

    def __len__(self):
        return self.n_images if self.loop_all else 1

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def get_scale_mat(self):
        return self.scale_mat

    def sobel(self, x):
        gx = cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=3)
        gm = np.sqrt(gx ** 2 + gy ** 2)
        return gm

    def load_data(self, idx):
        # load data from disk
        rgb = np.asarray(Image.open(self.rgb_paths[idx])).astype(np.float32) / 255
        if self.downscale > 1:
            # cv2.INTER_NEAREST  # 最邻近差值法
            # cv2.INTER_LINEAR  # 双线性差值法
            # cv2.INTER_AREA  # 基于局部像素的重采样法
            # cv2.INTER_CUBIC  # 三次差值法
            # cv2.INTER_LANCZOS4  # 基于8*8像素邻域的Lanczos差值法, 可能会溢出
            rgb = cv2.resize(rgb, (self.w, self.h), interpolation=cv2.INTER_AREA)
        rgb = torch.from_numpy(rgb.reshape(-1, 3)).float()
        if self.has_mono_depth:
            depth = np.load(self.mono_depth_paths[idx])  # (h,w)
            if self.downscale > 1:
                depth = cv2.resize(depth, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            depth = torch.from_numpy(depth).float().reshape(-1, 1)
        else:
            depth = torch.zeros_like(rgb[:, :1])
        if self.has_mono_normal:
            normal = np.load(self.mono_normal_paths[idx])
            if self.monosdf:  # FIXME：monosdf保存的normal是(3,h,w)
                normal = normal.transpose(1, 2, 0)  # (3,h,w) -> (h,w,3)
            if self.downscale > 1:
                normal = cv2.resize(normal, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            normal = torch.from_numpy(normal)
            normal = normal.reshape(-1, 3).float()
            normal = normal * 2 - 1
            normal = F.normalize(normal, p=2, dim=-1)
            # if self.rebuttal_bias_prior:
            #     pose = torch.from_numpy(self.poses[idx])
            #     c2w_R = pose[:3, :3]
            #     normal_gt = c2w_R @ normal.T  # (3, h*w)
            #     bg_mask_dir = os.path.join(self.data_dir, 'mask')
            #     mask_paths = glob(os.path.join(bg_mask_dir, 'res', '*.npy'))
            #     mask_paths.sort()
            #     mask_path = mask_paths[idx]
            #     bg_mask = np.load(mask_path).astype(np.float32)
            #     bg_mask = cv2.resize(bg_mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
            #     bg_mask = torch.from_numpy(bg_mask > 0).reshape(-1).bool()
            #     if bg_mask.sum():
            #         # 将bg_mask 选中的部分的normal进行bias
            #         if not self.randomly_bias:  # 朝self.axis旋转self.bias_angle度
            #             # bias_angle = self.bias_angle * np.pi / 180
            #             # bias_axis = self.axis
            #             # second choice: 向bias_axis方向旋转bias_angle度即绕（normal_gt叉乘bias_axis）旋转bias_angle度
            #             target_axis = torch.from_numpy(self.axis).float()[:, None]  # (3,1)
            #             bias_axis = F.normalize(F.normalize(normal_gt, p=2, dim=0).cross(target_axis, dim=0), p=2,dim=0)
            #             bias_angle = torch.tensor(self.bias_angle * np.pi / 180).float().repeat(1, self.w*self.h)
            #             bias_R = torch.from_numpy(rotation_matrix_batch(bias_axis, bias_angle)).float()
            #             normal_gt[:, bg_mask.flatten()] = (bias_R[:,:,bg_mask.flatten()].permute(2,0,1) @ normal_gt[:,None, bg_mask.flatten()].permute(2,0,1))[...,0].T
            #         else: # 朝着随机方向旋转随机角度，随机角度服从高斯分布G(0,self.randomly_sigma^2)
            #             bias_axis = F.normalize(F.normalize(normal_gt, p=2, dim=0).cross(torch.randn_like(normal_gt), dim=0), p=2,dim=0) # 完全随机轴
            #             # second choice: 轴固定，随机角度
            #             # target_axis = torch.from_numpy(self.axis).float()[:, None]  # (3,1)
            #             # bias_axis = F.normalize(F.normalize(normal_gt, p=2, dim=0).cross(target_axis, dim=0), p=2,dim=0)
            #             bias_angle = torch.randn_like(torch.zeros((1,self.w*self.h))) * self.randomly_sigma
            #             bias_R = torch.from_numpy(rotation_matrix_batch(bias_axis, bias_angle / 180 * np.pi)).float()
            #             normal_gt[:, bg_mask.flatten()] = (bias_R[:,:,bg_mask.flatten()].permute(2,0,1) @ normal_gt[:,None, bg_mask.flatten()].permute(2,0,1))[...,0].T
            #
            #         normal = (torch.linalg.inv(c2w_R) @ normal_gt).T
            #         vis_biased_normal = cv2.cvtColor(((normal.reshape(self.h, self.w, 3).numpy()+1)/2*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            #         if not self.randomly_bias:
            #             biased_save_dir = os.path.join(self.data_dir, f'biased_normal_{self.bias_angle}')
            #         else:
            #             biased_save_dir = os.path.join(self.data_dir, f'biased_normal_random_{self.randomly_sigma}')
            #         os.makedirs(biased_save_dir, exist_ok=True)
            #         cv2.imwrite(os.path.join(biased_save_dir, f'{idx}.png'), vis_biased_normal)
            #         # mo = torch.norm(normal_, dim=-1, keepdim=True)
            #         # biased_angle = torch.acos(torch.sum(normal * normal_, dim=-1).clip(-1,1))/np.pi*180
            #         # biased_angle = biased_angle[bg_mask]
            #         print(1)
        else:
            normal = torch.zeros_like(rgb)
        if self.has_mask:
            mask = np.load(self.mask_paths[idx])
            # mask_image = (mask * 255).astype(np.uint8)[..., None].repeat(3, axis=-1)
            # cv2.imshow('mask', mask_image)
            # cv2.waitKey(0)
            if self.downscale > 1:
                mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
            mask = torch.from_numpy(mask).long().reshape(-1, 1)
        else:
            mask = torch.ones_like(depth, dtype=torch.long)
        if self.has_uncertainty:
            uncertainty = np.load(self.uncertainty_paths[idx])
            if self.downscale > 1:
                uncertainty = cv2.resize(uncertainty, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            uncertainty_grad = self.sobel(uncertainty)
            uncertainty, uncertainty_grad = torch.from_numpy(uncertainty).float().reshape(-1, 1), torch.from_numpy(
                uncertainty_grad).float().reshape(-1, 1)
        else:
            uncertainty = torch.zeros_like(depth)
            uncertainty_grad = torch.zeros_like(depth)
        return rgb, depth, normal, mask, uncertainty, uncertainty_grad

    def get_rays(self, K, pose, x=None, y=None):
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        sk = K[0, 1]  # skew: 坐标轴倾斜参数, 理想情况下应该为0
        if x is None:
            x, y = np.meshgrid(np.arange(0, self.w, 1), np.arange(0, self.h, 1))
            x, y = x + 0.5, y + 0.5
        y_cam = (y - cy) / fy * 1
        x_cam = (x - cx - sk * y_cam) / fx * 1  # 把cam->pixel的转换写出来即可得到这个式子
        z_cam = np.ones_like(x_cam)  # 假设z=1
        d = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (h,w,3)
        d = d / np.linalg.norm(d, axis=-1, keepdims=True)
        depth_scale = d.reshape(-1, 3)[:, 2:]  # 即z轴余弦角cosθ，用于计算depth
        d = (pose[:3, :3] @ (d.reshape(-1, 3).T)).T  # c2w
        o = np.tile(pose[:3, -1][None, :], (d.shape[0], 1))

        return torch.from_numpy(o).float(), torch.from_numpy(d).float(), torch.from_numpy(depth_scale).float()

    def __getitem__(self, idx):
        if self.fewshot:
            # print('idx:', idx)
            # idx = self.fewshot_idx[idx] if self.loop_all else self.fewshot_idx[0] # FIXME：神奇，Dataset保证一个epoch遍历完所有图像但并不是纯随机的，一个epoch不能重复。测试不如直接纯随机
            i = random.randint(0, len(self.fewshot_idx) - 1)
            idx = self.fewshot_idx[i]
            # print('idx:', idx)
        else:
            pass
            # idx = random.randint(0, self.__len__()-1)
        sample = {
            "idx": torch.tensor(idx, dtype=torch.long),
            "K": torch.from_numpy(self.intrinsics[-1]),
            "pose": torch.from_numpy(self.poses[idx])
        }
        if self.preload:
            rgb, depth, normal, mask, uncertainty, uncertainty_grad = self.preload_cache[idx]
        else:
            rgb, depth, normal, mask, uncertainty, uncertainty_grad = self.load_data(idx)
        x, y = np.meshgrid(np.arange(0, self.w, 1), np.arange(0, self.h, 1))
        x, y = x + 0.5, y + 0.5
        if self.split == 'valid':
            rays_o, rays_d, depth_scale = self.get_rays(self.intrinsics[-1], self.poses[idx], x=x.ravel(), y=y.ravel())
            sample["rays_o"] = rays_o
            sample["rays_d"] = rays_d
            sample["depth_scale"] = depth_scale

            sample['h'], sample['w'] = torch.tensor(self.h).long(), torch.tensor(self.w).long()
            # sample["sampling_idx"] = torch.arange(self.total_pixels)
            sample["rgb"] = rgb
            sample["normal"] = normal
            sample["depth"] = depth
            sample["mask"] = mask
            sample["uncertainty"] = uncertainty
            sample["uncertainty_grad"] = uncertainty_grad
        elif self.split == 'train':  # train时减小cpu瓶颈就在forward前get_rays
            if self.custom_sampling:  # yield all pixels
                sample['perm'] = torch.randperm(self.total_pixels)
                sample['h'], sample['w'] = torch.tensor(self.h).long(), torch.tensor(self.w).long()
                # sample["sampling_idx"] = sampling_idx
                sample["rgb"] = rgb
                sample["normal"] = normal
                sample["depth"] = depth
                sample["mask"] = mask
                sample["uncertainty"] = uncertainty
                sample["uncertainty_grad"] = uncertainty_grad
            else:  # get rays here
                sampling_idx = torch.randperm(self.total_pixels)[:self.num_rays]
                rays_o, rays_d, depth_scale = self.get_rays(self.intrinsics[-1], self.poses[idx],
                                                            x.ravel()[sampling_idx], y.ravel()[sampling_idx])
                sample["rays_o"], sample["rays_d"], sample["depth_scale"] = rays_o, rays_d, depth_scale
                sample['sampling_idx'] = sampling_idx
                sample['h'], sample['w'] = torch.tensor(self.h).long(), torch.tensor(self.w).long()
                sample["rgb"] = rgb[sampling_idx]
                sample["normal"] = normal[sampling_idx]
                sample["depth"] = depth[sampling_idx]
                sample["mask"] = mask[sampling_idx]
                sample["uncertainty"] = uncertainty[sampling_idx]
                sample["uncertainty_grad"] = uncertainty_grad[sampling_idx]
        return sample

    def rnd_getitem(self):
        idx = random.randint(0, self.n_images - 1)
        return self.__getitem__(idx)

import time
from glob import glob

import argparse
import sys, os, datetime,shutil
from tqdm import tqdm

import numpy as np
import torch.distributed as dist
from torch.autograd import profiler
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf

from models.system import ImplicitReconSystem
from models.loss import ImplicitReconLoss, get_psnr, compute_scale_and_shift
from dataset.base_dataset import BaseDataset
from dataset.dataloader import MultiEpochsDataLoader
from utils.mesh import my_extract_mesh
from torch.utils.tensorboard import SummaryWriter
from utils.utils import *

# 添加cuda路径
env_list = os.environ['PATH'].split(':')
env_list.append('/usr/local/cuda/bin')
os.environ['PATH'] = ':'.join(env_list)

def init_processes():
    # 获取rank
    gpu = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])

    # local_rank用于初始化torch device
    torch.cuda.set_device(gpu)
    torch.manual_seed(0)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank, )
    print('device {}/{} started...'.format(rank, world_size))
    dist.barrier()
    return gpu


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='/home/dawn/projects/ND-SDF/confs/scannet.yaml', help='conf')
    parser.add_argument('--data_dir', type=str, default='', help='scan_id')
    parser.add_argument('--scan_id', type=str, default='1', help='scan_id')
    parser.add_argument('--epoches', type=int, default=9999999999)
    parser.add_argument('--root_dir', type=str, default='runs', help='实验根目录')
    parser.add_argument('--is_continue', action='store_true', help='continue')
    parser.add_argument('--checkpoint', default='latest', type=str, help='checkpoint')
    parser.set_defaults(is_continue=False)
    # parser.add_argument("--local_rank", type=int, required=True, help='local rank for DDP')
    opt = parser.parse_args()
    return opt

def get_parms_num(p):
    if isinstance(p, list):
        return sum([get_parms_num(i) for i in p])
    if isinstance(p, dict):
        return get_parms_num(p['params'])
    return p.numel()

class Trainer():
    def __init__(self, opt, gpu):
        self.conf = OmegaConf.load(opt.conf)
        print('desc:', getattr(self.conf, 'desc', 'no description')) # 打印描述
        self.conf.dataset.scan_id = opt.scan_id if opt.scan_id != '-1' else self.conf.dataset.scan_id # replace scan_id
        self.conf.dataset.data_dir = opt.data_dir if opt.data_dir != '' else self.conf.dataset.data_dir # replace data_dir
        self.root_dir = opt.root_dir
        self.exp_name = self.conf.train.exp_name if str(self.conf.dataset.scan_id) == '-1' else self.conf.train.exp_name + f'_{self.conf.dataset.scan_id}'
        self.epoches = opt.epoches
        self.last_epoch = 0
        self.cur_step = 0
        self.is_continue = opt.is_continue
        self.gpu = gpu
        self.batch_size = self.conf.train.batch_size
        self.chunk=self.conf.train.chunk
        self.custom_sampling = getattr(self.conf.train, 'custom_sampling', False)
        self.dynamic_sampling = getattr(self.conf.train, 'dynamic_sampling', False)
        self.anneal_quat_end = getattr(self.conf.optim.sched, 'anneal_quat_end', 0.2)
        self.init_num_rays = self.conf.train.num_rays
        self.num_rays = self.conf.train.num_rays
        self.ema_decay = getattr(self.conf.optim.sched, 'ema_decay', 0.9)
        self.train_downscale = self.conf.train.train_downscale
        self.valid_downscale = self.conf.train.valid_downscale
        print('exp_name:', self.exp_name)

        # 实验相关目录
        self.exp_dir = os.path.join(self.root_dir, self.exp_name, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(self.exp_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(self.checkpoint_dir,'angle'), exist_ok=True)
        self.log_dir = os.path.join(self.exp_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.plot_dir = os.path.join(self.exp_dir, 'plots')
        os.makedirs(self.plot_dir, exist_ok=True)

        # save conf
        with open(os.path.join(self.exp_dir, 'conf.yaml'), 'w') as f:
            OmegaConf.save(self.conf, f)

        # dataset
        self.train_dataset = BaseDataset(self.conf.dataset, split='train', num_rays=self.conf.train.num_rays, downscale=self.train_downscale, preload=True, custom_sampling=self.custom_sampling or self.dynamic_sampling, fewshot=getattr(self.conf.dataset, 'fewshot', False), fewshot_idx=getattr(self.conf.dataset, 'fewshot_idx', []))
        self.train_total_pixels, self.train_h, self.train_w = self.train_dataset.total_pixels, self.train_dataset.h, self.train_dataset.w
        self.valid_dataset = BaseDataset(self.conf.dataset, split='valid', num_rays=self.conf.train.num_rays, downscale=self.valid_downscale, preload=False, fewshot=getattr(self.conf.dataset, 'fewshot', False),fewshot_idx=getattr(self.conf.dataset, 'fewshot_idx',[]))
        self.valid_total_pixels, self.valid_h, self.valid_w = self.valid_dataset.total_pixels, self.valid_dataset.h, self.valid_dataset.w
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True)
        self.valid_sampler = torch.utils.data.distributed.DistributedSampler(self.valid_dataset, shuffle=False)
        self.dataloader = MultiEpochsDataLoader(self.train_dataset, batch_size=self.conf.train.batch_size,sampler=self.train_sampler,
                                                num_workers=6, pin_memory=True, drop_last=True,persistent_workers=False)
        self.valid_dataloader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=1, sampler=self.valid_sampler) # eval rendering 1 image per plot_freq
        self.bound = self.train_dataset.bound if getattr(self.conf.model, 'bound', -1) == -1 else self.conf.model.bound
        self.grid_bound = self.bound if getattr(self.conf.train, 'grid_bound', -1) == -1 else self.conf.train.grid_bound
        # save fewshot images
        if getattr(self.conf.dataset, 'fewshot', False):
            fewshot_dir = os.path.join(self.exp_dir, 'fewshot')
            os.makedirs(fewshot_dir, exist_ok=True)
            for i, idx in enumerate(self.train_dataset.fewshot_idx):
                rgb_path = self.train_dataset.rgb_paths[idx]
                shutil.copy(rgb_path, os.path.join(fewshot_dir, f'fewshot_{i}_{idx}.png'))

        # model
        self.model = ImplicitReconSystem(self.conf, bound=self.bound, device=gpu).cuda()

        # loss
        self.loss = ImplicitReconLoss(**self.conf.loss, optim_conf=self.conf.optim)
        self.loss.set_patch_size(self.conf.train.num_rays) # num_rays->(a, b), a,b|num_rays

        # record pixel-wise rgb/angle
        if self.conf.model.nbfield.enabled and self.conf.dataset.use_mono_normal:
            self.train_angle = torch.zeros(size=(self.train_dataset.n_images, self.train_total_pixels))

        # optimizer
        sdf_conf = self.conf.model.object.sdf
        rgb_conf = self.conf.model.object.rgb
        bg_conf = self.conf.model.background
        nb_conf = self.conf.model.nbfield
        optim_conf = self.conf.optim
        if optim_conf.type == 'AdamW':
            optim = torch.optim.AdamW
        elif optim_conf.type == 'Adam':
            optim = torch.optim.Adam
        ############################# Parameter Group&Optimizer #############################
        params = []
        params += [{'name': 'sdf-mlp', 'params': self.model.sdf.get_mlp_params(), 'lr': optim_conf.lr},
                   {'name': 'radiance', 'params': self.model.rgb.parameters(), 'lr': optim_conf.lr}]  # sdf mlp, radiance
        if sdf_conf.enable_hashgrid:  # multi-res hash encoder, 高学习率可以加速训练
            params += [{'name': 'hash-encoder', 'params': self.model.sdf.get_grid_params(),'lr': optim_conf.lr * optim_conf.lr_scale_grid}]
        if bg_conf.enabled:  # bg nerf
            if bg_conf.type == 'grid_nerf':
                params += [{'name': 'hash-encoder-bg', 'params': self.model.bg_nerf.get_grid_params(),'lr': optim_conf.lr * optim_conf.lr_scale_grid}]
                params += [{'name': 'mlp-bg', 'params': self.model.bg_nerf.get_mlp_params(), 'lr': optim_conf.lr}]
            elif bg_conf.type == 'nerf++':
                params += [{'name': 'background-nerf', 'params': self.model.bg_nerf.parameters(), 'lr': optim_conf.lr}]
            else:
                raise ValueError('Unknown background type')
        params += [{'name': 'density', 'params': self.model.density.parameters(), 'lr':optim_conf.lr * optim_conf.get('lr_scale_density', 1)}] # density
        if rgb_conf.enable_app:  # appearance object/scene
            params += [{'name': 'appearance', 'params': self.model.app.parameters(), 'lr': optim_conf.lr}]
        if bg_conf.enable_app and bg_conf.enabled:  # appearance bg
            params += [{'name': 'appearance-bg', 'params': self.model.app_bg.parameters(), 'lr': optim_conf.lr}]
        if nb_conf.enabled:  # nbfield
            params += [{'name': 'nbfield', 'params': self.model.nb.parameters(), 'lr': optim_conf.lr}]
        self.optimizer = optim(params=params, betas=(0.9, 0.99), eps=1e-15)
        self.optim_zero_grad_kwargs = {'set_to_none': True} # set_to_none=True, 释放梯度内存，可以减少内存占用并加速。
        print("--------Model Size---------")
        print(f"{get_parms_num(params)/1e6:.2f}M")

        # 1. LambdaLR自定义two_steps; 2. exponential
        self.max_step = min(self.conf.train.max_step, self.epoches * len(self.dataloader))
        self.scheduler = self.get_scheduler(self.optimizer,self.conf.optim.sched)

        # load checkpoint
        if self.is_continue:
            self.load_checkpoint(opt.checkpoint)

        # init DDP
        self.model = DDP(self.model, device_ids=[gpu], output_device=gpu, find_unused_parameters=True)

        # tensorboard
        self.loger = SummaryWriter(self.log_dir)

    def get_scheduler(self, optimizer, sched_conf):
        if self.conf.optim.sched.type == 'two_steps_lr':
            def lr_lambda(step): # 即lr:[warm, 1, gamma, gamma^2]对应的step:[0, warm_up_end, two_steps[0], two_steps[1], end]
                if step < self.conf.optim.sched.warm_up_end:
                    return step / self.conf.optim.sched.warm_up_end
                elif step < sched_conf.two_steps[0]:
                    return 1.
                elif step < sched_conf.two_steps[1]:
                    return sched_conf.gamma
                else:
                    return sched_conf.gamma ** 2

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        elif self.conf.optim.sched.type == 'exponential_lr':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=sched_conf.gamma**(1/self.max_step))
        else:
            raise ValueError('Unknown scheduler type')
        return scheduler

    def load_checkpoint(self, checkpoint):
        if checkpoint == 'latest':
            timestamp_dir = os.path.join(self.root_dir, self.exp_name)
            # 找到最新的timestamp dir
            timestamps = glob(os.path.join(timestamp_dir, '*'))
            ckpt_paths = []
            for timestamp in timestamps:
                ckpt_paths.extend(glob(os.path.join(timestamp, 'checkpoints', 'latest.pth')))
            ckpt_paths.sort(key=os.path.getmtime)
            if len(ckpt_paths)==0:
                print("Failing to continue training and restart from the beginning.")
                return
            checkpoint = ckpt_paths[-1] # -1是刚创建的
        ckpt = torch.load(checkpoint, map_location='cuda:{}'.format(self.gpu))
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.last_epoch = ckpt['epoch']
        self.cur_step = ckpt['step']
        if self.conf.model.nbfield.enabled and self.conf.dataset.use_mono_normal:
            self.train_angle = torch.load(os.path.join(os.path.dirname(checkpoint),'angle', 'angle.pt'))
        if self.gpu == 0:
            print(f'Continue training! Last epoch: {self.last_epoch}')
            print('Loaded checkpoint from {}'.format(dist.get_rank(), dist.get_world_size(),self.last_epoch, checkpoint))
            print(self.model)
            print(self.optimizer)

    def save_checkpoint(self, epoch, save_epoch=False):
        if save_epoch:
            ckpt = {
                'model': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': epoch,
                'step': self.cur_step,
                'scale_mat': self.train_dataset.scale_mat,
            }
            torch.save(ckpt, os.path.join(self.checkpoint_dir, 'latest.pth'))
            torch.save(ckpt, os.path.join(self.checkpoint_dir, f'epoch_{epoch}.pth'))
            if getattr(self, 'train_angle', None) is not None:
                torch.save(self.train_angle, os.path.join(self.checkpoint_dir,'angle', 'angle.pt'))

    def plot(self, epoch, if_rendering=True, if_extract_mesh=True):
        # FIXME：多卡训练rendering时需所有卡都rendering不然会死锁（所有gpu占用率100%）, 未知原因，所以一起rendering。但extract mesh不会。
        print('plotting...')
        self.model.eval() # eval mode
        # 1. plot rgb、depth、normal
        if if_rendering:
            sample = next(iter(self.valid_dataloader))
            # sample = self.valid_dataset.__getitem__(0)
            sample = {k: v.cuda() for k, v in sample.items()}
            split_sample = split_input(sample,self.valid_total_pixels, self.chunk)
            outputs = []
            for s in tqdm(split_sample, total=len(split_sample), desc=f'rendering valid...', file=sys.stdout):
                output = self.model(s)
                d= {'rgb': output['rgb'].detach(), 'depth': output['depth'].detach(), 'normal': output['normal'].detach()}
                if self.conf.model.nbfield.enabled and self.conf.dataset.use_mono_normal:
                    d['quat'] = output['quat'].detach()
                    d['biased_normal'] = output['biased_normal'].detach()
                    d['biased_mono_normal'] = output['biased_mono_normal'].detach()
                outputs.append(d)
            outputs = merge_output(outputs) # plot rgb、depth、normal
            plot_outputs = get_plot_data(outputs, sample, self.valid_h,self.valid_w, monocular_depth=self.train_dataset.has_mono_depth)
            for i, plot_output in enumerate(plot_outputs):
                idx = plot_output['idx']
                for k, v in plot_output.items(): # v是PIL.Image,k∈['idx', 'rgb', 'depth', 'normal', 'merge']
                    if k!='idx':
                        os.makedirs(os.path.join(self.plot_dir, f'{k}'), exist_ok=True)
                        v.save(os.path.join(self.plot_dir, f'{k}', f'epoch{epoch}_view{idx}.png'))
                        # v.save(os.path.join(self.plot_dir, f'{k}_epoch{epoch}_view{idx}.png'))
        # 2. extract mesh
        if if_extract_mesh and self.gpu==0:
            mesh=my_extract_mesh(sdf_func=self.model.module.sdf.get_sdf, bounds=np.array([[-self.grid_bound,self.grid_bound],[-self.grid_bound,self.grid_bound],[-self.grid_bound,self.grid_bound]]),
                                res=self.conf.train.mesh_resolution, block_res = self.conf.train.block_resolution)
            if mesh:
                mesh.export(os.path.join(self.plot_dir, f'mesh_{epoch}.ply'), 'ply')
        self.model.train() # train mode

    def angle_guided_sampling(self, sample):
        # TODO LIST 1：set一个适合的转换函数，将angle转换为概率分布，[5°, 30°]为smooth 和 high frequency/thin structures的two steps区间。！补充：>15°确保能基本找到所有high freq和thin structures。
        def scale_shift_sigmoid_func(angle, params=None):
            # angle ∈ [0, π]
            # y=1+1/(1+e^(-beta*(x-angle_threshold)))*times ∈ [1,1+times]
            # [choice0 smooth]     ：  [beta, angle_threshold, times] = [20, 15/180*np.pi, 1]： y( 5°)≈1.03, y(10°)≈1.15, y(15°)≈1.5, y(20°)≈1.85, y(30°)≈2
            # [choice1 abrupt]     ：  [beta, angle_threshold, times] = [25, 15/180*np.pi, 2]： y( 5°)≈1.02, y(10°)≈1.2,  y(15°)=2  , y(20°)≈2.9,  y(30°)=3
            # [choice2 more abrupt+]： [beta, angle_threshold, times] = [25, 15/180*np.pi, 4]： y( 5°)≈1.05, y(10°)≈1.4,  y(15°)=3  , y(20°)≈4.6,  y(30°)≈5
            # [choice3 more abrupt++]：[beta, angle_threshold, times] = [50, 15/180*np.pi, 9]： y( 5°)≈1.00, y(10°)≈1.1,  y(15°)=5.5, y(20°)≈9.9,  y(30°)≈10
            # 经过check angle，普遍bias>15°（high freq）是bias<5°（smooth）的10%-30%之间。
            beta = 25
            angle_threshold = 15/180*np.pi
            times = 4
            if params is not None:
                beta, angle_threshold, times = params[0], params[1]/180*np.pi, params[2]
            return 1 + 1/(1+torch.exp(-beta*(angle-angle_threshold)))*times
        angle_map = self.train_angle[sample['idx'].cpu()].cuda() # (B, train_total_pixels)
        prob_map = scale_shift_sigmoid_func(angle_map, getattr(self.conf.optim.sched, 'guided_sampling_params', None)) # (B, total_pixels) TODO LIST 2：实时更新self.train_angle，但使用阶段式固定的prob_map以减少prepare overhead。
        prob_map = prob_map / prob_map.sum(dim=1, keepdim=True) # 归一化
        # per-image sample B×self.num_rays 条rays
        sampling_idx = torch.multinomial(prob_map, self.num_rays, replacement=False) # (B, num_rays)
        sampled_angle = torch.gather(angle_map, 1, sampling_idx) # (B, num_rays)
        print('15° is {} times of 5°'.format((sampled_angle>(15/180*np.pi)).sum()/(sampled_angle<(5/180*np.pi)).sum()))
        return sampling_idx

    def prepare_sample(self, sample, progress=0.0):
        # sample: dict, 'rgb':(B,H*W,3)...
        # 'perm':(B, train_total_pixels), 'h':(B), 'w':(B), 'K':(B,3,3), 'pose':(B,4,4) to get_rays rays_o, rays_d, depth_scale
        from models.nerf_util import get_rays_batch_image
        # TODO:
        if self.custom_sampling or self.dynamic_sampling:
            # sampling_idx = torch.randperm(self.train_total_pixels, device=self.gpu)[:self.num_rays][None].repeat(self.batch_size, 1) # (B, num_rays), 1. all batch same sampling_idx
            sampling_idx = sample['perm'][:, :self.num_rays] # 2. batch different random sampling_idx
            # TODO: nbfield train angle
            if_angle_guided_sampling = getattr(self.conf.optim.sched, 'if_guided_sampling', False)
            anneal_start_guided_prog = self.anneal_quat_end
            if self.conf.model.nbfield.enabled and self.conf.dataset.use_mono_normal:
                sample['train_angle'] = torch.gather(self.train_angle[sample['idx'].cpu()], 1, sampling_idx.cpu()).cuda() # (B, num_rays)
                if if_angle_guided_sampling and progress>anneal_start_guided_prog:
                    sampling_idx = self.angle_guided_sampling(sample) # TODO：3. biased angle guided sampling, sample more rays on complex and

            sample['sampling_idx'] = sampling_idx
            rays_o, rays_d, depth_scale = get_rays_batch_image(sample['K'], sample['pose'], sample['h'], sample['w'], sampling_idx)
            sample = {k: (torch.gather(v,1,sampling_idx[...,None].repeat(1,1,v.shape[-1])) if v.ndim > 2 and v.shape[1] == self.train_total_pixels else v) for k, v in sample.items()} # (B, num_rays, -1)
            sample['rays_o'] = rays_o
            sample['rays_d'] = rays_d
            sample['depth_scale'] = depth_scale
        else:
            sampling_idx = sample['sampling_idx']
            if self.conf.model.nbfield.enabled and self.conf.dataset.use_mono_normal:
                sample['train_angle'] = torch.gather(self.train_angle[sample['idx'].cpu()], 1, sampling_idx.cpu()).cuda()
        if self.train_dataset.pts_path is not None:
            rand_idx = torch.randint(self.train_dataset.pts.shape[0], (self.num_rays,), device=rays_o.device)
            sample['pts'] = self.train_dataset.pts[rand_idx].to(rays_o.device)
            sample['pts_normal'] = self.train_dataset.pts_normal[rand_idx].to(rays_o.device)
            sample['pts_confidence'] = self.train_dataset.pts_confidence[rand_idx].to(rays_o.device)
        return sample

    def update_train_angle(self, output, sample):
        # angle: (B, num_rays, 1), sample: dict, 'idx':(B), 'sampling_idx':(B, num_rays)
        angle = output['angle']
        train_angle = sample['train_angle']
        iter_angle = torch.maximum(self.ema_decay*train_angle, angle[:,:,0])
        if output.get('rays_fg',None) is not None: # occ enabled
            foreground_mask = output['rays_fg'] # (B, R, 1)
        else:
            sdf = output['sdf']
            foreground_mask = (sdf > 0.).any(dim=-2) & (sdf < 0.).any(dim=-2) # foreground mask is scene mask
        iter_angle[~foreground_mask[:,:,0]] = 0 # set background angle=0
        self.train_angle[sample['idx'].cpu()] = torch.scatter(self.train_angle[sample['idx'].cpu()], dim=1, index=sample['sampling_idx'].cpu(), src=iter_angle.cpu())

    def set_num_rays(self, max_num_rays, num_samples_per_ray, num_samples):
        num_rays = int(self.num_rays * ((self.init_num_rays*num_samples_per_ray)/num_samples)) # per-image num_rays
        self.num_rays = min(int((self.num_rays*0.95 + num_rays*0.05)), max_num_rays)

    def train(self):
        self.grad_clip = -1
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        print('start training...')
        print('total steps:', self.max_step, 'total epochs:', min(self.epoches, self.max_step // len(self.dataloader)))
        for epoch in range(self.last_epoch + 1, self.epoches + 1):
            # epoch_st = time.time()
            # last_t = time.time()
            self.dataloader.sampler.set_epoch(epoch)
            dataloader_wrapper = tqdm(enumerate(self.dataloader), total=len(self.dataloader), file=sys.stdout, desc=f"Epoch{epoch}")
            for i, sample in dataloader_wrapper:
                progress = self.cur_step / self.max_step
                ##################################### Update start Step ##############################################
                self.model.module.update_occ(self.cur_step)
                if self.conf.model.object.sdf.enable_hashgrid:
                    self.model.module.sdf.set_active_levels(self.cur_step)
                    self.model.module.sdf.set_normal_epsilon()
                    self.loss.set_curvature_weight(self.cur_step, self.model.module.sdf.anneal_levels,self.model.module.sdf.per_level_scale)  # [CHANGE-1]
                if self.conf.model.background.enabled and self.conf.model.background.type == 'grid_nerf': # bg nerf的active levels
                    self.model.module.bg_nerf.set_active_levels(self.cur_step)
                ####################################################################################################
                
                # forward
                # start.record()
                sample = {k: v.to(self.gpu) for k, v in sample.items()}  # to gpu
                sample = self.prepare_sample(sample, progress)
                sample['progress'] = progress  # progress
                output = self.model(sample)
                # end.record()
                # torch.cuda.synchronize()
                # print(f'[forward total time] {start.elapsed_time(end) / 1000:.4f}s', )

                # loss
                # start.record()
                losses = self.loss(output, sample, progress)
                loss = losses['total']
                # end.record()
                # torch.cuda.synchronize()
                # print(f'[loss time] {start.elapsed_time(end) / 1000:.4f}s', )


                # backward
                # start.record()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # restrict gradient
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                # end.record()
                # torch.cuda.synchronize()
                # print(f'[backward time] {start.elapsed_time(end) / 1000:.4f}s', )
                self.cur_step += 1
                # start.record()
                # print per step info
                if self.gpu == 0:
                    psnr = get_psnr(output['rgb'], sample['rgb'], mask=~output['outside'])
                    alpha_inv_s = 1/self.model.module.density.get_beta(prog=progress) if self.conf.model.type == 'volsdf' else self.model.module.density.get_s()
                    loss_info = '[Losses] ' + ', '.join([f'{k}:{v.item():.4f}' for k, v in losses.items()])
                    info = loss_info+f' [psnr]:{psnr.item():.4f}, [α/inv_s]:{alpha_inv_s:.4f}, [num_samples]:{self.batch_size}×{output["num_samples"]}, [num_rays]:{self.batch_size}×{self.num_rays}'
                    if self.conf.model.object.sdf.enable_hashgrid:
                        info += f', [active_levels]:{self.model.module.sdf.active_levels}/{self.model.module.sdf.num_levels}'
                    dataloader_wrapper.set_postfix_str(info)

                # print(f'[loss]: total:{loss.item():.4f}, eik:{losses["eik"].item():.4f}, rgb_l1:{losses["rgb_l1"].item():.4f}, rgb_mse:{losses["rgb_mse"].item():.4f}, '
                #       f'smooth:{losses["smooth"].item():.4f}, normal_l1:{losses["normal_l1"].item():.4f}, normal_cos:{losses["normal_cos"].item():.4f}, '
                #       f'depth:{losses["depth"].item():.4f}, curvature:{losses["curvature"].item():.4f}, [psnr]: {psnr.item():.4f}, '
                #       f'[α/inv_s]: {alpha_inv_s:.4f}, [active_levels]: {self.model.module.sdf.active_levels}/{self.model.module.sdf.num_levels}')

                # log per log_freq step info
                if self.gpu == 0 and self.cur_step % self.conf.train.log_freq == 0:
                    for key, value in losses.items(): # log loss
                        self.loger.add_scalar(tag="loss" + '/' + key, scalar_value=value, global_step=self.cur_step)
                    self.loger.add_scalar(tag='scalar'+ '/psnr', scalar_value=psnr, global_step=self.cur_step)
                    if self.conf.model.type == 'volsdf':
                        self.loger.add_scalar(tag='scalar'+ '/alpha', scalar_value=1/self.model.module.density.get_beta(prog=progress), global_step=self.cur_step)
                    elif self.conf.model.type == 'neus':
                        self.loger.add_scalar(tag='scalar'+ '/inv_s', scalar_value=self.model.module.density.get_s(), global_step=self.cur_step)
                    if self.conf.model.object.sdf.enable_hashgrid:
                        self.loger.add_scalar(tag='scalar'+ '/active_levels', scalar_value=self.model.module.sdf.active_levels, global_step=self.cur_step)
                        self.loger.add_scalar(tag='scalar'+'/normal_epsilon',scalar_value=self.model.module.sdf.normal_epsilon,global_step=self.cur_step)
                        self.loger.add_scalar(tag='scalar' + '/lambda_curvature',scalar_value=self.loss.lambda_curvature, global_step=self.cur_step)
                    self.loger.add_scalar(tag='scalar'+'/epoch', scalar_value=epoch, global_step=self.cur_step)
                    self.loger.add_scalar(tag='scalar'+'lr', scalar_value=self.optimizer.param_groups[1]['lr'], global_step=self.cur_step)


                ##################################### Update end Step ##############################################
                self.scheduler.step() # scheduler step
                if self.dynamic_sampling: # dynamic sampling rays
                    self.set_num_rays(self.conf.train.max_num_rays, self.conf.train.num_samples_per_ray, output['num_samples'])
                    # set depth patch size
                    self.loss.set_patch_size(self.num_rays)
                if self.conf.model.nbfield.enabled and self.conf.dataset.use_mono_normal:
                    self.update_train_angle(output, sample)
                if self.cur_step == self.max_step:
                    break
                ####################################################################################################

                # end.record()
                # torch.cuda.synchronize()
                # print(f'[log time] {start.elapsed_time(end) / 1000:.4f}s', )

                # print(f'[step time] {time.time()-last_t:.4f}s',)
                # last_t = time.time()
            # print(f'[epoch time] {time.time()-epoch_st:.4f}s',)
            # save checkpoint
            self.save_checkpoint(epoch,save_epoch=self.gpu==0 and (epoch%self.conf.train.save_freq==0 or self.cur_step==self.max_step))
            # plot
            if epoch % self.conf.train.plot_freq == 0 or self.cur_step == self.max_step:
                self.plot(epoch)  # plot
            if self.cur_step == self.max_step:
                break


if __name__ == '__main__':
    opt = get_args()
    gpu = init_processes()

    ti=time.time()
    trainer = Trainer(opt, gpu)
    trainer.train()
    h,m,s=convert_seconds(time.time()-ti)
    print('successful!, total time: {}h {}m {}s'.format(h,m,s))

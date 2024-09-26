import time

import nerfacc.cuda
import numpy as np
import torch
from models.fields import *
from models.ray_sampler import *
from models.nerf_util import *
from models.loss import smooth_exp_confidence
from nerfacc import ContractionType, ray_marching, OccupancyGrid, render_weight_from_density, render_weight_from_alpha, accumulate_along_rays
class ImplicitReconSystem(torch.nn.Module):
    def __init__(self, conf, bound=None, device='cuda:0'):
        super().__init__()
        self.conf = conf
        self.type = self.conf.model.type # 'volsdf' or 'neus'
        self.bound = conf.model.bound if bound is None else bound # 场景边界, TODO: 区分hashgrid的边界和场景边界，暂时没必要。
        self.device = device
        self.bg_enabled = conf.model.background.enabled
        self.white_bg = conf.model.white_bg # 是否使用白色背景
        self.anneal_cos_end = 0.1
        self.use_mono_normal = self.conf.dataset.use_mono_normal

        density_conf = conf.model.density
        sdf_conf = conf.model.object.sdf
        rgb_conf = conf.model.object.rgb
        bg_conf = conf.model.background
        sampler_conf = conf.model.sampler
        occ_conf = conf.model.occupancy
        occ_bg_conf = conf.model.occupancy_bg
        nb_conf = conf.model.nbfield

        self.inside_outside = sdf_conf.inside_outside
        self.volsdf_type = self.type == 'volsdf' and self.inside_outside # volsdf、室内优化near surface eik，以及smooth。
        self.outside_sdf = 1000 # bound cube/spere外的sdf值
        self.occ_enabled = occ_conf.enabled
        self.occ_bg_enabled = occ_bg_conf.enabled
        self.bg_nerf_type = bg_conf.type
        self.nb_enabled = nb_conf.enabled
        self.rgb_enable_app = rgb_conf.enable_app
        self.bg_enable_app = bg_conf.enable_app and bg_conf.enabled
        self.nb_enable_app = nb_conf.mlp.enable_app and nb_conf.enabled

        # initialize models
        self.sdf = SDFNetwork(**sdf_conf,bound=self.bound)
        self.rgb = ColorNetwork(**rgb_conf)
        # background nerf
        if bg_conf.enabled:
            if bg_conf.type == 'nerf++': # nerf++
                self.bg_nerf = NeRFPlusPlus(**bg_conf.nerf_plus_plus, bound=self.bound,enable_app=bg_conf.enable_app,app_dim=bg_conf.app_dim)
            elif bg_conf.type == 'grid_nerf': # unbounded grid nerf
                self.bg_nerf = GridNeRF(**bg_conf.grid_nerf, bound=self.bound,enable_app=bg_conf.enable_app,app_dim=bg_conf.app_dim)
            if bg_conf.enable_app:
                self.app_bg = nn.Embedding(3000, bg_conf.app_dim)
                std=1e-4
                self.app_bg.weight.data.uniform_(-std, std)
        if rgb_conf.enable_app: # scene rgb启用appearance field
            self.app = nn.Embedding(3000, rgb_conf.app_dim)
            std=1e-4
            self.app.weight.data.uniform_(-std, std)

        # sampler and density
        if self.type == 'volsdf':
            self.density = LaplaceDensity(beta_init=density_conf.beta_init,beta_min=density_conf.beta_min, sched_conf=getattr(density_conf, 'sched', None))
            self.sampler = ErrorBoundSampler(**sampler_conf.error_bounded_sampler,scene_bounding_sphere=self.bound)
        elif self.type == 'neus':
            self.density = NeusDensity(density_conf.inv_s_init,density_conf.get('scale_factor',1.0))
            self.sampler = HierarchySampler(**sampler_conf.hierarchical_sampler, scene_bounding_sphere=self.bound)
            self.cos_anneal_ratio = 0

        # occupancy grid
        self.register_buffer('scene_aabb', torch.tensor([-self.bound, -self.bound, -self.bound, self.bound, self.bound, self.bound]).float())
        # self.scene_aabb = torch.tensor([-self.bound, -self.bound, -self.bound, self.bound, self.bound, self.bound]).float()
        if self.occ_enabled:
            self.occupancy = OccupancyGrid(roi_aabb=self.scene_aabb, resolution=occ_conf.resolution, contraction_type=ContractionType.AABB)
            # self.init_occ_grid(occ_conf.resolution)
            self.occ_res = occ_conf.resolution
            self.render_step_size = 2*np.sqrt(3)*self.bound/(occ_conf.resolution*2) # [CHANGE-3] render_step_size待调参。
            self.dynamic_render_step_size = self.render_step_size # FIXME：dynamic render step size higher quality but slower
            self.dynamic_step = getattr(occ_conf, 'dynamic_step', False) # if use dynamic render step size
            self.occ_prune_threshold = occ_conf.prune_threshold
        # occupancy_bg grid
        if self.occ_bg_enabled:
            self.occupancy_bg = OccupancyGrid(roi_aabb=self.scene_aabb, resolution=occ_bg_conf.resolution, contraction_type=ContractionType.UN_BOUNDED_SPHERE) # mip-nerf 360 无界场景压缩。
            self.render_bg_step_size = 0.01
            self.occ_bg_prune_threshold = occ_bg_conf.prune_threshold

        # NBfield
        self.anneal_quat_end = 1.0
        if self.nb_enabled:
            self.nb = NBField(**nb_conf.mlp)
            self.anneal_quat_end = getattr(conf.optim.sched, 'anneal_quat_end', 0.2)
            if self.nb.enable_app: # nbfield启用appearance field
                self.app_nb = nn.Embedding(3000, nb_conf.mlp.app_dim)
                std=1e-4
                self.app_nb.weight.data.uniform_(-std, std)

    def get_render_step_size(self):
        max_render_step_size = 2*np.sqrt(3)*self.bound/(self.occ_res/4)
        return min(self.dynamic_render_step_size,max_render_step_size) if self.dynamic_step else self.render_step_size

    def init_occ_grid(self, resolution):
        x = torch.linspace(-self.bound+self.bound/resolution, self.bound-self.bound/resolution, resolution)
        y = torch.linspace(-self.bound+self.bound/resolution, self.bound-self.bound/resolution, resolution)
        z = torch.linspace(-self.bound+self.bound/resolution, self.bound-self.bound/resolution, resolution)
        self.register_buffer('grid_pts', torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1).reshape(-1, 3))
        self.register_buffer('_binary', torch.ones([resolution]*3, dtype=torch.bool))
        self.register_buffer('occs', torch.zeros(resolution**3, dtype=torch.float32))
        # self.grid_pts = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1).reshape(-1, 3)
        # self._binary = torch.ones([resolution]*3, dtype=torch.bool)
    def forward(self, sample):
        # required: indices: (B ), rays_o: (B, R, 3), rays_d: (B, R, 3), depth_scale: (B, R, 1), pose: (B, 4, 4), pts(optional): (N_pts, 3)
        # if self.occ_enabled and self.training and sample['progress']>0.1: # warm-up
        if self.occ_enabled:  # 用nerf-acc实现的占用场(occupancy)和光线步进(ray marching)加速neus、volsdf体渲染。
            return self.forward_occ(sample)
        # start=torch.cuda.Event(enable_timing=True)
        # end=torch.cuda.Event(enable_timing=True)
        indices = sample['idx']
        rays_o, rays_d, depth_scale = sample['rays_o'], sample['rays_d'], sample['depth_scale']
        B, R = rays_o.shape[:2]
        rays_o, rays_d= rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

        output = {}
        # start.record()
        with torch.no_grad():
            # 这个z_near是每条ray只从z（表面附近）随机采样了一个点。
            z, z_near, near, far, outside = self.sampler.get_z_vals(ray_dirs=rays_d, cam_loc=rays_o, model=self, prog=sample.get('progress', 1.)) # z: (B*R,N), z_eik:(B*R,1), near:(B*R,1), far:(B*R,1), outside:(B*R,1)
        # end.record()
        # torch.cuda.synchronize()
        # print(f'[sampling time] {start.elapsed_time(end)/1000:.4f}s',)

        if outside.any():
            pass
            # print('there are rays outside the scene')
        rays_o, rays_d = rays_o.reshape(B, R, 3), rays_d.reshape(B, R, 3)

        # get sdf, rgbs
        z = z.reshape(B, R, -1, 1)
        far = far.reshape(B, R, 1, 1)
        outside = outside.reshape(B, R, 1)
        points = rays_o[:, :, None, :] + rays_d[:, :, None, :] * z
        sdf, feat, gradient, hessian= self.sdf.get_all(points, only_sdf=False, if_cal_hessian_x=self.sdf.gradient_mode=='numerical')
        sdf[outside[...,None].expand_as(sdf)] = self.outside_sdf
        app = None if not self.rgb_enable_app else self.app(indices)[:, None, None, :].tile(1, R, z.shape[2], 1)
        gradient_normalized = torch.nn.functional.normalize(gradient,dim=-1)
        rgbs = self.rgb(points, rays_d[:, :, None, :].tile(1, 1, z.shape[2], 1), gradient_normalized,feat, app) # (B, R, N, 3)
        if self.nb_enabled and self.use_mono_normal:
            app_nb = None if not self.nb_enable_app else self.app_nb(indices)[:, None, None, :].tile(1, R, z.shape[2], 1)  # FIXME：app_nb和color network app共享或者独立
            quats = self.nb(points, rays_d[:, :, None, :].tile(1, 1, z.shape[2], 1), gradient_normalized, feat, app_nb) # (B, R, N, 4)

        # get alphas
        if self.type =='volsdf':
            if_unbiased = getattr(self.conf.optim.sched, 'if_unbiased', False) # TODO：only unbias high biased angle area。
            if if_unbiased:
                confidence = torch.ones(B, R, z.shape[2], 1, device=z.device) # apply confidence to unbias
                if self.training and self.nb_enabled and self.use_mono_normal: # angle indicates bias confidence
                    angle = sample['train_angle']  # (B, R)
                    beta, threshold_angle = getattr(self.conf.optim.sched, 'unbiased_params', (25, 10))
                    threshold_angle = threshold_angle / 180 * torch.pi
                    confidence = smooth_exp_confidence(angle,beta=beta,threshold_angle=threshold_angle) # (B,R) biased_confidence：(25, 10)： [angle, confidence]有[0, 0.0.1], [5,0.1], [10,0.5], [15,0.9], [20, 0.98]
                    confidence = confidence[:, :, None, None].tile(1, 1, z.shape[2], 1)  # (B, R, N, 1)
                v = rays_d[:, :, None, :].tile(1, 1, z.shape[2], 1) # (B, R, N, 3)
                denom = torch.sum(gradient_normalized * v, dim=-1, keepdim=True).abs() + 1e-8 # (B, R, N, 1)
                # 1. confidence guiding unbias
                denom = denom * confidence + 1. * (1 - confidence)
                def two_steps_warm_up(prog):
                    two_steps = [self.anneal_quat_end, min(self.anneal_quat_end+0.3, 1)]  # two_steps[0]-two_steps[1]: 0-1
                    if prog < two_steps[0]:
                        return 0
                    elif prog < two_steps[1]:
                        return (prog - two_steps[0]) / (two_steps[1] - two_steps[0])
                    else:
                        return 1
                def sequence_warm_up(prog):
                    # progress: 0-0.2 0->1
                    anneal_unbiased_end = 0.2
                    if prog < anneal_unbiased_end:
                        return prog / anneal_unbiased_end
                    return 1.
                # 2. 训练sequence warm up unbias
                prog = two_steps_warm_up(sample.get('progress', 1.))
                denom = denom * prog + 1. * (1 - prog)
                # origin_densities = self.density(sdf)
                densities = self.density(sdf / torch.abs(denom))
            else:
                densities = self.density(sdf)
                # origin_densities = densities
            alphas = volume_rendering_alphas(densities=densities,dists=z) # (B, R, N)
        elif self.type =='neus':
            progress = sample['progress'] if self.training else 1.
            alphas = self.get_neus_alphas(rays_d, sdf, gradient_normalized, dists=z, dist_far=far, progress=progress)
        else:
            raise NotImplementedError

        # compositing using weights
        weights=alpha_compositing_weights(alphas) # (B, R, N, 1)
        rgb = composite(rgbs, weights)
        if self.nb_enabled and self.use_mono_normal:  # TODO: nbfield
            quat = composite(quats, weights) # (B, R, 4)
            quat = normalize_quat(quat)
        opacity = composite(1, weights) # (B, R, 1)
        normal = composite(gradient_normalized, weights)
        normal = torch.nn.functional.normalize(normal, p=2, dim=-1)
        output['normal_w'] = normal # scene normal in world space
        if self.training and self.nb_enabled and self.use_mono_normal: # TODO: nbfield, normal biased field, to fit in the normal prior.
            half_theta = torch.acos(quat[...,:1])
            n = quat[...,1:]/(torch.sin(half_theta)+1e-6)
            def sequence_warm_up(prog):
                # progress: 0-0.2 0->1
                anneal_quat_end = self.anneal_quat_end
                if prog < anneal_quat_end:
                    return prog/anneal_quat_end
                return 1.
            prog = sequence_warm_up(sample['progress'])
            iter_theta = half_theta * prog
            iter_n = torch.nn.functional.normalize(n*prog+normal*(1-prog),p=2,dim=-1)
            iter_quat = torch.cat([torch.cos(iter_theta),torch.sin(iter_theta)*iter_n],dim=-1)
            biased_normal = quat_mult(quat_mult(iter_quat,to_pure_quat(normal)),quat_conj(iter_quat))[...,1:]
            output['biased_normal_w'] = biased_normal
            with torch.no_grad():
                # 1. normal_w and biased normal_w
                output['angle'] = torch.acos(torch.sum(normal * biased_normal, dim=-1, keepdim=True).clip(-1, 1))
                # 2. normal_w and gt_normal_w
                # R_c2w = sample['pose'][:,:3,:3] # c2w
                # mono_normal = sample['normal'] # (B,R,3) in camera space
                # mono_normal = torch.bmm(R_c2w,mono_normal.permute(0,2,1)).permute(0,2,1) # (B,R,3) in world space
                # output['angle'] = torch.acos(torch.sum(normal*mono_normal,dim=-1,keepdim=True).clip(-1,1))
            normal = biased_normal
        elif not self.training and self.nb_enabled and self.use_mono_normal: # eval to visualize, camera space
            R_c2w = sample['pose'][:,:3,:3] # c2w
            R_w2c = R_c2w.permute(0,2,1) # w2c
            biased_normal = quat_mult(quat_mult(quat,to_pure_quat(normal)),quat_conj(quat))[...,1:] # bias normal
            biased_normal = torch.bmm(R_w2c,biased_normal.permute(0,2,1)).permute(0,2,1) # transform biased normal to C
            mono_normal = sample['normal'] # (B,R,3) in camera space
            mono_normal = torch.bmm(R_c2w,mono_normal.permute(0,2,1)).permute(0,2,1) # (B,R,3) in world space
            biased_mono_normal = quat_mult(quat_mult(quat_conj(quat),to_pure_quat(mono_normal)),quat)[...,1:] # bias mono normal to the scene normal
            biased_mono_normal = torch.bmm(R_w2c,biased_mono_normal.permute(0,2,1)).permute(0,2,1) # bias mono normal to camera space
            output['biased_normal'] = biased_normal
            output['biased_mono_normal'] = biased_mono_normal
        dist = composite(z, weights) # dist from camera origin
        depth=dist*depth_scale # depth


        # normal 世界坐标系->相机坐标系
        R_w2c=torch.transpose(sample['pose'][:,:3,:3],1,2)
        normal=torch.bmm(R_w2c,torch.transpose(normal,1,2))
        normal = torch.transpose(normal,1,2)

        # output
        output['outside'] = outside
        output['rgb'] = rgb
        if self.nb_enabled and self.use_mono_normal:
            output['quat'] = quat
            # output['quats'] = quats
        output['depth'] = depth
        output['opacity'] = opacity
        output['normal'] = normal
        output['sdf'] = sdf
        output['gradient'] = gradient
        output['hessian'] = hessian
        output['num_samples'] = R*z.shape[2]

        # record_density_weights = False
        # if record_density_weights:
        #     output['densities'] = densities[...,0]
        #     output['origin_densities'] = origin_densities[...,0]
        #     output['weights'] = weights[...,0]
        #     origin_alphas = volume_rendering_alphas(densities=origin_densities, dists=z)  # (B, R, N)
        #     origin_weights = alpha_compositing_weights(origin_alphas)  # (B, R, N, 1)
        #     output['origin_weights'] = origin_weights[...,0]
        #     output['distance'] = z[...,0]


        # background nerf rendering
        if self.bg_enabled:
            if self.occ_bg_enabled:
                output_bg = self.forward_bg_occ(sample)
            else:
                output_bg = self.forward_bg(sample, far=far)
            # update
            output['rgb']=rgb + (1-opacity) * output_bg['rgb']
            output['depth_bg'] = output_bg['depth']
            output['opacity_bg'] = output_bg['opacity']
            output['num_samples'] += output_bg['num_samples']

        # sample eikonal points、curvature points
        if self.training:
            volsdf_type=self.inside_outside and self.type=='volsdf'
            if self.volsdf_type:
                if self.sdf.enable_hashgrid:
                    z_near=z_near.reshape(B,R,-1,1)
                    N_near = z_near.shape[2]
                    points_near = rays_o[:, :, None, :] + rays_d[:, :, None, :] * z_near # (B, R, N_near, 3)
                    with torch.no_grad():
                        if self.sampler.intersection_type == 'cube':
                            points_uniform=sample_points_in_cube(side_length=2.*self.bound, shape=z_near.shape[:-2]+(1,),device=z.device)
                        elif self.sampler.intersection_type == 'sphere':
                            points_uniform=sample_points_in_sphere(radius=self.bound, shape=z_near.shape[:-2]+(1,),device=z.device)
                        else:
                            raise NotImplementedError
                    points_eik=torch.cat([points_near,points_uniform],dim=-2)  # (B, R, N_near+N_uniform, 3)
                    _, _, gradient_eik, hessian_eik = self.sdf.get_all(points_eik, only_sdf=True, if_cal_hessian_x=True)
                    with torch.no_grad():  # TODO：法平面采样neighbor points消融实验
                        # 1. 拓展：基于gradient_eik在切点切平面上采样neighbor points，以计算曲率
                        points_eik_neighbor = sample_neighbours_near_plane(gradient_eik, points_eik, device=z.device)
                        # 2. 在附近随机采样
                        # points_eik_neighbor = points_eik + (torch.rand_like(points_eik) - 0.5) * 0.01
                    _, _, gradient_eik_neighbor, _ = self.sdf.get_all(points_eik_neighbor, only_sdf=True, if_cal_hessian_x=False)

                    # FIXME：对所有点做eik和hessian。
                    # output['gradient_eik'] = gradient_eik
                    output['gradient_eik'] = torch.cat([gradient,gradient_eik[:,:,N_near:,:]],dim=-2) # eik：all points + uniform points
                    output['gradient_smooth'] = gradient_eik # smooth： near points + uniform points
                    output['gradient_smooth_neighbor'] = gradient_eik_neighbor

                    # hessian
                    # _,_,_,hessian_near=self.sdf.get_all(points_near,only_sdf=True,if_cal_hessian_x=True)
                    output['hessian']=hessian_eik[:,:,:N_near,:] # (B, R, N_near, 3)
                    # output['hessian'] = hessian # (B, R, N, 3)
                else: # monosdf type random neighbor points
                    z_near = z_near.reshape(B, R, -1, 1) # (B, R, N_near, 1)
                    N_near = z_near.shape[2]
                    points_near = rays_o[:, :, None, :] + rays_d[:, :, None, :] * z_near  # (B, R, N_near, 3)
                    with torch.no_grad():
                        if self.sampler.intersection_type == 'cube':
                            points_uniform = sample_points_in_cube(side_length=2. * self.bound, shape=z_near.shape[:-2] + (1,), device=z.device)
                        elif self.sampler.intersection_type == 'sphere':
                            points_uniform = sample_points_in_sphere(radius=self.bound, shape=z_near.shape[:-2] + (1,), device=z.device)
                        else:
                            raise NotImplementedError
                    points_eik = torch.cat([points_near, points_uniform], dim=-2)  # (B, R, N_near+N_uniform, 3)
                    # 在附近随机采样
                    points_eik_neighbor = points_eik + (torch.rand_like(points_eik) - 0.5) * 0.01
                    points_both = torch.cat([points_eik, points_eik_neighbor], dim=-2) # (B, R, (N_near+N_uniform)*2, 3)
                    _, _, gradient_both, _ = self.sdf.get_all(points_both, only_sdf=True,if_cal_hessian_x=False)

                    output['gradient_eik'] = gradient_both[:,:,:gradient_both.shape[2]//2,:]
                    output['gradient_smooth'] = gradient_both[:,:,:gradient_both.shape[2]//2,:]  # smooth： near points + uniform points
                    output['gradient_smooth_neighbor'] = gradient_both[:,:,gradient_both.shape[2]//2:,:]


                # output['hessian'] = hessian
            else: # TODO：try neus sample uniform points and near points[?]
                output['gradient_eik'] = gradient
                output['gradient_smooth'] = None
                output['gradient_smooth_neighbor'] = None
                output['hessian'] = hessian
            
            # TODO: sparse or dense points cloud of the scene, dense points can be deep-mvs(vis-mvs) prior. 参考instant-angelo
            if sample.get('pts', None) is not None:
                pts_sdf, _, pts_gradient, _ = self.sdf.get_all(sample['pts'], only_sdf=True, if_cal_hessian_x=False)
                output['pts_sdf'] = pts_sdf
                output['pts_gradient'] = pts_gradient
        # end.record()
        # torch.cuda.synchronize()
        # print(f'[compositing time] {start.elapsed_time(end)/1000:.4f}s',)

        return output

    def forward_occ(self, sample):
        # occ enabled，基于nerf-acc的rendering加速
        # indices: (B ), rays_o: (B, R, 3), rays_d: (B, R, 3), depth_scale: (B, R, 1)
        indices, rays_o, rays_d, depth_scale = sample['idx'], sample['rays_o'], sample['rays_d'], sample['depth_scale']
        output={}
        B, R = rays_o.shape[:2]
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3) # (B*R,3), (B*R,3), (B*R,1)
        with torch.no_grad():
            # 1. ray-marching using occupancy
            ray_indices, t_starts, t_ends = ray_marching(
                rays_o, rays_d,
                scene_aabb=self.scene_aabb,
                grid=self.occupancy,
                render_step_size=self.get_render_step_size(),
                stratified=self.training
            )  # ray_indices: (N_samples,), t_starts: (N_samples,1), t_ends: (N_samples,1).   ray_indices∈[0,B*R)
            # # TODO：using custom _binary
            # packed_info, ray_indices, t_starts, t_ends = nerfacc.cuda.ray_marching(
            #     rays_o.contiguous(), rays_d.contiguous(),
            #     torch.zeros(B*R,dtype=torch.float32,device=rays_o.device).contiguous(), # near
            #     2*np.sqrt(3)*self.bound*torch.ones(B*R,dtype=torch.float32,device=rays_o.device).contiguous(), # far
            #     self.scene_aabb.contiguous(),
            #     self._binary,
            #     self.occupancy.contraction_type.to_cpp_version(),
            #     self.render_step_size,
            #     0.0,
            # )  # ray_indices: (N_samples,), t_starts: (N_samples,1), t_ends: (N_samples,1).   ray_indices∈[0,B*R)
        # print(f'Num of occ samples: {ray_indices.shape[0]}')
        if ray_indices.shape[0] == 0: # all rays are empty
            output['outside'] = torch.ones(B, R, 1,dtype=torch.bool, device=rays_o.device)
            output['rgb'] = torch.zeros(B, R, 3, device=rays_o.device)
            output['depth'] = torch.zeros(B, R, 1, device=rays_o.device)
            output['opacity'] = torch.zeros(B, R, 1, device=rays_o.device)
            output['rays_fg'] = torch.zeros(B, R, 1, dtype=torch.bool, device=rays_o.device)
            output['normal'] = torch.zeros(B, R, 3, device=rays_o.device)
            output['num_samples'] = 0
            if self.nb_enabled and self.use_mono_normal:
                output['quat'] = torch.zeros(B, R, 4, device=rays_o.device)
                output['normal_w'] = torch.zeros(B, R, 3, device=rays_o.device)
                output['biased_normal_w'] = torch.zeros(B, R, 3, device=rays_o.device)
                if not self.training:
                    output['biased_normal'] = torch.zeros(B, R, 3, device=rays_o.device)
                    output['biased_mono_normal'] = torch.zeros(B, R, 3, device=rays_o.device)
            return output
        # get all sample points from ray-marching occ, and get sdf、rgbs
        t_mid = (t_starts + t_ends) / 2.0
        sample_rays_o, sample_rays_d = rays_o[ray_indices], rays_d[ray_indices]
        points = sample_rays_o + t_mid * sample_rays_d
        sdf, feat, gradient, hessian = self.sdf.get_all(points, only_sdf=False, if_cal_hessian_x=not self.volsdf_type)
        gradient_normalized = torch.nn.functional.normalize(gradient,dim=-1)
        app = None if not self.rgb_enable_app else torch.index_select(self.app(indices),0,ray_indices.view(-1)//R)
        rgbs = self.rgb(points, sample_rays_d, gradient_normalized, feat, app)
        if self.nb_enabled and self.use_mono_normal: # TODO: nbfield
            app_nb = None if not self.nb_enable_app else torch.index_select(self.app_nb(indices),0,ray_indices.view(-1)//R)
            quats = self.nb(points, sample_rays_d, gradient_normalized, feat, app_nb)

        # get alphas
        if self.type =='volsdf':
            densities = self.density(sdf)
            alpha = 1.-(-(t_ends-t_starts)*densities).exp_()

        elif self.type =='neus':
            progress = sample['progress'] if self.training else 1.
            alpha = self.get_neus_alphas_occ(sdf, gradient_normalized, sample_rays_d, t_ends - t_starts, progress=progress)
        else:
            raise NotImplementedError
        weights = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=B * R)
        # composite using nerf-acc
        rgb = accumulate_along_rays(weights, ray_indices=ray_indices, values=rgbs, n_rays=B*R).reshape(B, R, 3)
        if self.nb_enabled and self.use_mono_normal: # TODO: nbfield
            quat = accumulate_along_rays(weights, ray_indices=ray_indices, values=quats, n_rays=B*R).reshape(B, R, 4)
            quat = normalize_quat(quat)
        opacity = accumulate_along_rays(weights, ray_indices=ray_indices, values=None, n_rays=B*R).reshape(B, R, 1)
        normal = accumulate_along_rays(weights, ray_indices=ray_indices, values=gradient_normalized, n_rays=B*R).reshape(B, R, 3)
        dist = accumulate_along_rays(weights, ray_indices=ray_indices, values=t_mid, n_rays=B*R).reshape(B, R, 1)
        rays_fg = opacity > 0.1  # scene_aabb内的前景点。
        depth = dist * depth_scale
        normal = torch.nn.functional.normalize(normal, p=2, dim=-1)
        output['normal_w'] = normal # scene normal in world space
        if self.training and self.nb_enabled and self.use_mono_normal: # TODO: nbfield, normal biased field, to fit in the normal prior.
            half_theta = torch.acos(quat[...,:1])
            n = quat[...,1:]/(torch.sin(half_theta)+1e-6)
            def sequence_warm_up(prog):
                # progress: 0-0.2 0->1
                anneal_quat_end = self.anneal_quat_end
                if prog < anneal_quat_end:
                    return prog/anneal_quat_end
                return 1.
            prog = sequence_warm_up(sample['progress'])
            iter_theta = half_theta * prog
            iter_n = torch.nn.functional.normalize(n*prog+normal*(1-prog),p=2,dim=-1) # FIXME：这里anneal blend n&normal正常来讲应该detach normal，但detach后前期训练不稳定。先将就。
            iter_quat = torch.cat([torch.cos(iter_theta),torch.sin(iter_theta)*iter_n],dim=-1)
            biased_normal = quat_mult(quat_mult(iter_quat,to_pure_quat(normal)),quat_conj(iter_quat))[...,1:]
            output['biased_normal_w'] = biased_normal
            with torch.no_grad():
                output['angle'] = torch.acos(torch.sum(normal*biased_normal,dim=-1,keepdim=True).clip(-1,1))
            normal = biased_normal
        elif not self.training and self.nb_enabled and self.use_mono_normal: # eval to visualize, camera space
            R_c2w = sample['pose'][:,:3,:3] # c2w
            R_w2c = R_c2w.permute(0,2,1) # w2c
            biased_normal = quat_mult(quat_mult(quat,to_pure_quat(normal)),quat_conj(quat))[...,1:] # biased normal in W
            biased_normal = torch.bmm(R_w2c,biased_normal.permute(0,2,1)).permute(0,2,1) # biased normal in C
            mono_normal = sample['normal'] # (B,R,3) in camera space
            mono_normal = torch.bmm(R_c2w,mono_normal.permute(0,2,1)).permute(0,2,1) # (B,R,3) in world space
            biased_mono_normal = quat_mult(quat_mult(quat_conj(quat),to_pure_quat(mono_normal)),quat)[...,1:] # bias mono normal to the scene normal
            biased_mono_normal = torch.bmm(R_w2c,biased_mono_normal.permute(0,2,1)).permute(0,2,1) # bias mono normal to camera space
            output['biased_normal'] = biased_normal
            output['biased_mono_normal'] = biased_mono_normal
        outside = opacity == 0 # nerf-acc outside opacity=0
        if outside.any():
            pass
            # print('there are rays outside the scene')

        # normal world->camera
        R_w2c = torch.transpose(sample['pose'][:, :3, :3], 1, 2)
        normal = torch.bmm(R_w2c, torch.transpose(normal, 1, 2))
        normal = torch.transpose(normal, 1, 2)

        # output
        output['outside'] = outside
        output['rgb'] = rgb
        output['gradient'] = gradient
        if self.nb_enabled and self.use_mono_normal:
            output['quat'] = quat
            # output['quats'] = quats
        output['rays_fg'] = rays_fg
        output['depth'] = depth
        output['opacity'] = opacity
        output['normal'] = normal
        output['num_samples'] = ray_indices.shape[0]//B # per-image samples

        # bg nerf rendering
        if self.bg_enabled:
            if self.occ_bg_enabled:
                output_bg = self.forward_bg_occ(sample)
            else:
                output_bg = self.forward_bg(sample, far=None)
            output['rgb'] = rgb + (1 - opacity) * output_bg['rgb']
            output['depth_bg'] = output_bg['depth']
            output['opacity_bg'] = output_bg['opacity']
            output['num_samples'] += output_bg['num_samples']

        # add eikonal points, smooth points, curvature points, etc.
        if self.training:
            volsdf_type=self.inside_outside and self.type=='volsdf'
            if self.volsdf_type:
                # eikonal points
                # TODO: 当是室内场景volsdf type启用occ时，考虑near点怎么取(随机取B*R个点算了)。
                mask_fg = rays_fg.reshape(-1)[ray_indices]
                rand_idx_near = torch.randint(mask_fg.sum(), size=(B,R,1), device=mask_fg.device)
                points_near = points[mask_fg][rand_idx_near] # (B,R,1,3)
                gradient_near = gradient[mask_fg][rand_idx_near] # (B,R,1,3)
                # uniform points
                with torch.no_grad():
                    if self.occupancy.contraction_type == ContractionType.AABB:
                        points_uniform = sample_points_in_cube(side_length=2. * self.bound, shape=(B,R,1),device=mask_fg.device)
                    elif self.occupancy.contraction_type == ContractionType.UN_BOUNDED_SPHERE:
                        points_uniform = sample_points_in_sphere(radius=self.bound, shape=(B,R,1),device=mask_fg.device)
                    else:
                        raise NotImplementedError
                _, _, gradient_uniform, _ = self.sdf.get_all(points_uniform, only_sdf=True, if_cal_hessian_x=False)
                points_eik, gradient_eik = torch.cat([points_near, points_uniform], dim=-2), torch.cat([gradient_near, gradient_uniform], dim=-2)
                with torch.no_grad():
                    points_eik_neighbor = sample_neighbours_near_plane(gradient_eik, points_eik, device=mask_fg.device)
                    # points_eik_neighbor = points_eik + (torch.rand_like(points_eik) - 0.5) * 0.01

                _, _, gradient_eik_neighbor, _ = self.sdf.get_all(points_eik_neighbor, only_sdf=True, if_cal_hessian_x=False)

                # FIXME：对所有点做eik(+uniform点)和hessian。
                # output['gradient_eik'] = gradient_eik
                output['gradient_eik'] = torch.cat([gradient[mask_fg], gradient_eik[:,:,1:,:].reshape(-1,3)], dim=0)
                output['gradient_smooth'] = gradient_eik # (B,R,2,3)
                output['gradient_smooth_neighbor'] = gradient_eik_neighbor
                # output['hessian'] = hessian[mask_fg]
                output['hessian'] = hessian[mask_fg][rand_idx_near] # (B,R,1,3)
            else:
                # using rays_fg to filter points
                mask_fg = rays_fg.reshape(-1)[ray_indices]
                output['gradient_eik'] = gradient[mask_fg] # (N_fg,3)
                output['gradient_smooth'] = None
                output['gradient_smooth_neighbor'] = None
                output['hessian'] = hessian[mask_fg]

                # output['gradient_eik'] = gradient
                # output['gradient_smooth'] = None
                # output['gradient_smooth_neighbor'] = None
                # output['hessian'] = hessian
            
            # TODO: sparse or dense points cloud of the scene
            if sample.get('pts', None) is not None:
                pts_sdf, _, pts_gradient, _ = self.sdf.get_all(sample['pts'], only_sdf=True, if_cal_hessian_x=False)
                output['pts_sdf'] = pts_sdf
                output['pts_gradient'] = pts_gradient
        if output['rgb'].isnan().any():
            raise ValueError('rgb has nan value')
        return output


    def forward_bg(self, sample, far=None):
        # background nerf rendering，vanilla nerf, using nerf++ contraction即逆球面重参数化。
        rays_o, rays_d, depth_scale, indices = sample['rays_o'], sample['rays_d'], sample['depth_scale'], sample['idx'] # rays_o: (B, R, 3), rays_d: (B, R, 3), indices: (B)
        B, R = rays_o.shape[:2]
        output_bg = {}

        # inverse sampling
        with torch.no_grad():
            inverse_r = sample_dists(ray_size=(B, R), dist_range=(1, 0), intvs=self.sampler.N_samples_bg,stratified=self.training)  # (B, R, N_samples_bg, 1)
        if far is None:
            _, far, outside = near_far_from_cube(rays_o, rays_d, bound=self.bound)
            far[outside] = self.bound
            far = far.reshape(B, R, 1, 1)
        z_bg = far / (inverse_r + 1e-6)  # (B, R, N_samples_bg, 1)
        points_bg = rays_o[:, :, None, :] + rays_d[:, :, None, :] * z_bg  # (B, R, N_samples_bg, 3)
        if self.bg_nerf_type == 'grid_nerf':
            # TODO：Mip-NeRF 360, 无界场景压缩(场景点保持在半径self.bound球内，无界背景点压缩到unbounded_r内) nerf-studio: https://docs.nerf.studio/nerfology/model_components/visualize_spatial_distortions.html
            def un_bounded_sphere_contraction(points, unbounded_r=None):
                # points: (..., 3)
                # un_bounded_r: 无界半径
                if unbounded_r is None:
                    unbounded_r = 2 * self.bound
                points_norm = points.norm(dim=-1)
                points[points_norm > self.bound] = (unbounded_r - self.bound / (points_norm[points_norm > 1.][...,None] + 1e-6)) * points[points_norm > 1.] / (points_norm[points_norm > 1.][...,None] + 1e-6) # [-un_bounded_r,un_bounded_r]^3
                return points/unbounded_r*self.bound # [-bound, bound]^3
            points_bg = un_bounded_sphere_contraction(points_bg)

        app_bg = None if not self.bg_enable_app else self.app_bg(indices)[:, None, None, :].tile(1, R, z_bg.shape[2], 1)
        densities_bg, rgbs_bg = self.bg_nerf(points_bg, rays_d[:, :, None, :].tile(1, 1, self.sampler.N_samples_bg, 1), rays_o[:, :, None, :].tile(1, 1, self.sampler.N_samples_bg, 1), app_bg)
        alphas_bg = volume_rendering_alphas(densities=densities_bg, dists=z_bg)  # 不透明度, 为了方便合并object和bg的结果, 我们使用基于alpha、T的离散体渲染。
        weights_bg = alpha_compositing_weights(alphas_bg)
        rgb_bg = composite(rgbs_bg, weights_bg)
        opacity_bg = composite(1, weights_bg)
        dist_bg = composite(z_bg, weights_bg)
        depth_bg = dist_bg * depth_scale
        if self.white_bg:
            rgb_bg = rgb_bg + (1 - opacity_bg)

        output_bg['rgb'] = rgb_bg
        output_bg['depth'] = depth_bg
        output_bg['opacity'] = opacity_bg
        output_bg['num_samples'] = R * self.sampler.N_samples_bg
        return output_bg

    def forward_bg_occ(self, sample):
        # 无边界背景也能用occ加速,压缩方式为mip-nerf 360，所有点压缩到半径为2的球内。
        indices, rays_o, rays_d, depth_scale = sample['idx'], sample['rays_o'], sample['rays_d'], sample['depth_scale']
        output={}
        B, R = rays_o.shape[:2]
        rays_o, rays_d, depth_scale = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3), depth_scale.reshape(-1, 1) # (B*R,3), (B*R,3), (B*R,1)
        ray_indices, t_starts, t_ends = ray_marching(
            rays_o, rays_d,
            scene_aabb=self.scene_aabb,
            grid=self.occupancy_bg,
            render_step_size=self.render_bg_step_size,
            near_plane=0.05, far_plane=1e3,
            stratified=self.training
        ) # ray_indices: (N_samples,), t_starts: (N_samples,1), t_ends: (N_samples,1).   ray_indices∈[0,B*R)
        if ray_indices.shape[0] == 0:
            output['rgb'] = torch.zeros(B, R, 3, device=rays_o.device)
            output['depth'] = torch.zeros(B, R, 1, device=rays_o.device)
            output['opacity'] = torch.zeros(B, R, 1, device=rays_o.device)
            output['num_samples'] = 0
            return output
        # get all sample points from ray-marching occ background, and get nerf densities、rgbs
        t_mid = (t_starts + t_ends) / 2.0
        sample_rays_o, sample_rays_d = rays_o[ray_indices], rays_d[ray_indices]
        points = sample_rays_o + t_mid * sample_rays_d # (N_samples,3), ∈[-bound,bound]^3
        app_bg = None if not self.bg_enable_app else torch.index_select(self.app_bg(indices),0,ray_indices.view(-1)//R)
        densities, rgbs = self.bg_nerf(points, sample_rays_d, sample_rays_o, app_bg)
        weights = render_weight_from_density(t_starts,t_ends,densities,ray_indices=ray_indices,n_rays=B*R)

        # composite bg using nerf-acc
        rgb = accumulate_along_rays(weights, ray_indices=ray_indices, values=rgbs, n_rays=B*R).reshape(B, R, 3)
        opacity = accumulate_along_rays(weights, ray_indices=ray_indices, values=None, n_rays=B*R).reshape(B, R, 1)
        dist = accumulate_along_rays(weights, ray_indices=ray_indices, values=t_mid, n_rays=B*R).reshape(B, R, 1)
        depth = dist * depth_scale

        if self.white_bg:
            rgb = rgb + (1 - opacity)
        # output
        output['rgb'] = rgb
        output['depth'] = depth
        output['opacity'] = opacity
        output['num_samples'] = ray_indices.shape[0]//B
        return output


    def get_neus_alphas_occ(self, sdf, gradient, ray_dirs, delta, progress=1., eps=1e-5):
        # sdf: (N,1), gradient: (N,3), ray_dirs: (N,3), delta: (N,1)
        inv_s = self.density.get_s()
        true_cos = torch.sum(ray_dirs * gradient, dim=-1, keepdim=True)  # [N,1]
        iter_cos = self._get_iter_cos(true_cos, progress=progress)  # [N,1]
        est_prev_sdf = sdf - iter_cos * delta * 0.5  # [N,1]
        est_next_sdf = sdf + iter_cos * delta * 0.5  # [N,1]
        prev_cdf = (est_prev_sdf * inv_s).sigmoid()  # [N,1]
        next_cdf = (est_next_sdf * inv_s).sigmoid()  # [N,1]
        alphas = ((prev_cdf - next_cdf + eps) / (prev_cdf + eps)).clamp(0.0, 1.0)  # [N,1]
        return alphas

    def get_neus_alphas(self, rays_d, sdf, gradient, dists, dist_far=None, progress=1., eps=1e-5):
        # rays_d: [B,R,3], sdf: [B,R,N,1], gradient: [B,R,N,3], dists: [B,R,N,1], dist_far: [B,R,1,1]
        sdf = sdf[..., 0]  # [B,R,N]
        # SDF volume rendering in NeuS.
        inv_s = self.density.get_s()
        true_cos = (rays_d[..., None, :] * gradient).sum(dim=-1, keepdim=False)  # [B,R,N]
        iter_cos = self._get_iter_cos(true_cos, progress=progress)  # [B,R,N]
        # Estimate signed distances at section points
        if dist_far is None:
            dist_far = torch.empty_like(dists[..., :1, :]).fill_(1e10)  # [B,R,1,1]
        dists = torch.cat([dists, dist_far], dim=2)  # [B,R,N+1,1]
        dist_intvs = dists[..., 1:, 0] - dists[..., :-1, 0]  # [B,R,N]
        est_prev_sdf = sdf - iter_cos * dist_intvs * 0.5  # [B,R,N]
        est_next_sdf = sdf + iter_cos * dist_intvs * 0.5  # [B,R,N]
        prev_cdf = (est_prev_sdf * inv_s).sigmoid()  # [B,R,N]
        next_cdf = (est_next_sdf * inv_s).sigmoid()  # [B,R,N]
        alphas = ((prev_cdf - next_cdf+eps) / (prev_cdf + eps)).clip_(0.0, 1.0)  # [B,R,N]
        # weights = render.alpha_compositing_weights(alphas)  # [B,R,N,1]
        return alphas

    def _get_iter_cos(self, true_cos, progress):
        anneal_ratio = min(progress / self.anneal_cos_end, 1.)
        # The anneal strategy below keeps the cos value alive at the beginning of training iterations.
        return -((-true_cos * 0.5 + 0.5).relu() * (1.0 - anneal_ratio) + (-true_cos).relu() * anneal_ratio)  # always non-positive
    
    def get_occ_eval_fn(self):
        # return occ_eval_fn according to the sdf model type, volsdf or neus
        with torch.no_grad():
            if self.type == 'volsdf':
                def volsdf_occ_eval_fn(x): # return 1-exp(-σ*δ)
                    # x: (N,3)
                    sdf = self.sdf.get_sdf(x) # (N,1)
                    densities = self.density(sdf)
                    return 1. - (-densities * self.get_render_step_size()).exp_()
                return volsdf_occ_eval_fn
            elif self.type == 'neus':
                def neus_occ_eval_fn(x): # return alpha
                    # x:(N,3)
                    # (pre_cdf-af_cdf+1e-5)/(pre_cdf+1e-5)
                    inv_s = self.density.get_s()
                    sdf = self.sdf.get_sdf(x)
                    voxel_size = 2*self.bound/self.occ_res
                    sdf = torch.maximum(sdf-voxel_size*(3**0.5)/2, torch.zeros_like(sdf)) # consider all possible occupied voxels
                    pre_sdf, af_sdf = sdf + 0.5 * self.get_render_step_size(), sdf - 0.5 * self.get_render_step_size()
                    pre_cdf, af_cdf = (pre_sdf*inv_s).sigmoid(), (af_sdf*inv_s).sigmoid()
                    alpha = (pre_cdf-af_cdf+1e-5)/(pre_cdf+1e-5).clip_(0.0,1.0)
                    return alpha
                return neus_occ_eval_fn
            else:
                raise NotImplementedError

    def get_occ_bg_eval_fn(self):
        # return occ_eval_fn according to the sdf model type, volsdf or neus
        with torch.no_grad():
            def bg_occ_eval_fn(x):
                # x: (N,3)
                densities = self.bg_nerf.get_density(x, with_feat=False)
                return 1. - (-densities * self.render_bg_step_size).exp_()
            return bg_occ_eval_fn
    def update_occ(self, cur_step):
        with torch.no_grad():
            if self.occ_enabled:
                self.dynamic_render_step_size = 2*np.sqrt(3)*self.bound/(self.density.get_s() if self.type == 'neus' else torch.abs(1/self.density.get_beta()))
                self.occupancy.every_n_step(step=cur_step,occ_eval_fn=self.get_occ_eval_fn(),occ_thre=self.occ_prune_threshold)
                # self.update_binary_grid(cur_step)
            if self.occ_bg_enabled and self.bg_enabled:
                self.occupancy_bg.every_n_step(step=cur_step,occ_eval_fn=self.get_occ_bg_eval_fn(),occ_thre=self.occ_bg_prune_threshold)

    def update_binary_grid(self, cur_step, chunk=100000, decay=0.95):
        if cur_step % 16 != 0:
            return
        mask = self._binary.reshape(-1)
        pts = self.grid_pts[mask]
        occ_eval_fn = self.get_occ_eval_fn()
        occs = [] # occ即alpha
        for i in range(0, len(pts), chunk):
            occs.append(occ_eval_fn(pts[i:i + chunk]))
        occs = torch.cat(occs,dim=0).reshape(-1)
        self.occs[mask] = torch.maximum(self.occs[mask] * decay, occs)
        mask[mask.clone()] = self.occs[mask] > torch.clamp(self.occs[mask].mean(), max=self.occ_prune_threshold)
        self._binary = mask.reshape(self._binary.shape).contiguous()

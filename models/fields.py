# fields including SDF, Radiance, etc.
import os
import time

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import sys

sys.path.append(os.getcwd())
from models.encoder import *
# from models.hash_encoder.hash_encoder import HashEncoder # my hashgrid
# from models.hashencoder.hashgrid import HashEncoder
from models.nerf_util import MLPwithSkipConnection
import tinycudann as tcnn

# 截断的指数激活函数, 可用于background density的激活函数。
class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))

trunc_exp = _TruncExp.apply

# volsdf, sdf->density的laplace变换
class LaplaceDensity(nn.Module):
    def __init__(self, beta_init=0.1, beta_min=1e-4, sched_conf=None):
        super().__init__()
        self.beta_min = torch.tensor(beta_min)
        self.sched_conf = sched_conf
        self.sched_enabled = sched_conf is not None and getattr(sched_conf, 'enabled', False)
        requires_grad = not self.sched_enabled
        self.beta = nn.Parameter(torch.tensor(beta_init), requires_grad=requires_grad)

    def forward(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta # 为方便
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self, prog=1, detach=False):
        if not self.sched_enabled:
            beta = self.beta.abs() + self.beta_min
            beta = beta.detach() if detach else beta
        else:
            beta = self.sched_conf.beta_0/(1+(self.sched_conf.beta_0-self.sched_conf.beta_1)/self.sched_conf.beta_1*(prog**0.8)) # TODO：manual sched
            beta = torch.tensor(beta, device=self.beta.device, dtype=self.beta.dtype)
        return beta

# neus, sdf->density, density = max()
class NeusDensity(nn.Module):
    def __init__(self, init_val=3, scale_factor=1.0) -> None:
        super().__init__()
        self.s_var = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))
        self.factor = scale_factor

    def forward(self, sdf, inv_s=None):
        # sdf: ..., 1
        if inv_s is None:
            inv_s = (self.s_var*self.factor).exp()
        exp = torch.exp(-inv_s*sdf)
        density = inv_s * exp / (1 + exp) ** 2
        torch.nan_to_num(density, nan=0, posinf=0, neginf=0)
        return density
    def get_s(self):
        return (self.s_var*self.factor).exp() # TODO: 增大var的grad

# (x)->(sdf, z)
class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in=3,
                 d_hidden=256,
                 n_layers=8,
                 skip=[4],  # 1-n_layers
                 geometric_init=True,
                 bias=0.5,  # 初始化球半径
                 norm_weight=True,
                 inside_outside=False,  # default outside for object
                 bound=1.0,
                 enable_fourier=True,
                 N_freqs=10, # 'fourier'
                 enable_hashgrid=True,
                 num_levels=16,
                 per_level_dim=2,
                 log2_hashmap_size=19,
                 base_resolution=16,
                 max_resolution=2048,
                 resolution_list=None,
                 enable_progressive=False,
                 init_active_level=4,
                 active_step = 5000,
                 gradient_mode='analytical',
                 taps=6,
                 ):
        super(SDFNetwork, self).__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.skip = skip
        self.geometric_init = geometric_init
        self.bias = bias
        self.norm_weight = norm_weight
        self.inside_outside = inside_outside
        self.bound = bound
        self.enable_fourier = enable_fourier
        self.N_freqs = N_freqs
        self.enable_hashgrid = enable_hashgrid
        self.num_levels = num_levels
        self.per_level_dim = per_level_dim

        ############### progressive grid ##################
        self.enable_progressive = enable_progressive
        self.init_active_level = init_active_level
        self.active_levels = init_active_level
        self.active_step = active_step
        self.warm_up = 0

        # epsilon for numerical gradient
        self.gradient_mode = gradient_mode if enable_hashgrid else 'analytical' # 'numerical' or 'analytical', and 'numerical' only for hashgrid
        self.taps = taps  # 6 or 4
        self.normal_epsilon = 0
        self.per_level_scale = np.exp(np.log(max_resolution / base_resolution) / (num_levels - 1))
        # encoder
        if self.enable_hashgrid:
            # tcnn
            print(f'[hashgrid] per_level_scale: {self.per_level_scale}')
            self.grid_encoder = tcnn.Encoding(3, {
                        "otype": "HashGrid",
                        "n_levels": num_levels,
                        "n_features_per_level": per_level_dim,
                        "log2_hashmap_size": log2_hashmap_size,
                        "base_resolution": base_resolution,
                        "per_level_scale": self.per_level_scale,
                        "interpolation": "Smoothstep"
                    })
            self.grid_encoder.resolution_list = []
            for lv in range(0, num_levels):
                size = np.floor(base_resolution * self.per_level_scale ** lv).astype(int) + 1
                self.grid_encoder.resolution_list.append(size)
            print(f'[hashgrid] resolution_list: {self.grid_encoder.resolution_list}')

            # # my grid_encoder
            # self.grid_encoder = HashEncoder(x_dim=d_in, num_levels=num_levels, per_level_dim=per_level_dim,
            #                                 log2_hashmap_size=log2_hashmap_size, base_resolution=base_resolution,
            #                                 max_resolution=max_resolution, resolution_list=resolution_list)

            self.d_in = num_levels*per_level_dim + self.d_in
        if self.enable_fourier: # [DEBUG-1] input 同时包括fourier和grid [√]
            self.fourier_encoder = FourierEncoder(d_in=d_in, max_log_freq=N_freqs - 1, N_freqs=N_freqs, log_sampling=True)
            self.d_in = self.fourier_encoder.encoded_length() + self.d_in


        # net initialization
        self.linears = torch.nn.ModuleList()
        in_features = None
        # layer_dims: [d_in]+[d_hidden]*n_layers+[d_hidden]
        for l in range(1, n_layers + 2): # n_layers+1层得到隐式feature, n_layer层通过sdf_layer->sdf
            in_features = self.d_hidden + (self.d_in if l-1 in self.skip else 0) # 上一层要cat输入
            if l == 1:
                layer = torch.nn.Linear(self.d_in, self.d_hidden)
            else:
                layer = torch.nn.Linear(in_features, self.d_hidden)
            # geometric initialization
            if geometric_init:  # 《SAL: Sign Agnostic Learning of Shapes from Raw Data》
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(self.d_hidden))
                if l == 1:  # 第一层
                    torch.nn.init.constant_(layer.weight[:, 3:], 0.0)  # 高频初始置0
                elif l - 1 in self.skip:  # 上一层skip
                    torch.nn.init.constant_(layer.weight[:, -self.d_in:], 0.0) # cat置0
                else: # 其他层
                    pass
            if norm_weight:
                layer = nn.utils.weight_norm(layer)
            self.linears.append(layer)
        self.sdf_layer = torch.nn.Linear(in_features, 1)
        if geometric_init:
            if self.inside_outside: # inside
                torch.nn.init.normal_(self.sdf_layer.weight, -np.sqrt(np.pi)/np.sqrt(in_features), std=0.0001)
                torch.nn.init.constant_(self.sdf_layer.bias,self.bias)
            else: # outside
                torch.nn.init.normal_(self.sdf_layer.weight, np.sqrt(np.pi)/np.sqrt(in_features), std=0.0001)
                torch.nn.init.constant_(self.sdf_layer.bias, -self.bias)
        self.activation = nn.Softplus(beta=100)
        # self.activation = nn.ReLU()

    def get_grid_params(self):
        return self.grid_encoder.parameters()

    def get_mlp_params(self):
        return self.linears.parameters()

    def get_feature_mask(self, feature):
        mask = torch.zeros_like(feature)
        if self.enable_progressive:
            mask[..., :(self.active_levels * self.per_level_dim)] = 1
        else:
            mask[...] = 1
        return mask

    def forward(self, x, if_cal_hessian_x=False,only_sdf=False):
        # x: (..., 3)
        # TODO: if_cal_hessian_x: 传给my grid_encoder是否计算hessian ，暂时舍弃不用
        x_enc = x
        if self.enable_fourier:
            x_enc = torch.cat([x, self.fourier_encoder(x)], dim=-1)
        if self.enable_hashgrid:
            # tcnn
            x_norm = (x.view(-1, 3)+self.bound)/(2*self.bound)
            grid_enc = self.grid_encoder(x_norm)
            grid_enc = grid_enc.view(*x.shape[:-1], -1)

            # # ---------------- my hash encoder ----------------- #
            # grid_enc=self.grid_encoder(x/self.bound)

            mask = self.get_feature_mask(grid_enc)
            x_enc = torch.cat([x_enc, grid_enc*mask], dim=-1)
        x=x_enc
        # forward
        # 1-n_layers
        for l in range(1, self.n_layers + 1):
            layer = self.linears[l - 1]
            if l - 1 in self.skip:
                x = torch.cat([x, x_enc], dim=-1)
            x = self.activation(layer(x))
        if self.n_layers in self.skip:
            x = torch.cat([x, x_enc], dim=-1)
        sdf = self.sdf_layer(x)
        if only_sdf:
            return sdf
        feat = self.activation(self.linears[-1](x))
        return torch.cat([sdf, feat], dim=-1)

    def get_sdf(self, x):
        return self.forward(x,only_sdf=True)

    def get_sdf_feat(self, x):
        output=self.forward(x)
        sdf, feat = output[..., :1], output[..., 1:]
        return sdf, feat

    def get_all(self, x, only_sdf=False, if_cal_hessian_x=True):
        # return sdf, feat, gradients
        # if if_cal_hessian_x: return sdf, feat, gradients, hessian
        # TODO: 注意这里的hessian,当为analytical时, hessian这里实际得到的是一个[...,3]的向量而不是[...,3,3]的矩阵，分别表示梯度三个分量对 x 的偏导数和(同理y、z)。
        if self.gradient_mode == 'analytical':
            required_grad = x.requires_grad
            x.requires_grad_(True)
            output = self.forward(x, if_cal_hessian_x and self.gradient_mode == 'analytical', only_sdf=only_sdf)
            sdf = output[..., :1]
            feat = None if only_sdf else output[..., 1:]
            gradients = torch.autograd.grad(outputs=sdf.sum(),inputs=x,create_graph=True, retain_graph=True)[0]
            if if_cal_hessian_x:
                hessian = torch.autograd.grad(outputs=gradients.sum(), inputs=x, create_graph=True)[0]
                x.requires_grad_(required_grad)
                return sdf, feat, gradients, hessian
            else:
                x.requires_grad_(required_grad)
                return sdf, feat, gradients, None
        elif self.gradient_mode == 'numerical':
            output = self.forward(x, False, only_sdf=only_sdf)
            sdf= output[..., :1]
            feat = None if only_sdf else output[..., 1:]
            if self.taps == 6:
                eps = self.normal_epsilon
                # 1st-order gradient
                eps_x = torch.tensor([eps, 0., 0.], dtype=x.dtype, device=x.device)  # [3]
                eps_y = torch.tensor([0., eps, 0.], dtype=x.dtype, device=x.device)  # [3]
                eps_z = torch.tensor([0., 0., eps], dtype=x.dtype, device=x.device)  # [3]
                sdf_x_pos = self.get_sdf(x + eps_x)  # [...,1]
                sdf_x_neg = self.get_sdf(x - eps_x)  # [...,1]
                sdf_y_pos = self.get_sdf(x + eps_y)  # [...,1]
                sdf_y_neg = self.get_sdf(x - eps_y)  # [...,1]
                sdf_z_pos = self.get_sdf(x + eps_z)  # [...,1]
                sdf_z_neg = self.get_sdf(x - eps_z)  # [...,1]
                gradient_x = (sdf_x_pos - sdf_x_neg) / (2 * eps)
                gradient_y = (sdf_y_pos - sdf_y_neg) / (2 * eps)
                gradient_z = (sdf_z_pos - sdf_z_neg) / (2 * eps)
                gradient = torch.cat([gradient_x, gradient_y, gradient_z], dim=-1) # [...,3]
                # 2nd-order gradient (hessian)
                if if_cal_hessian_x:
                    assert sdf is not None  # computed when feed-forwarding through the network
                    hessian_xx = (sdf_x_pos + sdf_x_neg - 2 * sdf) / (eps ** 2)  # [...,1]
                    hessian_yy = (sdf_y_pos + sdf_y_neg - 2 * sdf) / (eps ** 2)  # [...,1]
                    hessian_zz = (sdf_z_pos + sdf_z_neg - 2 * sdf) / (eps ** 2)  # [...,1]
                    hessian = torch.cat([hessian_xx, hessian_yy, hessian_zz], dim=-1)  # [...,3]
                    return sdf, feat, gradient, hessian
                else:
                    return sdf, feat, gradient, None
            elif self.taps == 4:
                eps = self.normal_epsilon / np.sqrt(3)
                k1 = torch.tensor([1, -1, -1], dtype=x.dtype, device=x.device)  # [3]
                k2 = torch.tensor([-1, -1, 1], dtype=x.dtype, device=x.device)  # [3]
                k3 = torch.tensor([-1, 1, -1], dtype=x.dtype, device=x.device)  # [3]
                k4 = torch.tensor([1, 1, 1], dtype=x.dtype, device=x.device)  # [3]
                sdf1 = self.get_sdf(x + k1 * eps)  # [...,1]
                sdf2 = self.get_sdf(x + k2 * eps)  # [...,1]
                sdf3 = self.get_sdf(x + k3 * eps)  # [...,1]
                sdf4 = self.get_sdf(x + k4 * eps)  # [...,1]
                gradient = (k1 * sdf1 + k2 * sdf2 + k3 * sdf3 + k4 * sdf4) / (4.0 * eps)
                if if_cal_hessian_x:
                    # hessian = None
                    assert sdf is not None  # computed when feed-forwarding through the network
                    # the result of 4 taps is directly trace, but we assume they are individual components
                    # so we use the same signature as 6 taps
                    hessian_xx = ((sdf1 + sdf2 + sdf3 + sdf4) / 2.0 - 2 * sdf) / eps ** 2   # [N,1]
                    hessian = torch.cat([hessian_xx, hessian_xx, hessian_xx], dim=-1) / 3.0
                    return sdf, feat, gradient, hessian
                else:
                    return sdf, feat, gradient, None
            else:
                raise NotImplementedError

    def set_active_levels(self, cur_step):
        if not self.enable_hashgrid:
            return
        self.anneal_levels = min(max((cur_step - self.warm_up) // self.active_step, 1), self.num_levels)
        self.active_levels = max(self.anneal_levels, self.init_active_level) if self.enable_progressive else self.num_levels
    def set_normal_epsilon(self):
        if not self.enable_hashgrid:
            return
        if self.enable_progressive: # normal_epsilon是grid Voxel边长的1/4
            if self.inside_outside: # indoor should be more smooth in the beginning
                self.normal_epsilon = 2.0 *self.bound/ (self.grid_encoder.resolution_list[self.anneal_levels - 1])/4
            else:
                self.normal_epsilon = 2.0 *self.bound/ (self.grid_encoder.resolution_list[self.anneal_levels - 1])/4 # FIXME：这里如果用active levels更新eps，前期（即anneal levels<init_active_levels）表面非常smooth、flat学得非常快，对物体而言很容易导致大块的凸起肿块（unaware-abrupt-depth），且后期很难消去，看使用场景，如果需要前期非常smooth就active_levels。
        else:
            self.normal_epsilon = 2.0 *self.bound / (self.grid_encoder.resolution_list[-1])/4

    def get_feature_mask(self, feature):
        mask = torch.zeros_like(feature)
        if self.enable_progressive:
            mask[..., :(self.active_levels * self.per_level_dim)] = 1
        else:
            mask[...] = 1
        return mask

# (x,v,n,z)->(color) 即idr模式
class ColorNetwork(nn.Module):
    def __init__(self,
                 feat_dim=256,  # sdf feature
                 d_hidden=256,
                 n_layers=4,
                 skip=[],
                 N_freqs=3,
                 encoding_view='spherical',  # 'fourier' 或者 'spherical'
                 weight_norm=True,
                 layer_norm=False,
                 enable_app=False, 
                 app_dim=8,
                 enable_uncertainty=False,):
        super(ColorNetwork, self).__init__()
        self.feat_dim = feat_dim
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.N_freqs = N_freqs
        self.encoding_view = encoding_view
        self.enable_app = enable_app
        self.app_dim = app_dim
        self.enable_uncertainty = enable_uncertainty

        if self.encoding_view == 'fourier':
            self.view_encoder = FourierEncoder(d_in=3, max_log_freq=N_freqs - 1, N_freqs=N_freqs)
        elif self.encoding_view == 'spherical':
            self.view_encoder = SphericalEncoder(levels=N_freqs)
        else:
            raise NotImplementedError
        view_enc_dim = self.view_encoder.encoded_length()

        # build mlp, idr mode: C=f(x,v,n,z,app)
        layer_dims = [3 + view_enc_dim + 3 + feat_dim + (app_dim if enable_app else 0)] + [d_hidden] * n_layers + [3]
        self.mlp = MLPwithSkipConnection(layer_dims, skip_connection=skip, activ=nn.ReLU(), use_layernorm=layer_norm,use_weightnorm=weight_norm)

    def forward(self, x, v, n, z, app=None):
        # x: (..., 3)
        # v: (..., 3)
        # n: (..., 3)
        # z: (..., feat_dim)
        # app: (..., app_dim)
        # return: rgb(..., 3)

        # view encoding 无需cat方向自身
        view_encoding = self.view_encoder(v)
        # view_encoding = torch.cat([v, view_encoding], dim=-1)
        if app is None and self.enable_app == True: # 添0
            app = torch.zeros_like(x[...,:1]).tile(*((x.ndim-1)*[1]),self.app_dim)
        input = torch.cat([x, view_encoding, n, z], dim=-1) if app is None else torch.cat([x, view_encoding, n, z, app],dim=-1)
        rgb = self.mlp(input).sigmoid_()
        return rgb

# TODO: background也用hash grids建模
# Background NeRF
class NeRFPlusPlus(nn.Module):
    def __init__(self,
                 bound=1.0,
                 d_hidden=256,
                 n_layers=8,
                 skip=[4],  # density network
                 d_hidden_rgb=128,
                 n_layers_rgb=2,
                 skip_rgb=[],
                 encoding='fourier',
                 N_freqs=10,
                 encoding_view='spherical',
                 N_freqs_view=3,
                 enable_app=False,
                 app_dim=8):
        super().__init__()
        self.bound=bound
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.d_hidden_rgb = d_hidden_rgb
        self.n_layers_rgb = n_layers_rgb
        self.encoding = encoding
        self.encoding_view = encoding_view
        self.enable_app=enable_app
        self.app_dim=app_dim

        if self.encoding == 'fourier':  # inverse sphere sampling即(x',y',z',1/r)，输入是4维。
            self.encoder = FourierEncoder(d_in=4, max_log_freq=N_freqs - 1, N_freqs=N_freqs)
        else:
            raise NotImplementedError

        if self.encoding_view == 'fourier':
            self.view_encoder = FourierEncoder(d_in=3, max_log_freq=N_freqs_view - 1, N_freqs=N_freqs_view)
        elif self.encoding_view == 'spherical':
            self.view_encoder = SphericalEncoder(levels=N_freqs_view)
        else:
            raise NotImplementedError

        self.activ_mlp = nn.ReLU()
        # density mlp
        layer_dims_density = [self.encoder.encoded_length() + 4] + [d_hidden] * (n_layers - 1) + [1 + d_hidden]
        self.density_mlp = MLPwithSkipConnection(layer_dims_density, skip, self.activ_mlp)
        self.activ_density = nn.Softplus()

        # rgb mlp
        layer_dims_rgb = [d_hidden + self.view_encoder.encoded_length() + (app_dim if enable_app else 0)] + [
            d_hidden_rgb] * (n_layers_rgb - 1) + [3]
        self.rgb_mlp = MLPwithSkipConnection(layer_dims_rgb, skip_rgb, self.activ_mlp)

    def get_density(self,x, with_feat=False):
        dist = torch.norm(x, dim=-1, keepdim=True)+1e-6
        norm_x = torch.cat([x / dist, 1 / dist], dim=-1)
        x_encoding = torch.cat([norm_x, self.encoder(norm_x)], dim=-1)
        density_output = self.density_mlp(x_encoding)
        sigma = self.activ_density(density_output[..., :1])
        if with_feat:
            feat = self.activ_mlp(density_output[..., 1:])
            return sigma, feat
        return sigma

    def forward(self, x, v, o, app=None):
        # x: (..., 3)
        # v: (..., 3): 光线方向
        # o: (..., 3): 光线origin
        # app: (..., app_dim)

        # 1. 逆球面重参数化
        # norm_x = self.reverse_spherical_reparameterization(x, v, o)

        # 2. 简化版本
        dist = torch.norm(x, dim=-1, keepdim=True)+1e-6
        norm_x = torch.cat([x / dist, 1 / dist], dim=-1)

        x_encoding = torch.cat([norm_x, self.encoder(norm_x)], dim=-1)  # （..., 4+encoded_length）
        view_encoding = self.view_encoder(v)

        density_output = self.density_mlp(x_encoding)
        sigma, feat = self.activ_density(density_output[..., :1]), self.activ_mlp(density_output[..., 1:])

        rgb_input = torch.cat([feat, view_encoding, app], dim=-1) if app is not None else torch.cat(
            [feat, view_encoding], dim=-1)
        rgb = self.rgb_mlp(rgb_input).sigmoid_()
        return sigma, rgb

    def reverse_spherical_reparameterization(self, x, v, o):
        # 逆球面重参数化(注意边界球是bound，但要重参数化为单位球面)，绕v×x（为旋转轴）旋转w度。
        # x, v, o: (..., 3)
        # return: reparam_x (..., 4)
        r = torch.norm(x, dim=-1, keepdim=True)+1e-6
        x = x / r
        rot_axis = F.normalize(torch.cross(v, x, dim=-1), p=2, dim=-1)
        h=(torch.sum(o*o,dim=-1,keepdim=True)-torch.sum(o*v,dim=-1,keepdim=True)**2).sqrt()
        theta = torch.arccos(h/r)
        phi = torch.arccos(h/self.bound)
        w = theta - phi
        # 将x绕a旋转w度: x'=x*cos(w)+sin(w)*(a×x)+a*(a·x)(1-cos(w))
        x_rot = x * torch.cos(w) + torch.sin(w) * torch.cross(rot_axis, x, dim=-1) + rot_axis*torch.sum(rot_axis*x,dim=-1,keepdim=True)*(1-torch.cos(w))
        return torch.cat([x_rot, 1 / r], dim=-1)

# 无边界场景的背景Nerf，压缩方式：Mip-Nerf 360，position encoding：hashgrid和fourier
class GridNeRF(nn.Module):
    def __init__(self,
                 bound=1.0,
                 d_hidden=64,
                 n_layers=2,
                 skip=[],
                 d_hidden_rgb=64,
                 n_layers_rgb=2,
                 skip_rgb=[],
                 norm_weight=True,
                 enable_fourier=True,
                 N_freqs=6,
                 enable_hashgrid=True,
                 num_levels=16,
                 per_level_dim=2,
                 log2_hashmap_size=19,
                 base_resolution=32,
                 max_resolution=2048,
                 resolution_list=None,
                 enable_progressive=True,
                 init_active_level=4,
                 active_step=1000,
                 encoding_view='spherical',
                 N_freqs_view=3,
                 enable_app=False,
                 app_dim=8):
        super().__init__()
        self.bound=bound
        self.enable_fourier = enable_fourier
        self.enable_hashgrid = enable_hashgrid
        self.num_levels = num_levels
        self.per_level_dim = per_level_dim
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.enable_progressive = enable_progressive
        self.init_active_level = init_active_level
        self.active_levels = init_active_level
        self.active_step = active_step
        self.warm_up = 0
        self.encoding_view = encoding_view
        self.enable_app = enable_app
        self.app_dim = app_dim

        self.d_in = 3
        # position encoding
        if self.enable_hashgrid:
            # tcnn
            self.per_level_scale = np.exp(np.log(max_resolution / base_resolution) / (num_levels - 1))
            print(f'[hashgrid] per_level_scale: {self.per_level_scale}')
            self.grid_encoder = tcnn.Encoding(3, {
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": per_level_dim,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": self.per_level_scale,
                "interpolation": "Smoothstep"
            })
            self.grid_encoder.resolution_list = []
            for lv in range(0, num_levels):
                size = np.floor(base_resolution * self.per_level_scale ** lv).astype(int) + 1
                self.grid_encoder.resolution_list.append(size)
            print(f'[hashgrid] resolution_list: {self.grid_encoder.resolution_list}')

            # # my grid_encoder
            # self.grid_encoder = HashEncoder(x_dim=d_in, num_levels=num_levels, per_level_dim=per_level_dim,
            #                                 log2_hashmap_size=log2_hashmap_size, base_resolution=base_resolution,
            #                                 max_resolution=max_resolution, resolution_list=resolution_list)

            self.d_in = num_levels * per_level_dim + self.d_in
        if self.enable_fourier:
            self.fourier_encoder = FourierEncoder(d_in=3, max_log_freq=N_freqs - 1, N_freqs=N_freqs,log_sampling=True)
            self.d_in = self.fourier_encoder.encoded_length() + self.d_in

        # view encoding
        if self.encoding_view == 'fourier':
            self.view_encoder = FourierEncoder(d_in=3, max_log_freq=N_freqs_view - 1, N_freqs=N_freqs_view)
        elif self.encoding_view == 'spherical':
            self.view_encoder = SphericalEncoder(levels=N_freqs_view)
        else:
            raise NotImplementedError

        self.activ_mlp = nn.ReLU()
        # density mlp
        layer_dims_density = [self.d_in] + [d_hidden] * (n_layers - 1) + [1 + d_hidden]
        self.density_mlp = MLPwithSkipConnection(layer_dims_density, skip, self.activ_mlp)
        self.activ_density = trunc_exp # [CHANGE-2] 用trunc_exp或者nn.Softplus()

        # rgb mlp
        layer_dims_rgb = [d_hidden + self.view_encoder.encoded_length() + (app_dim if enable_app else 0)] + [
            d_hidden_rgb] * (n_layers_rgb - 1) + [3]
        self.rgb_mlp = MLPwithSkipConnection(layer_dims_rgb, skip_rgb, self.activ_mlp)

    def get_density(self, x, with_feat=False):
        x_enc = x
        if self.enable_fourier:
            x_enc = torch.cat([x, self.fourier_encoder(x)], dim=-1)
        if self.enable_hashgrid:
            # tcnn
            x_norm = (x.view(-1, 3) + self.bound) / (2 * self.bound)
            grid_enc = self.grid_encoder(x_norm)
            grid_enc = grid_enc.view(*x.shape[:-1], -1)
            # ------------------------------------- #
            # grid_enc=self.grid_encoder(x/self.bound, if_cal_hessian_x)

            mask = self.get_feature_mask(grid_enc)
            x_enc = torch.cat([x_enc, grid_enc * mask], dim=-1)
        density_output = self.density_mlp(x_enc)
        sigma = self.activ_density(density_output[..., :1])
        if with_feat:
            feat = self.activ_mlp(density_output[..., 1:])
            return sigma, feat
        return sigma

    def forward(self, x, v, o, app=None):
        # x: (..., 3): ∈[-bound,bound]^3
        # v: (..., 3): 光线方向
        # o: (..., 3): 光线origin, 无用
        # app: (..., app_dim)

        x_enc = x
        if self.enable_fourier:
            x_enc = torch.cat([x, self.fourier_encoder(x)], dim=-1)
        if self.enable_hashgrid:
            # tcnn
            x_norm = (x.view(-1, 3) + self.bound) / (2*self.bound)
            grid_enc = self.grid_encoder(x_norm)
            grid_enc = grid_enc.view(*x.shape[:-1], -1)
            # ------------------------------------- #
            # grid_enc=self.grid_encoder(x/self.bound, if_cal_hessian_x)

            mask = self.get_feature_mask(grid_enc)
            x_enc = torch.cat([x_enc, grid_enc * mask], dim=-1)
        view_encoding = self.view_encoder(v)

        density_output = self.density_mlp(x_enc)
        sigma, feat = self.activ_density(density_output[..., :1]), self.activ_mlp(density_output[..., 1:])

        rgb_input = torch.cat([feat, view_encoding, app], dim=-1) if app is not None else torch.cat(
            [feat, view_encoding], dim=-1)
        rgb = self.rgb_mlp(rgb_input).sigmoid_()
        return sigma, rgb

    def get_feature_mask(self, feature):
        mask = torch.zeros_like(feature)
        if self.enable_progressive:
            mask[..., :(self.active_levels * self.per_level_dim)] = 1
        else:
            mask[...] = 1
        return mask
    def set_active_levels(self, cur_step):
        self.anneal_levels = min(max((cur_step - self.warm_up) // self.active_step, 1), self.num_levels)
        self.active_levels = max(self.anneal_levels, self.init_active_level)

    def get_grid_params(self):
        return self.grid_encoder.parameters()
    def get_mlp_params(self):
        # density mlp and rgb mlp
        return list(self.density_mlp.parameters()) + list(self.rgb_mlp.parameters())

# NBField: (x,v,z) -> (quat)
class NBField(nn.Module):
    def __init__(self,
                 feat_dim=256,  # sdf feature
                 d_hidden=256,
                 n_layers=2,
                 skip=[],
                 N_freqs=3,
                 encoding_view='spherical',  # 'fourier' 或者 'spherical'
                 weight_norm=True,
                 layer_norm=False,
                 enable_app=False,
                 app_dim=8) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.N_freqs = N_freqs
        self.encoding_view = encoding_view
        self.enable_app = enable_app
        self.app_dim = app_dim

        if self.encoding_view == 'fourier':
            self.view_encoder = FourierEncoder(d_in=3, max_log_freq=N_freqs - 1, N_freqs=N_freqs)
        elif self.encoding_view == 'spherical':
            self.view_encoder = SphericalEncoder(levels=N_freqs)
        else:
            raise NotImplementedError
        view_enc_dim = self.view_encoder.encoded_length()

        # build mlp, idr mode: C=f(x,v,z, app=None)
        layer_dims = [3 + 3 + view_enc_dim + 3 + feat_dim + (app_dim if enable_app else 0)] + [d_hidden] * n_layers + [4]
        self.mlp = MLPwithSkipConnection(layer_dims, skip_connection=skip, activ=nn.LeakyReLU(0.2), use_layernorm=layer_norm,
                                         use_weightnorm=weight_norm)
        torch.nn.init.constant_(self.mlp.linears[-1].weight, 0.0)
        self.mlp.linears[-1].bias.data = torch.tensor([0., 1., 0., 0.])

    def forward(self, x, v, n, z, app=None):
        # x: (..., 3)
        # v: (..., 3)
        # n: (..., 3)
        # z: (..., feat_dim)
        # return: quat(..., 4)

        # view encoding 无需cat方向自身
        view_encoding = self.view_encoder(v)
        view_encoding = torch.cat([v, view_encoding], dim=-1)
        if app is None and self.enable_app == True: # 添0
            app = torch.zeros_like(x[...,:1]).tile(*((x.ndim-1)*[1]),self.app_dim)
        input = torch.cat([x, view_encoding, n, z], dim=-1) if app is None else torch.cat([x, view_encoding, n, z, app], dim=-1)
        quat = self.mlp(input)
        quat = quat / (1e-12 + quat.norm(p=2, dim=-1, keepdim=True))
        return quat



class LightDensityField(nn.Module):
    def __init__(self,
                 bound=1.0,
                 d_hidden=64,
                 n_layers=2,
                 skip=[],
                 norm_weight=True,
                 enable_fourier=False,
                 N_freqs=6,
                 enable_hashgrid=True,
                 num_levels=8,
                 per_level_dim=2,
                 log2_hashmap_size=18,
                 base_resolution=16,
                 max_resolution=1024,
                 resolution_list=None,):
        super().__init__()
        self.bound=bound
        self.enable_fourier = enable_fourier
        self.enable_hashgrid = enable_hashgrid
        self.num_levels = num_levels
        self.per_level_dim = per_level_dim
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution

        self.d_in = 3
        # position encoding
        if self.enable_hashgrid:
            # tcnn
            self.per_level_scale = np.exp(np.log(max_resolution / base_resolution) / (num_levels - 1))
            print(f'[hashgrid] per_level_scale: {self.per_level_scale}')
            self.grid_encoder = tcnn.Encoding(3, {
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": per_level_dim,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": self.per_level_scale,
                "interpolation": "Smoothstep"
            })
            self.grid_encoder.resolution_list = []
            for lv in range(0, num_levels):
                size = np.floor(base_resolution * self.per_level_scale ** lv).astype(int) + 1
                self.grid_encoder.resolution_list.append(size)
            print(f'[hashgrid] resolution_list: {self.grid_encoder.resolution_list}')

            # # my grid_encoder
            # self.grid_encoder = HashEncoder(x_dim=d_in, num_levels=num_levels, per_level_dim=per_level_dim,
            #                                 log2_hashmap_size=log2_hashmap_size, base_resolution=base_resolution,
            #                                 max_resolution=max_resolution, resolution_list=resolution_list)

            self.d_in = num_levels * per_level_dim + self.d_in
        if self.enable_fourier:
            self.fourier_encoder = FourierEncoder(d_in=3, max_log_freq=N_freqs - 1, N_freqs=N_freqs,log_sampling=True)
            self.d_in = self.fourier_encoder.encoded_length() + self.d_in

        self.activ_mlp = nn.ReLU()
        # density mlp
        layer_dims_density = [self.d_in] + [d_hidden] * (n_layers - 1) + [1]
        self.density_mlp = MLPwithSkipConnection(layer_dims_density, skip, self.activ_mlp, use_weightnorm=norm_weight)
        self.activ_density = nn.Softplus() # FIXME：用trunc_exp或者nn.Softplus()

    def get_density(self, x):
        x_enc = x
        if self.enable_fourier:
            x_enc = torch.cat([x, self.fourier_encoder(x)], dim=-1)
        if self.enable_hashgrid:
            # tcnn
            x_norm = (x.view(-1, 3) + self.bound) / (2 * self.bound)
            grid_enc = self.grid_encoder(x_norm)
            grid_enc = grid_enc.view(*x.shape[:-1], -1)
            # ------------------------------------- #
            # grid_enc=self.grid_encoder(x/self.bound, if_cal_hessian_x)

            mask = self.get_feature_mask(grid_enc)
            x_enc = torch.cat([x_enc, grid_enc * mask], dim=-1)
        mlp_output = self.density_mlp(x_enc)
        sigma = self.activ_density(mlp_output)
        return sigma


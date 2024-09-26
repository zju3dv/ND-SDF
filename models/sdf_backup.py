# class SDFNetwork(nn.Module):
#     def __init__(self,
#                  d_in=3,
#                  d_hidden=256,
#                  n_layers=8,
#                  skip=[4],  # 1-n_layers
#                  geometric_init=True,
#                  bias=0.5,  # 初始化球半径
#                  norm_weight=True,
#                  inside_outside=False,  # default outside for object
#                  bound=1.0,
#                  enable_fourier=True,
#                  N_freqs=10, # 'fourier'
#                  enable_hashgrid=True,
#                  num_levels=16,
#                  per_level_dim=2,
#                  log2_hashmap_size=19,
#                  base_resolution=16,
#                  max_resolution=2048,
#                  resolution_list=None,
#                  enable_progressive=False,
#                  init_active_level=4,
#                  active_step = 5000,
#                  gradient_mode='analytical',
#                  taps=6,
#                  ):
#         super(SDFNetwork, self).__init__()
#         self.d_in = d_in
#         self.d_hidden = d_hidden
#         self.n_layers = n_layers
#         self.skip = skip
#         self.geometric_init = geometric_init
#         self.bias = bias
#         self.norm_weight = norm_weight
#         self.inside_outside = inside_outside
#         self.bound = bound
#         self.enable_fourier = enable_fourier
#         self.N_freqs = N_freqs
#         self.enable_hashgrid = enable_hashgrid
#         self.num_levels = num_levels
#         self.per_level_dim = per_level_dim
#
#         ############### progressive grid ##################
#         self.enable_progressive = enable_progressive
#         self.init_active_level = init_active_level
#         self.active_levels = init_active_level
#         self.active_step = active_step
#         self.warm_up = 0
#
#         # epsilon for numerical gradient
#         self.gradient_mode = gradient_mode  # 'numerical' or 'analytical'
#         self.taps = taps  # 6 or 4
#         self.normal_epsilon = 0
#
#         # encoder
#         if self.enable_hashgrid:
#             # tcnn
#             per_level_scale = np.exp(np.log(max_resolution/base_resolution) / (num_levels-1))
#             print(f'[hashgrid] per_level_scale: {per_level_scale}')
#             self.grid_encoder = tcnn.Encoding(3, {
#                         "otype": "HashGrid",
#                         "n_levels": num_levels,
#                         "n_features_per_level": per_level_dim,
#                         "log2_hashmap_size": log2_hashmap_size,
#                         "base_resolution": base_resolution,
#                         "per_level_scale": per_level_scale,
#                         "interpolation": "Smoothstep"
#                     })
#             self.grid_encoder.resolution_list = []
#             for lv in range(0, num_levels):
#                 size = np.floor(base_resolution * per_level_scale ** lv).astype(int) + 1
#                 self.grid_encoder.resolution_list.append(size)
#             print(f'[hashgrid] resolution_list: {self.grid_encoder.resolution_list}')
#
#             # # my grid_encoder
#             # self.grid_encoder = HashEncoder(x_dim=d_in, num_levels=num_levels, per_level_dim=per_level_dim,
#             #                                 log2_hashmap_size=log2_hashmap_size, base_resolution=base_resolution,
#             #                                 max_resolution=max_resolution, resolution_list=resolution_list)
#
#             # self.grid_encoder = HashEncoder(input_dim=d_in, num_levels=num_levels, per_level_scale=1.381817, base_resolution=base_resolution,
#             #                            log2_hashmap_size=log2_hashmap_size,desired_resolution=max_resolution)
#             # self.grid_encoder.resolution_list = []
#             # for lv in range(0, num_levels):
#             #     size = np.floor(base_resolution * 1.381817 ** lv).astype(int) + 1
#             #     self.grid_encoder.resolution_list.append(size)
#             # print(f'[hashgrid] resolution_list: {self.grid_encoder.resolution_list}')
#
#             self.d_in = num_levels*per_level_dim + self.d_in
#         if self.enable_fourier: # [DEBUG-1] input 同时包括fourier和grid [√]
#             self.fourier_encoder = FourierEncoder(d_in=d_in, max_log_freq=N_freqs - 1, N_freqs=N_freqs, log_sampling=True)
#             self.d_in = self.fourier_encoder.encoded_length() + self.d_in
#
#
#         # net initialization
#         self.linears = torch.nn.ModuleList()
#         for l in range(1, n_layers + 2):
#             in_features = self.d_hidden + (self.d_in if l-1 in self.skip else 0) # 上一层要cat输入
#             if l == 1:
#                 layer = torch.nn.Linear(self.d_in, self.d_hidden)
#             elif l <= n_layers:
#                 layer = torch.nn.Linear(in_features, self.d_hidden)
#             else:
#                 layer = torch.nn.Linear(in_features, self.d_hidden+1)
#             # geometric initialization
#             if geometric_init:  # 《SAL: Sign Agnostic Learning of Shapes from Raw Data》
#                 if l == n_layers + 1:  # 输出层
#                     if inside_outside:  # inside
#                         torch.nn.init.normal_(layer.weight, mean=-np.sqrt(np.pi) / np.sqrt(in_features), std=0.0001)
#                         torch.nn.init.constant_(layer.bias, bias)  # 保证scene内中心sdf为正
#                     else:  # outside
#                         torch.nn.init.normal_(layer.weight, mean=np.sqrt(np.pi) / np.sqrt(in_features), std=0.0001)
#                         torch.nn.init.constant_(layer.bias, -bias)  # 保证object内中心sdf为负
#                 elif l == 1:  # 第一层
#                     torch.nn.init.constant_(layer.bias, 0.0)
#                     torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(self.d_hidden))
#                     torch.nn.init.constant_(layer.weight[:, 3:], 0.0)  # 高频初始置0
#                 elif l - 1 in self.skip:  # 上一层skip
#                     torch.nn.init.constant_(layer.bias, 0.0)
#                     torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(self.d_hidden))
#                     torch.nn.init.constant_(layer.weight[:, -self.d_in:], 0.0) # cat置0
#                 else: # 其他层
#                     torch.nn.init.constant_(layer.bias, 0.0)
#                     torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(self.d_hidden))
#             if norm_weight:
#                 layer = nn.utils.weight_norm(layer)
#             self.linears.append(layer)
#
#         self.activation = nn.Softplus(beta=100)
#         # self.activation = nn.ReLU()
#
#     def get_grid_params(self):
#         return self.grid_encoder.parameters()
#
#     def get_mlp_params(self):
#         return self.linears.parameters()
#
#     def get_feature_mask(self, feature):
#         mask = torch.zeros_like(feature)
#         if self.enable_progressive:
#             mask[..., :(self.active_levels * self.per_level_dim)] = 1
#         else:
#             mask[...] = 1
#         return mask
#
#     def forward(self, x, if_cal_hessian_x=False):
#         # if_cal_hessian_x: 传给my grid_encoder是否计算hessian
#         x_enc = x
#         if self.enable_fourier:
#             x_enc = torch.cat([x, self.fourier_encoder(x)], dim=-1)
#         if self.enable_hashgrid:
#             # tcnn
#             x_norm = (x.view(-1, 3)+self.bound)/(2*self.bound)
#             grid_enc = self.grid_encoder(x_norm)
#             grid_enc = grid_enc.view(*x.shape[:-1], -1)
#             # ------------------------------------- #
#             # grid_enc=self.grid_encoder(x/self.bound, if_cal_hessian_x)
#
#             mask = self.get_feature_mask(grid_enc)
#             x_enc = torch.cat([x_enc, grid_enc*mask], dim=-1)
#         x=x_enc
#         for l in range(1, self.n_layers + 2):
#             layer = self.linears[l - 1]
#             if l - 1 in self.skip:
#                 x = torch.cat([x, x_enc], dim=-1)
#             x = layer(x)
#             if l < self.n_layers + 1:
#                 x = self.activation(x)
#         return x
#
#     def get_sdf(self, x):
#         return self.forward(x)[..., :1]
#
#     def get_sdf_feat(self, x):
#         output=self.forward(x)
#         sdf, feat = output[..., :1], output[..., 1:]
#         return sdf, feat
#
#     # def gradient(self, x):
#     #     if self.gradient_mode == 'analytical':
#     #         x.requires_grad_(True)
#     #         y = self.forward(x)[..., :1]
#     #         d_output = torch.ones_like(y, requires_grad=False, device=y.device)
#     #         gradients = torch.autograd.grad(
#     #             outputs=y,
#     #             inputs=x,
#     #             grad_outputs=d_output,
#     #             create_graph=True,
#     #             retain_graph=True,
#     #             only_inputs=True)[0]
#     #         # TODO: 应用hessian曲率loss进行场景平滑
#     #
#     #         return gradients
#     #     elif self.gradient_mode == 'numerical':
#     #         if self.taps == 6:
#     #             eps = self.normal_epsilon
#     #             # 1st-order gradient
#     #             eps_x = torch.tensor([eps, 0., 0.], dtype=x.dtype, device=x.device)  # [3]
#     #             eps_y = torch.tensor([0., eps, 0.], dtype=x.dtype, device=x.device)  # [3]
#     #             eps_z = torch.tensor([0., 0., eps], dtype=x.dtype, device=x.device)  # [3]
#     #             sdf_x_pos = self.get_sdf(x + eps_x).cpu()  # [...,1]
#     #             sdf_x_neg = self.get_sdf(x - eps_x).cpu()  # [...,1]
#     #             sdf_y_pos = self.get_sdf(x + eps_y).cpu()  # [...,1]
#     #             sdf_y_neg = self.get_sdf(x - eps_y).cpu()  # [...,1]
#     #             sdf_z_pos = self.get_sdf(x + eps_z).cpu()  # [...,1]
#     #             sdf_z_neg = self.get_sdf(x - eps_z).cpu()  # [...,1]
#     #             gradient_x = (sdf_x_pos - sdf_x_neg) / (2 * eps)
#     #             gradient_y = (sdf_y_pos - sdf_y_neg) / (2 * eps)
#     #             gradient_z = (sdf_z_pos - sdf_z_neg) / (2 * eps)
#     #             gradient = torch.cat([gradient_x, gradient_y, gradient_z], dim=-1).cuda()  # [...,3]
#     #             # 2nd-order gradient (hessian)
#     #             if self.training:
#     #                 hessian=None
#     #                 # assert sdf is not None  # computed when feed-forwarding through the network
#     #                 # hessian_xx = (sdf_x_pos + sdf_x_neg - 2 * sdf) / (eps ** 2)  # [...,1]
#     #                 # hessian_yy = (sdf_y_pos + sdf_y_neg - 2 * sdf) / (eps ** 2)  # [...,1]
#     #                 # hessian_zz = (sdf_z_pos + sdf_z_neg - 2 * sdf) / (eps ** 2)  # [...,1]
#     #                 # hessian = torch.cat([hessian_xx, hessian_yy, hessian_zz], dim=-1)  # [...,3]
#     #             else:
#     #                 hessian = None
#     #             return gradient
#     #         elif self.taps == 4:
#     #             eps = self.normal_eps / np.sqrt(3)
#     #             k1 = torch.tensor([1, -1, -1], dtype=x.dtype, device=x.device)  # [3]
#     #             k2 = torch.tensor([-1, -1, 1], dtype=x.dtype, device=x.device)  # [3]
#     #             k3 = torch.tensor([-1, 1, -1], dtype=x.dtype, device=x.device)  # [3]
#     #             k4 = torch.tensor([1, 1, 1], dtype=x.dtype, device=x.device)  # [3]
#     #             sdf1 = self.get_sdf(x + k1 * eps)  # [...,1]
#     #             sdf2 = self.get_sdf(x + k2 * eps)  # [...,1]
#     #             sdf3 = self.get_sdf(x + k3 * eps)  # [...,1]
#     #             sdf4 = self.get_sdf(x + k4 * eps)  # [...,1]
#     #             gradient = (k1*sdf1 + k2*sdf2 + k3*sdf3 + k4*sdf4) / (4.0 * eps)
#     #             if self.training:
#     #                 hessian=None
#     #                 # assert sdf is not None  # computed when feed-forwarding through the network
#     #                 # # the result of 4 taps is directly trace, but we assume they are individual components
#     #                 # # so we use the same signature as 6 taps
#     #                 # hessian_xx = ((sdf1 + sdf2 + sdf3 + sdf4) / 2.0 - 2 * sdf) / eps ** 2   # [N,1]
#     #                 # hessian = torch.cat([hessian_xx, hessian_xx, hessian_xx], dim=-1) / 3.0
#     #             else:
#     #                 hessian = None
#     #             return gradient
#     #         else:
#     #             raise NotImplementedError
#
#     def get_all(self, x, if_cal_hessian_x=False):
#         # return sdf, feat, gradients
#         # if if_cal_hessian_x: return sdf, feat, gradients, hessian
#         # TODO: 注意这里的hessian,当为analytical时, hessian这里实际得到的是一个[...,3]的向量而不是[...,3,3]的矩阵，分别表示梯度三个分量对 x 的偏导数和(同理y、z)。
#         x.requires_grad_(True)
#         output = self.forward(x, False and self.gradient_mode == 'analytical')
#         sdf, feat = output[..., :1], output[..., 1:]
#         if self.gradient_mode == 'analytical':
#             d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
#             gradients = torch.autograd.grad(
#                 outputs=sdf,
#                 inputs=x,
#                 grad_outputs=d_output,
#                 create_graph=True,
#                 retain_graph=True,
#                 only_inputs=True)[0]
#             if if_cal_hessian_x:
#                 hessian = torch.autograd.grad(outputs=gradients.sum(), inputs=x, create_graph=True)[0]
#                 return sdf, feat, gradients, hessian
#             else:
#                 return sdf, feat, gradients
#         elif self.gradient_mode == 'numerical':
#             if self.taps == 6:
#                 eps = self.normal_epsilon
#                 # 1st-order gradient
#                 eps_x = torch.tensor([eps, 0., 0.], dtype=x.dtype, device=x.device)  # [3]
#                 eps_y = torch.tensor([0., eps, 0.], dtype=x.dtype, device=x.device)  # [3]
#                 eps_z = torch.tensor([0., 0., eps], dtype=x.dtype, device=x.device)  # [3]
#                 sdf_x_pos = self.get_sdf(x + eps_x)  # [...,1]
#                 sdf_x_neg = self.get_sdf(x - eps_x)  # [...,1]
#                 sdf_y_pos = self.get_sdf(x + eps_y)  # [...,1]
#                 sdf_y_neg = self.get_sdf(x - eps_y)  # [...,1]
#                 sdf_z_pos = self.get_sdf(x + eps_z)  # [...,1]
#                 sdf_z_neg = self.get_sdf(x - eps_z)  # [...,1]
#                 gradient_x = (sdf_x_pos - sdf_x_neg) / (2 * eps)
#                 gradient_y = (sdf_y_pos - sdf_y_neg) / (2 * eps)
#                 gradient_z = (sdf_z_pos - sdf_z_neg) / (2 * eps)
#                 gradient = torch.cat([gradient_x, gradient_y, gradient_z], dim=-1) # [...,3]
#                 # 2nd-order gradient (hessian)
#                 if if_cal_hessian_x:
#                     assert sdf is not None  # computed when feed-forwarding through the network
#                     hessian_xx = (sdf_x_pos + sdf_x_neg - 2 * sdf) / (eps ** 2)  # [...,1]
#                     hessian_yy = (sdf_y_pos + sdf_y_neg - 2 * sdf) / (eps ** 2)  # [...,1]
#                     hessian_zz = (sdf_z_pos + sdf_z_neg - 2 * sdf) / (eps ** 2)  # [...,1]
#                     hessian = torch.cat([hessian_xx, hessian_yy, hessian_zz], dim=-1)  # [...,3]
#                     return sdf, feat, gradient, hessian
#                 else:
#                     return sdf, feat, gradient
#             elif self.taps == 4:
#                 eps = self.normal_epsilon / np.sqrt(3)
#                 k1 = torch.tensor([1, -1, -1], dtype=x.dtype, device=x.device)  # [3]
#                 k2 = torch.tensor([-1, -1, 1], dtype=x.dtype, device=x.device)  # [3]
#                 k3 = torch.tensor([-1, 1, -1], dtype=x.dtype, device=x.device)  # [3]
#                 k4 = torch.tensor([1, 1, 1], dtype=x.dtype, device=x.device)  # [3]
#                 sdf1 = self.get_sdf(x + k1 * eps)  # [...,1]
#                 sdf2 = self.get_sdf(x + k2 * eps)  # [...,1]
#                 sdf3 = self.get_sdf(x + k3 * eps)  # [...,1]
#                 sdf4 = self.get_sdf(x + k4 * eps)  # [...,1]
#                 gradient = (k1 * sdf1 + k2 * sdf2 + k3 * sdf3 + k4 * sdf4) / (4.0 * eps)
#                 if if_cal_hessian_x:
#                     # hessian = None
#                     assert sdf is not None  # computed when feed-forwarding through the network
#                     # the result of 4 taps is directly trace, but we assume they are individual components
#                     # so we use the same signature as 6 taps
#                     hessian_xx = ((sdf1 + sdf2 + sdf3 + sdf4) / 2.0 - 2 * sdf) / eps ** 2   # [N,1]
#                     hessian = torch.cat([hessian_xx, hessian_xx, hessian_xx], dim=-1) / 3.0
#                     return sdf, feat, gradient, hessian
#                 else:
#                     return sdf, feat, gradient
#             else:
#                 raise NotImplementedError
#
#     def set_active_levels(self, cur_step):
#         self.anneal_levels = min(max((cur_step - self.warm_up) // self.active_step, 1), self.num_levels)
#         self.active_levels = max(self.anneal_levels, self.init_active_level)
#
#     def set_normal_epsilon(self):
#         if self.enable_progressive: # normal_epsilon是grid Voxel边长的1/4
#             self.normal_epsilon = 2.0 *self.bound/ (self.grid_encoder.resolution_list[self.active_levels - 1] - 1)/4
#         else:
#             self.normal_epsilon = 2.0 *self.bound / (self.grid_encoder.resolution_list[-1] - 1)/4
#
#     def get_feature_mask(self, feature):
#         mask = torch.zeros_like(feature)
#         if self.enable_progressive:
#             mask[..., :(self.active_levels * self.per_level_dim)] = 1
#         else:
#             mask[...] = 1
#         return mask
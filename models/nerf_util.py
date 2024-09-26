import cv2
import numpy as np
import torch
import torch.nn.functional as torch_F
from torch.cuda.amp import autocast



# adapted from official implementing of neuralangelo
class MLPwithSkipConnection(torch.nn.Module):

    def __init__(self, layer_dims, skip_connection=[], activ=None, use_layernorm=False, use_weightnorm=False):
        """Initialize a multi-layer perceptron with skip connection.
        Args:
            layer_dims: A list of integers representing the number of channels in each layer.
            skip_connection: A list of integers representing the index of layers to add skip connection.
        """
        super().__init__()
        self.skip_connection = skip_connection
        self.use_layernorm = use_layernorm
        self.linears = torch.nn.ModuleList()
        if use_layernorm:
            self.layer_norm = torch.nn.ModuleList()
        layer_dim_pairs = list(zip(layer_dims[:-1], layer_dims[1:]))
        for li, (k_in, k_out) in enumerate(layer_dim_pairs):
            if li in self.skip_connection:
                k_in += layer_dims[0]
            linear = torch.nn.Linear(k_in, k_out)
            if use_weightnorm:
                linear = torch.nn.utils.weight_norm(linear)
            self.linears.append(linear)
            if use_layernorm and li != len(layer_dim_pairs) - 1:
                self.layer_norm.append(torch.nn.LayerNorm(k_out))
            if li == len(layer_dim_pairs) - 1:
                self.linears[-1].bias.data.fill_(0.0)
        self.activ = activ or torch_F.relu_

    def forward(self, input):
        feat = input
        for li, linear in enumerate(self.linears):
            if li in self.skip_connection:
                feat = torch.cat([feat, input], dim=-1)
            feat = linear(feat)
            if li != len(self.linears) - 1:
                if self.use_layernorm:
                    feat = self.layer_norm[li](feat)
                feat = self.activ(feat)
        return feat


def volume_rendering_alphas(densities, dists, dist_far=None):
    """The volume rendering function. Details can be found in the NeRF paper.
    Args:
        densities (tensor [batch,ray,samples,1]): The predicted volume density samples.
        dists (tensor [batch,ray,samples,1]): The corresponding distance samples.
        dist_far (tensor [batch,ray,1,1]): The farthest distance for computing the last interval.
    Returns:
        alphas (tensor [batch,ray,samples,1]): The occupancy of each sampled point along the ray (in [0,1]).
    """
    if dist_far is None:
        dist_far = torch.empty_like(dists[..., :1, :]).fill_(1e10)  # [B,R,1,1]
    dists = torch.cat([dists, dist_far], dim=2)  # [B,R,N+1,1]
    # Volume rendering: compute rendering weights (using quadrature).
    dist_intvs = dists[..., 1:, :] - dists[..., :-1, :]  # [B,R,N]
    sigma_delta = densities * dist_intvs  # [B,R,N]
    alphas = 1 - (-sigma_delta).exp_()  # [B,R,N]
    return alphas[..., 0]


def alpha_compositing_weights(alphas):
    """Alpha compositing of (sampled) MPIs given their RGBs and alphas.
    Args:
        alphas (tensor [batch,ray,samples]): The predicted opacity values.
    Returns:
        weights (tensor [batch,ray,samples,1]): The predicted weight of each MPI (in [0,1]).
    """
    alphas_front = torch.cat([torch.zeros_like(alphas[..., :1]),
                              alphas[..., :-1]], dim=-1)  # [B,R,N]
    with autocast(enabled=False):  # TODO: may be unstable in some cases.
        visibility = (1 - alphas_front).cumprod(dim=-1)  # [B,R,N]
    weights = (alphas * visibility)[..., None]  # [B,R,N,1]
    return weights


def composite(quantities, weights):
    """Composite the samples to render the RGB/depth/opacity of the corresponding pixels.
    Args:
        quantities (tensor [batch,ray,samples,k]): The quantity to be weighted summed.
        weights (tensor [batch,ray,samples,1]): The predicted weight of each sampled point along the ray.
    Returns:
        quantity (tensor [batch,ray,k]): The expected (rendered) quantity.
    """
    # Integrate RGB and depth weighted by probability.
    quantity = (quantities * weights).sum(dim=-2)  # [B,R,K]
    return quantity


def sample_points_in_sphere(radius, shape, device='cpu'):
    u = torch.empty(shape, device=device).uniform_(0, 1)
    theta = torch.empty(shape, device=device).uniform_(0, 2 * torch.pi)
    phi = torch.empty(shape, device=device).uniform_(0, torch.pi)

    # 立方根运算对应了球体体积增长的速率，因此使用 torch.pow(u, 1/3) 可以确保生成的点在球体内部均匀分布。
    x = radius * torch.pow(u, 1 / 3) * torch.sin(phi) * torch.cos(theta)
    y = radius * torch.pow(u, 1 / 3) * torch.sin(phi) * torch.sin(theta)
    z = radius * torch.pow(u, 1 / 3) * torch.cos(phi)

    points = torch.stack((x, y, z), dim=-1)
    return points


def sample_points_in_cube(side_length, shape, device='cpu'):
    # side_length: 边长
    # shape: 采样点数量
    points = torch.empty(list(shape)+[3], device=device).uniform_(-side_length / 2, side_length / 2)
    return points


def sample_neighbours_near_plane(n, p, device='cpu'):
    # n: (..., 3), p: (..., 3)
    # 在与法向垂直的切平面上采样邻近点
    pre_shape=p.shape[:-1]
    n,p=n.reshape(-1,3),p.reshape(-1,3)
    v = torch.randn_like(n, device=device)
    perpendicular = torch.cross(n, v,dim=1)  # 生成与n垂直的随机向量
    perpendicular = perpendicular / perpendicular.norm(dim=-1, keepdim=True)
    # 切点邻近点
    dist = torch.rand_like(n[:, :1], device=device) * 0.01
    neighbours = p + dist * perpendicular
    neighbours=neighbours.reshape(pre_shape+(3,))
    return neighbours

def load_K_Rt_from_P(P=None):
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def get_rays_batch_image(K, pose, h, w, sampling_idx=None):
    # K: (B, 4, 4), pose: (B, 4, 4), h: (B,), w: (B,) sampling_idx: (B, R) 或 None
    if sampling_idx is None:
        sampling_idx = torch.arange(h[0] * w[0], device=pose.device)[None].repeat(K.shape[0], 1)
    y_range = torch.arange(h[0], dtype=torch.float32, device=pose.device).add_(0.5)
    x_range = torch.arange(w[0], dtype=torch.float32, device=pose.device).add_(0.5)
    Y,X = torch.meshgrid(y_range, x_range, indexing='ij')
    Y = Y.ravel()[None].tile(K.shape[0], 1) # (B, h*w)
    X = X.ravel()[None].tile(K.shape[0], 1) # (B, h*w)
    y = torch.gather(Y, 1, sampling_idx) # (B, R)
    x = torch.gather(X, 1, sampling_idx) # (B, R)

    fx, fy, cx, cy = K[:,0, 0][:,None], K[:,1, 1][:,None], K[:,0, 2][:,None], K[:,1, 2][:,None] # (B, 1)
    sk = K[:,0, 1][:,None] # skew: 坐标轴倾斜参数, 理想情况下应该为0
    y_cam = (y - cy) / fy * 1.
    x_cam = (x - cx -sk*y_cam) / fx * 1. # 把cam->pixel的转换写出来即可得到这个式子
    z_cam = torch.ones_like(x_cam)  # 假设z=1
    d = torch.stack([x_cam, y_cam, z_cam], dim=-1) # (B,R,3)
    d = d/torch.norm(d,dim=-1,keepdim=True)
    depth_scale = d[:,:, 2:] # 即z轴余弦角cosθ，用于计算depth
    d=torch.bmm(pose[:, :3, :3], d.permute(0, 2, 1)).permute(0, 2, 1) # c2w
    o = pose[:,:3, -1].unsqueeze(1).tile(1,d.shape[1], 1)

    return o, d, depth_scale

def sample_dists(ray_size, dist_range, intvs, stratified, device="cuda"):
    """Sample points on ray shooting from pixels using distance.
    Args:
        ray_size (int [2]): Integers for [batch size, number of rays].
        range (float [2]): Range of distance (depth) [min, max] to be sampled on rays.
        intvs: (int): Number of points sampled on a ray.
        stratified: (bool): Use stratified sampling or constant 0.5 sampling.
    Returns:
        dists (tensor [batch_size, num_ray, intvs, 1]): Sampled distance for all rays in a batch.
    """
    batch_size, num_rays = ray_size
    dist_min, dist_max = dist_range
    if stratified:
        rands = torch.rand(batch_size, num_rays, intvs, 1, device=device)
    else:
        rands = torch.empty(batch_size, num_rays, intvs, 1, device=device).fill_(0.5)
    rands += torch.arange(intvs, dtype=torch.float, device=device)[None, None, :, None]  # [B,R,N,1]
    dists = rands / intvs * (dist_max - dist_min) + dist_min  # [B,R,N,1]
    return dists

def sample_dists_from_pdf(bin, weights, intvs_fine):
    """Sample points on ray shooting from pixels using the weights from the coarse NeRF.
    Args:
        bin (tensor [..., intvs]): bins of distance values from the coarse NeRF.
        weights (tensor [..., intvs-1]): weights from the coarse NeRF.
        intvs_fine: (int): Number of fine-grained points sampled on a ray.
    Returns:
        dists (tensor [batch_size, num_ray, intvs, 1]): Sampled distance for all rays in a batch.
    """
    pdf = torch_F.normalize(weights, p=1, dim=-1)
    # Get CDF from PDF (along last dimension).
    cdf = pdf.cumsum(dim=-1)  # [...,N-1]
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # [...,N]
    # Take uniform samples.
    grid = torch.linspace(0, 1, intvs_fine + 1, device=pdf.device)  # [Nf+1]
    unif = 0.5 * (grid[:-1] + grid[1:]).repeat(*cdf.shape[:-1], 1)  # [...,Nf]
    idx = torch.searchsorted(cdf, unif, right=True)  # [...,Nf] \in {1...N}
    # Inverse transform sampling from CDF.
    low = (idx - 1).clamp(min=0)  # [...,Nf]
    high = idx.clamp(max=cdf.shape[-1] - 1)  # [...,Nf]
    dist_min = bin.gather(dim=-1, index=low)  # [...,Nf]
    dist_max = bin.gather(dim=-1, index=high)  # [...,Nf]
    cdf_low = cdf.gather(dim=-1, index=low)  # [...,Nf]
    cdf_high = cdf.gather(dim=-1, index=high)  # [...,Nf]
    # Linear interpolation.
    t = (unif - cdf_low) / (cdf_high - cdf_low + 1e-8)  # [...,Nf]
    dists = dist_min + t * (dist_max - dist_min)  # [...,Nf]
    return dists # [...,Nf]

def near_far_from_sphere(rays_o, rays_d, bound):
    # sphere的bound指半径
    # intersect with sphere
    o_projection = torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
    h_square = torch.sum(rays_o * rays_o, dim=-1, keepdim=True) - o_projection ** 2
    # (h_square - bound ** 2) >= 0, means no intersection or tangent
    edge = (bound ** 2 - h_square).sqrt()

    far = edge - o_projection
    near = -edge - o_projection
    outside = far.isnan()
    return near, far, outside


def near_far_from_cube(rays_o, rays_d, bound):
    # cube的bound指半边长
    # intersect with cube
    tmin = (-bound - rays_o) / (rays_d + 1e-15)  # [B, N, 3]
    tmax = (bound - rays_o) / (rays_d + 1e-15)
    near = torch.where(tmin < tmax, tmin, tmax).max(dim=-1, keepdim=True)[0]
    far = torch.where(tmin > tmax, tmin, tmax).min(dim=-1, keepdim=True)[0]
    # if far <= near, means no intersection seg, set both near and far to inf (1e9 here)
    outside = far < near
    return near, far, outside

def to_pure_quat(q):
    # q:(...,3)
    # 把一个四元数转化为纯四元数
    return torch.cat([torch.zeros_like(q[..., :1]), q], dim=-1)

def quat_conj(q):
    # q:(...,4)
    # 四元数共轭
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

def normalize_quat(q):
    # q:(...,4)
    # 归一化四元数
    return q / (1e-6+torch.norm(q, dim=-1, keepdim=True))

def quat_mult(Q, P):
    """
    四元数乘法: 对于标准的两个四元数Q(4,)=(q0,q)、P(4,)=(p0,p)
    QP=(p0q0-p·q, p0q+q0p+q×p) [ ×为叉乘 ]

    :param Q: (...,4)
    :param P: (...,4)
    :return:
    """
    mul_w = Q[..., :1] * P[..., :1] - torch.sum(Q[..., 1:] * P[..., 1:], dim=-1, keepdim=True)
    mul_xyz = Q[..., :1] * P[..., 1:] + P[..., :1] * Q[..., 1:] + torch.cross(Q[..., 1:], P[..., 1:], dim=-1)
    return torch.cat([mul_w, mul_xyz], dim=-1)
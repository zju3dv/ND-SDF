import torch
from torch import nn
from functools import partial
from models.spherical_harmonics import get_spherical_harmonics

# position encoding, 增强模型对xyz、direction的感知
# x-> (x,sin(2^k*x),co(2^k*x)...), shape:(N_freqs+1,)
# 一般在log_sampling且max_log_freq取N_freqs-1,从而freqs=[2^0,2^1,...,2^(N_freqs-1)]
class FourierEncoder(nn.Module):
    def __init__(self, d_in, max_log_freq, N_freqs, log_sampling=True):
        """
        position embedding

        :param d_in: input dimension
        :param max_log_freq: log2(maximum frequency)
        :param N_freqs: number of frequencies
        :param log_sampling: whether to sample in log space
        """
        super().__init__()
        self.d_in = d_in
        self.d_out = d_in * 2 * N_freqs
        self.funcs = [torch.sin, torch.cos]
        if log_sampling:
            self.freqs = 2 ** torch.linspace(0, max_log_freq, N_freqs)
        else:
            self.freqs = torch.linspace(1, 2 ** max_log_freq, N_freqs)

    def forward(self, x):
        """

        :param x: (...,D)
        :return: (...,D*2*N_freqs)
        """
        res = []
        for freq in self.freqs:
            for func in self.funcs:
                res.append(func(freq * x))
        return torch.cat(res,dim=-1)

    def encoded_length(self):
        return self.d_out


# 基于手动设置球谐系数的encoding
# 也是增强感知。
class SphericalEncoder(nn.Module):
    def __init__(self,levels=3):
        super().__init__()
        self.levels=levels
        self.spherical_coder=partial(get_spherical_harmonics,levels=levels)

    def forward(self,rays_d):
        return self.spherical_coder(rays_d)
    def encoded_length(self):
        return (self.levels+1)**2


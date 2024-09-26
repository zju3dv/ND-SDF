# S3IM loss, adapted from https://github.com/Madaoer/S3IM-Neural-Fields
from .ssim import SSIM
import torch



class S3IM(torch.nn.Module):
    r"""Implements Stochastic Structural SIMilarity(S3IM) algorithm.
    It is proposed in the ICCV2023 paper  
    `S3IM: Stochastic Structural SIMilarity and Its Unreasonable Effectiveness for Neural Fields`.

    Arguments:
        kernel_size (int): kernel size in ssim's convolution(default: 4)
        stride (int): stride in ssim's convolution(default: 4)
        repeat_time (int): repeat time in re-shuffle virtual patch(default: 10)
        patch_height (height): height of virtual patch(default: 64)
        patch_width (height): width of virtual patch(default: 64)
    """
    def __init__(self, kernel_size=4, stride=4, repeat_time=10):
        super(S3IM, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.repeat_time = repeat_time
        self.ssim_loss = SSIM(window_size=self.kernel_size, stride=self.stride)
    def forward(self, src_vec, tar_vec, patch_size):
        # src_vec:[B, R, c]
        # tar_vec:[B, R, c]
        # patch_size:[h, w]
        B, R = src_vec.shape[:2]
        index_list = []
        for i in range(self.repeat_time):
            if i == 0:
                tmp_index = torch.arange(R)
                index_list.append(tmp_index)
            else:
                ran_idx = torch.randperm(R)
                index_list.append(ran_idx)
        res_index = torch.cat(index_list)
        tar_all = tar_vec[:,res_index]
        src_all = src_vec[:,res_index]
        tar_patch = tar_all.permute(0, 2, 1).reshape(B, 3, patch_size[0], patch_size[1] * self.repeat_time)
        src_patch = src_all.permute(0, 2, 1).reshape(B, 3, patch_size[0], patch_size[1] * self.repeat_time)
        loss = (1 - self.ssim_loss(src_patch, tar_patch))
        return loss









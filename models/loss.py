import time

import cv2
import torch
import torch.nn as nn
from models.metrics.s3im import S3IM
from models.nerf_util import quat_mult,quat_conj,to_pure_quat

# TODO: 这里计算scale和shift应该分batch各自计算，只考虑了单个batch的情况 [√]
def compute_scale_and_shift(pred, target, mask=None):
    # 标量之间求解线性回归最小二乘解系数即：min||s*pred+d-target||
    # pred、target: (B, ...), (B, ...), tensor
    # 构造最小二乘：min L=∑|Aix-bi|^2, A=[pred, 1], x=[s,t]^T为求解系数,b则为[target]
    # 得A^TAx=A^Tb,求解即可
    if mask is not None:
        pred, target = pred * mask.float(), target * mask.float()
    else:
        mask = torch.ones_like(pred)
    dim = tuple(range(1, pred.dim()))
    a00 = (pred ** 2).sum(dim)
    a01 = pred.sum(dim)
    a10 = a01
    a11 = mask.float().sum(dim)

    b00 = (pred * target).sum(dim)
    b10 = target.sum(dim)
    b = torch.stack([b00, b10], dim=-1)[..., None]  # (B, 2, 1)
    # 手解线性方程组，对于Ax=b, 直接x=A^-1b, A^-1=A*/det(A)
    det = a00 * a11 - a01 * a10 # (B, )
    det=det.unsqueeze(-1).unsqueeze(-1) # (B, 1, 1)
    adjoint = torch.stack([a11, -a10, -a01, a00], dim=-1).reshape(pred.shape[0], 2, 2)
    valid = torch.nonzero(det.ravel()) # (n, 1), 第二维指det有多少维，标量因此为1
    valid = valid.ravel()
    x=torch.zeros_like(b)
    x[valid] = torch.bmm(adjoint[valid] / det[valid], b[valid]) # (B, 2, 1)
    return x


def get_psnr(pred, gt, mask=None):
    mse_signal = mse_loss(pred, gt, mask)
    return -10 * torch.log10(mse_signal)

def gradient_loss(prediction, target, mask=None, weight=None):
    # prediction，target: (B, H, W, 1)，估计图像和真实图像的patch（但实际做的时候H,W的patch像素点都是随机、乱序的，互相没有关系，计算grad意义不明确，但确实有效。）
    # mask: (B, H, W, 1)
    if mask is None:
        mask = torch.ones_like(prediction)
    M = mask.float().sum()
    if weight is not None:
        mask = mask * weight

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    if M==0:
        return torch.tensor(0.0, device=mask.device)
    return image_loss.sum() / M

def mse_loss(pred, gt, mask=None, weight=None):
    # mask: (B, R, 1), weight: (B, R, 1)
    error = (pred - gt) ** 2  # [B,R,C]
    error = error.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    if weight is not None:
        error = error * weight
    if mask is not None:
        M=mask.float().sum()
        if M==0:
            return torch.tensor(0.0, device=mask.device)
        return (error * mask.float()).sum()/M/error.shape[-1] # ÷C
    else:
        return error.mean()


def l1_loss(pred, gt, mask=None, weight=None):
    # pred:(...,C), gt:(...,C), mask:(...,1), weight:(...,1)
    error = (pred - gt).abs()  # [B,R,C]
    error = error.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    if weight is not None: # uncertainty越大, normal越不可信, rgb权重越大
        # weight = torch.minimum(0.5 + uncertainty, torch.ones_like(uncertainty))
        error = error * weight
    if mask is not None:
        M = mask.float().sum()
        if M == 0:
            return torch.tensor(0.0, device=mask.device)
        return (error * mask.float()).sum()/M/error.shape[-1]
    return error.mean()


def eikonal_loss(gradients, mask=None):
    # eikonal-loss规范sdf
    # gradients: (B,R,N_eik,3), mask:(B,R,1)
    # 一般gradient:(...,3), mask:None
    error = (gradients.norm(dim=-1, keepdim=False) - 1.0) ** 2 # [B,R,N_eik]
    error = error.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    if mask is not None:
        M = mask.float().sum()
        if M == 0:
            return torch.tensor(0.0, device=mask.device)
        return (error * mask.float()).sum()/M/error.shape[-1]
    else:
        return error.mean()


# angelo提出的曲率loss：最小化hessian矩阵和的绝对值。
def curvature_loss(hessian, mask=None, device='cuda'):
    # hessian: (B, R, N_hessian, 3), mask: (B, R, 1)
    if hessian is None:
        return torch.tensor(0.0, device=device)
    laplacian = hessian.sum(dim=-1).abs()  # [B,R,N_hessian]
    laplacian = laplacian.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)  # [B,R,N_hessian]
    if mask is not None:
        M=mask.float().sum()
        if M==0:
            return torch.tensor(0.0, device=mask.device)
        return (laplacian * mask.float()).sum()/M/laplacian.shape[-1]
    else:
        return laplacian.mean()


def smooth_loss(g1, g2, mask=None, weight=None):
    # g1、g2分别是场景中邻居点的sdf gradient:(B,R,N_smooth,3), mask、weight:(B,R,1)
    if g2 is None or g1 is None:
        return torch.tensor(0.0, device=mask.device)
    normals_1 = g1 / (g1.norm(2, dim=-1,keepdim=True) + 1e-6)
    normals_2 = g2 / (g2.norm(2, dim=-1,keepdim=True) + 1e-6)
    smooth_error = torch.norm(normals_1 - normals_2, dim=-1,keepdim=False) # [B,R,N_smooth]
    smooth_error = smooth_error.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    if weight is not None:
        smooth_error = smooth_error * weight
    if mask is not None:
        M = mask.float().sum()
        if M == 0:
            return torch.tensor(0.0, device=mask.device)
        return (smooth_error * mask.float()).sum()/M/smooth_error.shape[-1]
    return smooth_error.mean()

def adaptive_smooth_loss(angle, g1, g2, mask=None):
    # angle:(B,R,1), g1、g2:(B,R,N_smooth,3), mask:(B,R,1)
    w_smooth = 1-smooth_exp_confidence(angle, beta=12.5, threshold_angle=15/180*torch.pi)  # (B,R,1) smooth choice=1-biased_confidence：(12.5, 15)： [angle, 1-confidence]有[0, 0.965],[5,0.9], [15,0.5],[25,0.1]。
    return smooth_loss(g1, g2, mask=mask, weight=w_smooth)

def normal_loss(normal_pred, normal_gt, mask=None, weight=None):
    # normal_pred, normal_gt: (B, R, 3), mask: (B, R, 1), weight: (B, R, 1)
    normal_gt = torch.nn.functional.normalize(normal_gt, p=2, dim=-1)
    normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1)
    l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1, keepdim=True)
    cos = (1. - torch.sum(normal_pred * normal_gt, dim=-1, keepdim=True))
    l1 = l1.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    cos = cos.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    if weight is not None:
        l1, cos = l1 * weight, cos * weight
    if mask is not None:
        M=mask.float().sum()
        if M==0:
            return torch.tensor(0.0, device=mask.device), torch.tensor(0.0, device=mask.device)
        return (l1 * mask.float()).sum()/M, (cos * mask.float()).sum()/M
    return l1.mean(), cos.mean()

def depth_loss(depth_pred, depth_gt, mask=None, weight=None, monocular=True, cal_grad_loss=True, patch_size=(32, 32)):
    # 单目/真实深度监督
    # depth_pred, depth_gt: (B, R, 1), mask: (B, R, 1)
    if monocular:
        scale_shift = compute_scale_and_shift(depth_pred, depth_gt, mask=mask) # (B, 2, 1)
        scale, shift = scale_shift[:, :1, :], scale_shift[:, 1:, :]
        depth_pred = scale * depth_pred + shift
    loss = mse_loss(depth_pred, depth_gt, mask=mask, weight=weight)
    if cal_grad_loss: # depth gradient loss
        depth_pred = depth_pred.reshape(-1, *patch_size,1)
        depth_gt = depth_gt.reshape(-1, *patch_size,1)
        if mask is not None:
            mask = mask.reshape(-1, *patch_size,1)
        if weight is not None:
            weight = weight.reshape(-1, *patch_size,1)
        loss = loss+ 0.5*gradient_loss(depth_pred, depth_gt, mask=mask, weight=weight)
    return loss

def adaptive_depth_loss(angle, depth_pred, depth_gt, mask=None, monocular=True, cal_grad_loss=True, patch_size=(32, 32)):
    # angle:(B,R,1), depth_pred, depth_gt: (B, R, 1), mask: (B, R, 1)
    w_depth = 1-smooth_exp_confidence(angle, beta=12.5, threshold_angle=15/180*torch.pi)  # (B,R,1) depth choice=1-biased_confidence：(25, 10)： [angle, 1-confidence]有[0, 0.99], [5,0.9], [10,0.5], [15,0.1], [20, 0.02]
    loss = depth_loss(depth_pred, depth_gt, mask=mask, weight=w_depth, monocular=monocular, cal_grad_loss=cal_grad_loss, patch_size=patch_size)
    return loss

# 1. n aligned with scene normal
def nb_regu_loss(quat, normal_w, mask=None):
    # quat:(B,R,4), normal:(B,R,3),  mask:(...,1)
    # both normal_w and quat are at world space
    normal_w = torch.nn.functional.normalize(normal_w, p=2, dim=-1)# .detach() # FIXME：是否detach？实测不detach更稳定。
    half_theta = torch.acos(quat[..., :1]) # (...,1)
    n = quat[..., 1:] / (1e-15 + torch.sin(half_theta)) # (...,3)
    regu_theta = torch.abs(half_theta)
    regu_n = (1-(normal_w*n).sum(dim=-1, keepdim=True).abs())
    regu_theta = regu_theta.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    regu_n = regu_n.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    if mask is not None:
        M=mask.float().sum()
        if M==0:
            return torch.tensor(0.0, device=mask.device)
        return (regu_n * mask.float()).sum()/M + (regu_theta * mask.float()).sum()/M
    return regu_n.mean() + regu_theta.mean()

# 2. n aligned with mono normal
def nb_regu_loss2(quat, normal_gt, pose, mask=None):
    # quat:(B,R,4), normal_gt:(B,R,3), pose:(B,4,4), mask:(...,1)
    # both normal_w and quat are at world space
    R_c2w = pose[:, :3, :3]
    normal_gt = torch.nn.functional.normalize(normal_gt, p=2, dim=-1) # camera space
    normal_gt = torch.bmm(R_c2w, normal_gt.permute(0, 2, 1)).permute(0, 2, 1) # mono normal to world space
    half_theta = torch.acos(quat[..., :1]) # (...,1)
    n = quat[..., 1:] / (1e-15 + torch.sin(half_theta)) # (...,3)
    regu_theta = torch.abs(half_theta)
    regu_n = (1-(normal_gt*n).sum(dim=-1, keepdim=True).abs())
    regu_theta = regu_theta.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    regu_n = regu_n.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    if mask is not None:
        M=mask.float().sum()
        if M==0:
            return torch.tensor(0.0, device=mask.device)
        return (regu_n * mask.float()).sum()/M + (regu_theta * mask.float()).sum()/M
    return regu_n.mean() + regu_theta.mean()

# 3. directly regularize angle between normal_w and biased_normal_w
def nb_regu_loss3(quat, normal_w, mask=None):
    # normal_w:(B,R,3), biased_normal_w:(B,R,3), mask:(...,1)
    normal_w = torch.nn.functional.normalize(normal_w, p=2, dim=-1).detach() # detach gradient
    biased_normal_w = quat_mult(quat_mult(quat, to_pure_quat(normal_w)), quat_conj(quat))[..., 1:] # (B,R,3)
    angle = torch.acos((normal_w * biased_normal_w).sum(dim=-1, keepdim=True).clip(-1,1)) # (B,R,1), [0,π]
    angle = angle.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    if mask is not None:
        M=mask.float().sum()
        if M==0:
            return torch.tensor(0.0, device=mask.device)
        return (angle * mask.float()).sum()/M
    return angle.mean()

def angle2weight(angle, re_weight_params=None):
    # angle:(B,R,1)
    def scale_shift_sigmoid_func(angle, params=None):
        # angle ∈ [0, π]
        # y=1+1/(1+e^(-beta*(x-angle_threshold)))*times ∈ [1,1+times]
        # [choice-3 best smooth]     ：  [beta, angle_threshold, times] = [8, 30/180*np.pi, 0.5]：    y( 5°)≈1.02, y(10°)≈1.03, y(15°)≈1.06, y(20°)≈1.1, y(30°)≈1.25, y(40°)=1.4
        # [choice-2 very smooth]     ：  [beta, angle_threshold, times] = [8, 30/180*np.pi, 1]：    y( 5°)≈1.03, y(10°)≈1.06, y(15°)≈1.11, y(20°)≈1.2, y(30°)≈1.5, y(40°)=1.8
        # [choice-1 more smooth]     ：  [beta, angle_threshold, times] = [12.5, 30/180*np.pi, 1]： y( 5°)≈1.00, y(10°)≈1.01, y(15°)≈1.04, y(20°)≈1.1, y(30°)≈1.5, y(40°)=1.9
        # [choice0 smooth]           ：  [beta, angle_threshold, times] = [20, 15/180*np.pi, 1]：   y( 5°)≈1.03, y(10°)≈1.15, y(15°)≈1.5, y(20°)≈1.85, y(30°)≈2
        # [choice1 abrupt]           ：  [beta, angle_threshold, times] = [25, 15/180*np.pi, 2]：   y( 5°)≈1.02, y(10°)≈1.2,  y(15°)=2  , y(20°)≈2.9,  y(30°)=3
        # [choice2 more abrupt+]     ： [beta, angle_threshold, times] = [25, 15/180*np.pi, 4]：    y( 5°)≈1.05, y(10°)≈1.4,  y(15°)=3  , y(20°)≈4.6,  y(30°)≈5.0
        # [choice3 more abrupt++]    ：[beta, angle_threshold, times] = [50, 15/180*np.pi, 9]：     y(10°)≈1.1,  y(15°)=5.5, y(20°)≈9.9,  y(30°)≈10
        beta = 25
        angle_threshold = 15/180*torch.pi
        times = 2
        if params is not None:
            beta, angle_threshold, times = params
            angle_threshold = angle_threshold/180*torch.pi
        return 1.0+1/(1+(-beta*(angle-angle_threshold)).exp())*times
    return scale_shift_sigmoid_func(angle, re_weight_params) # (B,R,1)∈[1,2]

def smooth_exp_confidence(angle, beta=12.5, threshold_angle=15/180*torch.pi):
    # 5°:0.087rad, 15°:0.261rad, 20°:0.349rad, 25°:0.436rad, 30°:0.523rad
    # choice1[更bias]：     (25,  5): [angle, confidence]有[0, 0.1], [5,0.5], [15,0.99]。
    # choice2[更smooth]：   (12.5, 15): [angle, confidence]有[0, 0.035],[5,0.1], [15,0.5], [25,0.9]。 该smooth setting配0.025 abg和0.05depth，1.0 rgb不够detail
    # choice4[更更smooth]:  (12.5, 20): [angle, confidence]有[0, 0.013],[5,0.04],[20,0.5], [30,0.9]。
    # choice5[超级smooth]:  (8,    25): [angle, confidence]有[0, 0.03 ],[5,0.06],[25,0.5], [35,0.8],[40,0.9]。
    # choice6[超超smooth]:  (8,    30): [angle, confidence]有[0, 0.02 ],[5,0.03],[25,0.33],[30,0.5],[40,0.8],[50,0.94]。
    # depth choice=1-biased_confidence：(25, 10)： [angle, 1-confidence]有[0, 0.99], [5,0.9], [10,0.5], [15,0.1]。
    return 1/(1+(-beta*(angle-threshold_angle)).exp())

def angle2confidence(angle):
    # angle: (...) ∈ [0, π]
    # func: angle to biased normal confidence
    # TODO: 还可以根据angle>angle.mean()来加权 [?]
    def two_step_confidence(angle):
        # angle ∈ [0, π]
        two_steps = [5/180*torch.pi, 30/180*torch.pi]
        confidence = torch.zeros_like(angle)
        confidence[angle<two_steps[0]]=0.
        confidence[(two_steps[0]<=angle) & (angle<two_steps[1])]=(angle[(two_steps[0]<=angle) & (angle<two_steps[1])]-two_steps[0])/(two_steps[1]-two_steps[0])
        confidence[angle>=two_steps[1]]=1.
        return confidence
        # if angle < two_steps[0]:
        #     return 0.
        # elif angle < two_steps[1]:
        #     return (angle - two_steps[0]) / (two_steps[1] - two_steps[0])
        # else:
        #     return 1.
    def abrupt_exp_confidence(angle):
        beta = 100 # 抖度
        threshold_angle=5/180*torch.pi # 认为biased threshold_angle°后wrong prior置信度极速上升
        # threshold_angle = angle.mean()
        return 1/(1+(-beta*(angle-threshold_angle)).exp())
    
    return smooth_exp_confidence(angle, beta=12.5, threshold_angle=15/180*torch.pi) # smooth choice

def adaptive_biased_normal_loss(quat, normal_w, biased_normal_w, normal_gt, pose, angle=None, mask=None):
    # quat:(B,R,4), normal_w:(B,R,3), biased_normal_w:(B,R,3), normal_gt:(B,R,3), pose:(B,4,4), mask:(B,R,1)
    R_c2w = pose[:, :3, :3]
    normal_gt_w = torch.bmm(R_c2w, normal_gt.permute(0, 2, 1)).permute(0, 2, 1) # mono normal to world space
    normal_gt_w = torch.nn.functional.normalize(normal_gt_w, p=2, dim=-1)
    normal_w = torch.nn.functional.normalize(normal_w, p=2, dim=-1)
    biased_normal_w = torch.nn.functional.normalize(biased_normal_w, p=2, dim=-1)
    half_theta = torch.acos(quat[..., :1]) # (...,1)
    n = quat[..., 1:] / (1e-15 + torch.sin(half_theta)) # (...,3)
    w_biased = ...  # (B,R,1),
    w_origin = ...  # (B,R,1),

    # TODO: 根据normal_w 和 n的夹角，还是half_theta来，或者是旋转角度即normal_w和biased_normal_w的夹角来计算confidence？目前是normal_w和biased_normal_w的夹角
    if angle is None:
        with torch.no_grad():
            # detach gradient
            angle = torch.acos((normal_w * biased_normal_w).sum(dim=-1, keepdim=True).clip(-1,1)) # (B,R,1), [0,180°],angle>5度基本可确定为高置信度wrong prior need biased normal
    w_biased = angle2confidence(angle) # (B,R,1)
    w_origin = (1 - w_biased) # (B,R,1)
    # FIXME: detach >60° supervision
    # w_biased[angle>60/180*torch.pi]=1e-15
    # w_origin[angle>60/180*torch.pi]=1e-15
    ########################################################################################################################

    l1 = torch.abs(normal_w - normal_gt_w).sum(dim=-1, keepdim=True) # (B,R,1)
    cos = (1. - torch.sum(normal_w * normal_gt_w, dim=-1, keepdim=True)) # (B,R,1)
    l1_biased = torch.abs(biased_normal_w - normal_gt_w).sum(dim=-1, keepdim=True) # (B,R,1)
    cos_biased = (1. - torch.sum(biased_normal_w * normal_gt_w, dim=-1, keepdim=True)) # (B,R,1)

    l1 = l1.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    cos = cos.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    l1_biased = l1_biased.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    cos_biased = cos_biased.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    if mask is not None:
        M=mask.float().sum()
        if M==0:
            return torch.tensor(0.0, device=mask.device), torch.tensor(0.0, device=mask.device), torch.tensor(0.0, device=mask.device), torch.tensor(0.0, device=mask.device)
        l1_mean = (l1 * mask.float()*w_origin).sum()/M
        cos_mean = (cos * mask.float()*w_origin).sum()/M
        l1_biased_mean = (l1_biased * mask.float()*w_biased).sum()/M
        cos_biased_mean = (cos_biased * mask.float()*w_biased).sum()/M
        return l1_mean, cos_mean, l1_biased_mean, cos_biased_mean
    return (l1 * w_origin).mean(), (cos * w_origin).mean(), (l1_biased * w_biased).mean(), (cos_biased * w_biased).mean()
    pass

# loss概览：
# 1. eikonal_loss:
# 2. color_loss:
# 3. smooth_loss/curvature loss:
# 4. normal_loss:
# *5. depth_loss:
# *6. l2 osf mask loss
# *7. l3 osf loss
# *8. mvs refine osf loss
class ImplicitReconLoss(nn.Module):
    def __init__(self,
                 lambda_rgb_l1=1.0,
                 lambda_rgb_mse=0.0,
                 lambda_s3im=0.0,
                 lambda_eik=0.05,
                 lambda_smooth=0.005,
                 lambda_normal=0.0,
                 lambda_depth=0.0,
                 lambda_curvature=0.0,
                 lambda_pts_sdf_l1=0.0,
                 lambda_pts_normal=0.0,
                 lambda_nb_regu=0.0,
                 lambda_ab_normal=0.0,
                 lambda_ab_depth=0.0,
                 lambda_ab_smooth=0.0,
                 optim_conf=None):
        super().__init__()
        def sequential_lr(progress, lr):
            # 1. lr: float
            # 2. lr: list[4]: (start_progress, end_progress, start_lr, end_lr)
            if isinstance(lr, (int, float)):
                return lr
            return lr[2] + min(1.0, max(0.0, (progress - lr[0]) / (lr[1] - lr[0]))) * (lr[3] - lr[2])
        from functools import partial
        self.lambda_rgb_l1 = partial(sequential_lr, lr=lambda_rgb_l1)
        self.lambda_rgb_mse = partial(sequential_lr, lr=lambda_rgb_mse)
        self.lambda_s3im = partial(sequential_lr, lr=lambda_s3im)
        self.lambda_eik = partial(sequential_lr, lr=lambda_eik)
        self.lambda_smooth = partial(sequential_lr, lr=lambda_smooth)
        self.lambda_normal = partial(sequential_lr, lr=lambda_normal)
        self.lambda_depth = partial(sequential_lr, lr=lambda_depth)
        # self.lambda_curvature = partial(sequential_lr, lr=lambda_curvature)
        self.lambda_curvature = lambda_curvature # [CHANGE-4]
        self.lambda_pts_sdf_l1 = partial(sequential_lr, lr=lambda_pts_sdf_l1)
        self.lambda_pts_normal = partial(sequential_lr, lr=lambda_pts_normal)
        self.lambda_nb_regu = partial(sequential_lr, lr=lambda_nb_regu)
        self.lambda_ab_normal = partial(sequential_lr, lr=lambda_ab_normal)
        self.lambda_ab_depth = partial(sequential_lr, lr=lambda_ab_depth)
        self.lambda_ab_smooth = partial(sequential_lr, lr=lambda_ab_smooth)

        self.init_lambda_curvature = lambda_curvature

        self.warm_up_end = getattr(optim_conf.sched, 'warm_up_end', 2000)
        self.anneal_quat_end = getattr(optim_conf.sched, 'anneal_quat_end', 0.2)
        self.if_reweight = getattr(optim_conf.sched, 'if_reweight', False)
        self.re_weight_params = getattr(optim_conf.sched, 're_weight_params', [25,15,2])
        self.patch_size = (32, 32) # patch size is applied in depth_loss or s3im_loss
        self.torch_l1=torch.nn.L1Loss()
        self.s3im=S3IM()

    def set_curvature_weight(self, cur_step, anneal_levels, grow_rate):
        # 1.38098是grid res指数增长系数
        sched_weight = self.init_lambda_curvature
        if cur_step <= self.warm_up_end:
             sched_weight *= cur_step / self.warm_up_end
        else:
            decay_factor = grow_rate ** (anneal_levels - 1)
            sched_weight /= decay_factor
        self.lambda_curvature = sched_weight

    def set_patch_size(self, num_rays=1024):
        a = [i for i in range(1,int(num_rays**0.5)+1) if num_rays%i==0][-1]
        self.patch_size = (a, num_rays//a)

    def forward(self, output, sample, prog):
        # prog: [0,1] 代表百分比训练进度用来控制sequential_lr
        outside = output['outside'] # (B, R, 1) out of scene_aabb
        mask = sample['mask'] # prior mask

        if output.get('rays_fg',None) is not None: # occ enabled
            foreground_mask = output['rays_fg'] # (B, R, 1)
        else:
            sdf = output['sdf']
            foreground_mask = (sdf > 0.).any(dim=-2) & (sdf < 0.).any(dim=-2) # foreground mask is scene mask

        # TODO：biased angle guided rgb loss
        if_angle_guided_weighting = output.get('angle', None) is not None and self.if_reweight
        angle_guided_weight=None
        anneal_start_guided_weighting = self.anneal_quat_end
        anneal_end_regu = 0.0
        if if_angle_guided_weighting and prog>anneal_start_guided_weighting:
            angle_guided_weight = angle2weight(output['angle'], self.re_weight_params) # (B,R,1)
            angle_guided_weight[~foreground_mask] = 1.0  # only for foreground sdf scene

        # Accumulate loss
        losses = {}
        # 1. eikonal_loss
        loss_eik = eikonal_loss(output['gradient_eik'],mask=None) if output.get('gradient_eik', None) is not None else torch.tensor(0.0, device=mask.device)
        losses['eik'] = loss_eik
        loss= self.lambda_eik(prog) * loss_eik
        # 2. l1 rgb
        if self.lambda_rgb_l1(prog) > 0:
            loss_rgb_l1 = l1_loss(output['rgb'], sample['rgb'], mask=(~outside), weight=angle_guided_weight)
            loss += self.lambda_rgb_l1(prog) * loss_rgb_l1
            losses['rgb_l1'] = loss_rgb_l1
        # 3. mse rgb
        if self.lambda_rgb_mse(prog) > 0:
            loss_rgb_mse = mse_loss(output['rgb'], sample['rgb'], mask=(~outside))
            loss += self.lambda_rgb_mse(prog) * loss_rgb_mse
            losses['rgb_mse'] = loss_rgb_mse
        # 4. smooth_loss
        if self.lambda_smooth(prog) > 0:
            loss_smooth = smooth_loss(output['gradient_smooth'], output['gradient_smooth_neighbor'], mask=(~outside)&foreground_mask)
            loss += self.lambda_smooth(prog) * loss_smooth
            losses['smooth'] = loss_smooth
        # 5. normal_loss
        if self.lambda_normal(prog) > 0:
            loss_normal_l1, loss_normal_cos = normal_loss(output['normal'], sample['normal'], mask=(~outside)&foreground_mask&mask) # bg或者object的平整区域
            loss += self.lambda_normal(prog) * (loss_normal_l1 + loss_normal_cos)
            losses['normal_l1'] = loss_normal_l1
            losses['normal_cos'] = loss_normal_cos
        # 6. depth_loss
        if self.lambda_depth(prog) > 0:
            loss_depth = depth_loss(output['depth'], 50*sample['depth']+0.5, mask=(~outside)&foreground_mask&mask, monocular=True, patch_size=self.patch_size)
            loss += self.lambda_depth(prog) * loss_depth
            losses['depth'] = loss_depth
        # 7. curvature_loss
        if self.lambda_curvature > 0 and 'hessian' in output:
            loss_curvature = curvature_loss(output['hessian'], mask=None)
            loss += self.lambda_curvature * loss_curvature
            losses['curvature'] = loss_curvature
        # 8. pts_sdf_l1
        if self.lambda_pts_sdf_l1(prog) > 0 and 'pts' in sample:
            loss_pts_sdf_l1 = l1_loss(output['pts_sdf'], torch.zeros_like(output['pts_sdf']), mask=sample['pts_confidence'])
            loss += self.lambda_pts_sdf_l1(prog) * loss_pts_sdf_l1
            losses['pts_sdf_l1'] = loss_pts_sdf_l1
        # 9. pts_normal
        if self.lambda_pts_normal(prog) > 0 and 'pts' in sample:
            loss_pts_normal_l1, loss_pts_normal_cos = normal_loss(output['pts_gradient'], sample['pts_normal'], mask=sample['pts_confidence'])
            loss += self.lambda_pts_normal(prog) * (loss_pts_normal_l1 + loss_pts_normal_cos)
            losses['pts_normal_l1'] = loss_pts_normal_l1
            losses['pts_normal_cos'] = loss_pts_normal_cos
        # 10. s3im
        if self.lambda_s3im(prog) > 0:
            loss_s3im = self.s3im(output['rgb'], sample['rgb'], self.patch_size)
            loss += self.lambda_s3im(prog) * loss_s3im
            losses['s3im'] = loss_s3im
        # 11. nb_regu: ignore
        if self.lambda_nb_regu(prog) > 0 and output.get('quat', None) is not None and prog < anneal_end_regu:
            # TODO: [?] 怎么构造 eg：loss_nb_regu = min 1-abs|normal*n|+ half_theta
            # loss_nb_regu = torch.tensor(0.0, device=loss.device)
            loss_nb_regu = nb_regu_loss(output['quat'], output['normal_w'], mask=(~outside)&foreground_mask)
            # loss_nb_regu = nb_regu_loss2(output['quat'], sample['normal'], sample['pose'], mask=None)
            # loss_nb_regu = nb_regu_loss3(output['quat'], output['normal_w'], mask=(~outside)&foreground_mask)
            loss += self.lambda_nb_regu(prog) * loss_nb_regu
            losses['nb_regu'] = loss_nb_regu
        # 12. adaptive_biased_normal
        if self.lambda_ab_normal(prog) > 0:
            # TODO: 自适应加权的normal loss
            loss_ab_normal_l1, loss_ab_normal_cos, loss_ab_biased_l1, loss_ab_biased_cos =\
                adaptive_biased_normal_loss(output['quat'], output['normal_w'], output['biased_normal_w'], sample['normal'], sample['pose'], mask=(~outside)&foreground_mask&mask)
            loss += self.lambda_ab_normal(prog) * (loss_ab_normal_l1 + loss_ab_normal_cos + loss_ab_biased_l1 + loss_ab_biased_cos)
            losses['ab_normal_l1'] = loss_ab_normal_l1
            losses['ab_normal_cos'] = loss_ab_normal_cos
            losses['ab_biased_l1'] = loss_ab_biased_l1
            losses['ab_biased_cos'] = loss_ab_biased_cos
        # 13. ab_depth
        if self.lambda_ab_depth(prog) > 0:
            lambda_ab_depth = adaptive_depth_loss(output['angle'], output['depth'], 50*sample['depth']+0.5, mask=(~outside)&foreground_mask&mask, monocular=True, patch_size=self.patch_size)
            loss += self.lambda_ab_depth(prog) * lambda_ab_depth
            losses['ab_depth'] = lambda_ab_depth
        # 14. ab_smooth
        if self.lambda_ab_smooth(prog) > 0:
            loss_ab_smooth = adaptive_smooth_loss(output['angle'], output['gradient_smooth'], output['gradient_smooth_neighbor'], mask=(~outside)&foreground_mask)
            loss += self.lambda_ab_smooth(prog) * loss_ab_smooth
            losses['ab_smooth'] = loss_ab_smooth

        losses['total'] = loss

        if torch.isnan(loss):
            raise ValueError('loss is nan')
        return losses
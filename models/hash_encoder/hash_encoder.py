'''
# TODO： 经过测试tcnn的综合效率要稍高于我，具体来说batch_size=4, num_rays=1024， 包括forward和backward快0.03s左右。
'''
import torch
from torch.autograd.function import Function
import torch.nn as nn
from .backend import backend
import numpy as np


# 0-order func: hash encoder forward
class HashEncoderFunc(Function):
    @staticmethod
    def forward(ctx,x,embeddings,args):
        # x: B,D
        # embeddings: offsets[-1],C

        # 保证内存连续
        x=x.contiguous()
        embeddings=embeddings.contiguous()
        args['offsets']=args['offsets'].contiguous()
        args['resolution_list']=args['resolution_list'].contiguous()
        y=torch.zeros(args['B'],args['L']*args['C'],device=x.device, dtype=args['dtype']) # y: B,L,C
        dy_dx=torch.zeros(args['B']*args['L']*args['D']*args['C'],device=x.device, dtype=args['dtype']) # dy/dx: B,L,D,C, if_cal_grad_x is True 计算grad_x, 需提前计算dy/dx。
        backend.hash_encoder_forward(x,embeddings,args['offsets'],args['resolution_list'],args['B'],args['D'],args['C'],args['L'],args['if_cal_grad_x'],args['active_levels'],y,dy_dx)
        ctx.save_for_backward(x,embeddings,dy_dx)
        ctx.args=args
        return y
    @staticmethod
    def backward(ctx, grad_y):
        # grad_y: B,L,C
        # make sure grad_y is contiguous
        grad_y=grad_y.contiguous()

        x,embeddings,dy_dx=ctx.saved_tensors
        args=ctx.args
        grad_x,grad_embeddings=HashEncoderBackwardFunc.apply(grad_y, x, embeddings,dy_dx,args)
        return grad_x,grad_embeddings,None

'''
double backward需要的原因： backward计算图终点关于grad_x即normal的loss标量, 因此double backward。
double backward，计算图得到sdf，一阶backward计算图得到的L(normal)，double backward是计算关于L(normal)的梯度
两个grad前缀即L(normal)的梯度，一个grad前缀即sdf的梯度。
'''
# 1-order func: hash encoder backward
class HashEncoderBackwardFunc(Function):
    @staticmethod
    def forward(ctx, grad_y, x, embeddings,dy_dx,args):
        # print('grad_y',grad_y)
        # print('dy_dx',dy_dx)
        grad_x=torch.zeros_like(x,dtype=args['dtype'])
        grad_embeddings=torch.zeros_like(embeddings)
        backend.hash_encoder_backward(grad_y,x,dy_dx,args['offsets'],args['resolution_list'],args['B'],args['D'],args['C'],args['L'],args['if_cal_grad_x'],args['active_levels'],grad_x,grad_embeddings)
        # save for second backward
        ctx.save_for_backward(grad_y,x,embeddings,dy_dx)
        ctx.args=args
        if args['if_cal_grad_x']:
            return grad_x,grad_embeddings
        return None,grad_embeddings
    # FIXME：
    @staticmethod
    def backward(ctx, grad2_grad_x, grad2_grad_embeddings):
        # grad2_grad_x: B,D
        # make sure grad2_grad_x is contiguous
        # print('grad2_grad_x',grad2_grad_x)
        grad2_grad_x=grad2_grad_x.contiguous()

        grad_y,x,embeddings,dy_dx=ctx.saved_tensors
        args=ctx.args
        grad2_embeddings=torch.zeros_like(grad2_grad_embeddings)
        grad2_grad_y=torch.zeros_like(grad_y)
        # FIXME: 应该直接计算累积二阶hessian即hessian_x，不该计算grad2_x，有可能grad_x和grad2_x发生冲突。
        grad2_x = torch.zeros_like(x)
        backend.hash_encoder_second_backward(grad2_grad_x,x,embeddings,grad_y,dy_dx,args['offsets'],args['resolution_list'],args['B'],args['D'],args['C'],args['L'],args['if_cal_hessian_x'],args['active_levels'],grad2_embeddings, grad2_grad_y, grad2_x)
        if args['if_cal_hessian_x']:
            return grad2_grad_y,grad2_x,grad2_embeddings,None,None
        return grad2_grad_y,None,grad2_embeddings,None,None


# hashmap: h(grid_idx)->hash_idx
class HashEncoder(nn.Module):

    def __init__(self, x_dim=3, num_levels=16, per_level_dim=2, log2_hashmap_size=19, base_resolution=16, max_resolution=2048, resolution_list=None,type='float') -> None:
        '''
        hash encoder

        Args:
            x_dim: input dimension, 2 or 3
            num_levels: number of grid levels
            per_level_dim: per level feature dimension
            log2_hashmap_size: log2(hashmap_size)
            base_resolution: base resolution
            max_resolution: max resolution
            resolution_list: resolution list, if None, will be calculated by base_resolution and max_resolution
            type: embeddings type, ['float','half','double'], determine the dtype of output
        '''
        super().__init__()
        self.x_dim=x_dim
        assert x_dim in [2,3], "[ERROR]-[hash-encoder] x_dim must be 2 or 3!!!!"
        self.num_levels=num_levels
        self.per_dim=per_level_dim
        self.log2_hashmap_size=log2_hashmap_size
        self.base_resolution=base_resolution

        # resolution_list
        if resolution_list is None or resolution_list=='None':
            per_level_scale=np.exp2((np.log2(max_resolution/base_resolution))/(num_levels-1))
            resolution_list=[int(np.ceil(base_resolution*per_level_scale**i))for i in range(num_levels)]
        else:
            self.num_levels=len(resolution_list)
        self.register_buffer('resolution_list',torch.tensor(resolution_list, dtype=torch.int32))

        # offsets: prefix sum of per level hashmap size
        offsets=[0]
        max_hashmap_size=2**log2_hashmap_size
        for res in resolution_list:
            offsets.append(offsets[-1]+min(res**x_dim, max_hashmap_size))
        self.register_buffer('offsets',torch.tensor(offsets, dtype=torch.int32))

        # embeddings
        # embeddings type: ['float','half','double'], 默认存储为float32, forward时根据type转换为对应的dtype
        type2dtype={'float':torch.float32,'half':torch.half,'double':torch.double}
        assert type in type2dtype.keys(), "[ERROR]-[hash-encoder] type must be one of ['float','half','double']!!!!"
        self.dtype=type2dtype[type]
        self.embeddings=nn.Parameter(torch.empty(offsets[-1],per_level_dim,dtype=self.dtype),requires_grad=True)
        std=1e-4
        self.embeddings.data.uniform_(-std,std) # 均匀分布初始化
        # with torch.no_grad():
        #     self.embeddings.fill_(1)
        #     self.embeddings.data[:2]=0

        # log
        self.log_info()

    def forward(self, x, if_cal_hessian_x=False, active_levels=666):
        # input: x[..., D], default: x ∈ [-1,1]
        # output: y[..., L*C]
        # when x.requires_grad=True: backward will cal grad_x
        # active_levels: active levels, default: all levels
        prefix_shape_saved = x.shape[:-1]
        x=(x+1)/2 # x -> [0,1]
        x=x.view(-1,self.x_dim)
        args = {'offsets':self.offsets,
                'resolution_list':self.resolution_list,
                'B':x.shape[0],
                'D':self.x_dim,
                'C':self.per_dim,
                'L':self.num_levels,
                'if_cal_grad_x': x.requires_grad,
                'if_cal_hessian_x': False, # TODO: 关于x的二阶hessian暂时有问题（好像不用手动实现）
                'active_levels':active_levels,
                'dtype':self.dtype}
        y=HashEncoderFunc.apply(x,self.embeddings,args)
        return y.view(*prefix_shape_saved,-1)

    def log_info(self):
        print("-------------------HASH ENCODER, Built by dawnzyt-------------------")
        print("[hash encoder info] x_dim:{}, num_levels:{}, per_dim:{}, log2_hashmap_size:{}, base_resolution:{}, max_resolution:{}, resolution_list:{}, type:{}".format(
            self.x_dim,self.num_levels,self.per_dim,self.log2_hashmap_size,self.base_resolution,self.resolution_list[-1],self.resolution_list,self.dtype
        ))
    def encoded_length(self):
        return self.num_levels*self.per_dim

'''
[DEBUG]

'''
# hash_encoder=HashEncoder(x_dim=3,resolution_list=[2,4,8],type='float').cuda()
# print('embeddings',hash_encoder.embeddings)
# x=torch.tensor([[0,0,0]],dtype=torch.float32,requires_grad=True).cuda()
# y=hash_encoder(x, if_cal_hessian_x=True)
# print('y',y)
# grad_x=torch.autograd.grad(y.sum(),x, create_graph=True, retain_graph=True)[0]
#
# # print('grad_x',grad_x)
# # # grad2_grad_x=torch.autograd.grad(grad_x.sum(),x)[0]
# # # print(grad2_grad_x)
# # grad2_embeddings=torch.autograd.grad(grad_x.sum(),hash_encoder.embeddings)[0]
# # print(grad2_embeddings)
#
# # FIXME: 二阶hessian计算，已实现的grad2_grad_x不知为什么失效。
# output_grads = torch.zeros((3, len(x), 3), device=grad_x.device)
# output_grads[torch.arange(3),...,torch.arange(3)] = 1.
# hessian = torch.autograd.grad(grad_x, x, grad_outputs=output_grads, create_graph=True,retain_graph=True,is_grads_batched=True)[0] #
# hessian = hessian.permute(1, 0, 2).reshape(len(x), 3, 3)
# print(hessian)
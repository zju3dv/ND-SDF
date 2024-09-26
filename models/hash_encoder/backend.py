
import torch
from torch.utils.cpp_extension import load
import os

rt_dir=os.path.dirname(__file__)
os.makedirs(os.path.join(rt_dir,'build'),exist_ok=True)

backend=load(
    name='hash_encoder_backend',
    sources=[os.path.join(rt_dir,'src','bindings.cpp'),os.path.join(rt_dir,'src','hash_encoder.cu')],
    extra_cflags=['-O3','-std=c++14'],
    extra_cuda_cflags=['-O3','-std=c++14','-allow-unsupported-compiler','-U__CUDA_NO_HALF_OPERATORS__',
                       '-U__CUDA_NO_HALF_CONVERSIONS__','-U__CUDA_NO_HALF2_OPERATORS__',],
    build_directory=os.path.join(rt_dir,'build'),
    verbose=True
)

__all__=[backend]
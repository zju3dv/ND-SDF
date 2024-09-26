import setuptools
import os
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup, find_packages
import sys

extension=CUDAExtension(name='hash_encoder._C',
                        sources=[os.path.join('src','bindings.cpp'),os.path.join('src','hash_encoder.cu')],
                        extra_compile_args={'cxx':['-O3','-std=c++14'],'nvcc':['-O3','-std=c++14','-allow-unsupported-compiler','-U__CUDA_NO_HALF_OPERATORS__',
                       '-U__CUDA_NO_HALF_CONVERSIONS__','-U__CUDA_NO_HALF2_OPERATORS__',]},
                        verbose=True)
setup(
    name='hash_encoder',
    packages=['hash_encoder'],
    version='0.0.1',
    ext_modules=[extension],
    cmdclass={'build_ext':BuildExtension},
    author='Ziyu Tang',
)
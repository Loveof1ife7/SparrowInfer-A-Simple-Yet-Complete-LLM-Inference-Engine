from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="flashattn",
    ext_modules=[
        CUDAExtension(
            name="flashattn",   # import flashattn
            sources=[
                os.path.join(this_dir, "flashattn.cpp"),
                os.path.join(this_dir, "flashattn_cuda.cu"),
                os.path.join(this_dir, "kernels/fa_passthrough.cu"),
                os.path.join(this_dir, "kernels/fa_naive.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                ],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)

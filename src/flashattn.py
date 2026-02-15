import os
from torch.utils.cpp_extension import load

def _build_extension():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    sources = [
        os.path.join(this_dir, "flashattn.cpp"),
        os.path.join(this_dir, "flashattn_cuda.cu"),
        os.path.join(this_dir, "kernels", "fa_passthrough.cu"),
        os.path.join(this_dir, "kernels", "fa_tiled.cu"),
        os.path.join(this_dir, "kernels", "fa_tensor_core.cu"),
        os.path.join(this_dir, "kernels", "fa_vectorized16x16.cu"),
        os.path.join(this_dir, "kernels", "fa_vectorized16x64.cu"),
        os.path.join(this_dir, "kernels", "fa_warp_shuffle.cu")
    ]
    extra_cflags = ["-O3"]
    extra_cuda_cflags = ["-O3", "--use_fast_math", "-lineinfo"]
    # extra_cflags = ["-O0", "-g"]
    # extra_cuda_cflags = [
    #     "-O0",
    #     "-G", "-g",                 # device debug
    #     "-lineinfo",
    #     "--fmad=false",           
    #     "-U__CUDA_NO_HALF_OPERATORS__",    
    #     "-U__CUDA_NO_HALF2_OPERATORS__",   
    #     "-U__CUDA_NO_HALF_CONVERSIONS__",      
    # ]   
    ext = load(
        name="flashattn_ext",
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=True,
    )
    return ext

_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        _ext = _build_extension()
    return _ext

def flash_attention_forward_v1(q, k, v):
    return _get_ext().flash_attention_forward_v1(q, k, v)

def flash_attention_forward_v2(q, k, v):
    return _get_ext().flash_attention_forward_v2(q, k, v)

def flash_attention_forward_v3(q, k, v):
    return _get_ext().flash_attention_forward_v3(q, k, v)

def flash_attention_forward_v4(q, k, v):
    return _get_ext().flash_attention_forward_v4(q, k, v)

def flash_attention_forward_v5(q, k, v):
    return _get_ext().flash_attention_forward_v5(q, k, v)
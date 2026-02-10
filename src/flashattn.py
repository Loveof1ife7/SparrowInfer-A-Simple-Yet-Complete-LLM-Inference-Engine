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
        os.path.join(this_dir, "kernels", "fa_vectorized.cu")
    ]
    extra_cflags = ["-O3"]
    extra_cuda_cflags = ["-O3", "--use_fast_math", "-lineinfo"]
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

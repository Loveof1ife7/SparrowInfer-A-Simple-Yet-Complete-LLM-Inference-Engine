// CUDA launcher + kernels

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "utils/check.cuh"

// declare the kernel launchers. cpp linker will find the definitions in kernels/*.cu
void launch_attn_tiled(
    const half *q, const half *k, const half *v, half *out,
    int B, int H, int N, int D);

void launch_fa_tensor_core(
    const half *Q,
    const half *K,
    const half *V,
    half *out,
    int B, int H, int N, int D,
    float softmax_scale);

torch::Tensor flash_attention_forward_v1(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v)
{
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda() && v.is_cuda(), "k/v must be CUDA");
    TORCH_CHECK(q.dtype() == k.dtype() && k.dtype() == v.dtype(), "dtype mismatch");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "must be contiguous");
    TORCH_CHECK(q.dtype() == torch::kHalf, "目前仅支持fp16");

    auto out = torch::empty_like(q);
    launch_attn_tiled(
        (const half *)q.data_ptr<at::Half>(),
        (const half *)k.data_ptr<at::Half>(),
        (const half *)v.data_ptr<at::Half>(),
        (half *)out.data_ptr<at::Half>(),
        q.size(0), q.size(1), q.size(2), q.size(3));
    CUDA_KERNEL_CHECK();
    return out;
}

torch::Tensor flash_attention_forward_v2(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v)
{
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda() && v.is_cuda(), "k/v must be CUDA");
    TORCH_CHECK(q.dtype() == k.dtype() && k.dtype() == v.dtype(), "dtype mismatch");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "must be contiguous");
    TORCH_CHECK(q.dtype() == torch::kHalf, "目前仅支持fp16");

    auto out = torch::empty_like(q);

    launch_fa_tensor_core(
        (const half *)q.data_ptr<at::Half>(),
        (const half *)k.data_ptr<at::Half>(),
        (const half *)v.data_ptr<at::Half>(),
        (half *)out.data_ptr<at::Half>(),
        q.size(0), q.size(1), q.size(2), q.size(3), 1.0f / sqrtf((float)q.size(3)));

    CUDA_KERNEL_CHECK();
    return out;
}

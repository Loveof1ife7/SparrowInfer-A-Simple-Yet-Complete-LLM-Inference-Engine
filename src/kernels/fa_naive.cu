// src/kernels/fa_naive.cu
// 这是一个非常基础的实现，主要用于理清Attention的计算流程和CUDA编程模型。它没有使用任何高级优化技术，如共享内存、分块计算、Warp Shuffle等，因此性能较低，但代码结构清晰，易于理解。

// Layout: [B, H, N, D], dtype: fp16 (half)

// scores[t,s] = 1 / sqrt(D) * \sum_d q[t,d] * k[s,d]
// attn[t,s] = softmax_s(scores[t,s])
// out[t,d] = \sum_s attn[t,s] * v[s,d]

// ┌───────────────────────────────────── Grid (B*H × N) ────────────────────────────────────-┐
// │ blockIdx.y = 0 (t=0)  │ blockIdx.y = 1 (t=1)  │ ... │ blockIdx.y = N-1 (t=N-1)           │
// ├──────────────────────────────────────────────────────────────────────────────────────────┤
// │ bh=0 (b=0,h=0)        │ bh=0 (b=0,h=0)        │ ... │ bh=0 (b=0,h=0)                     │
// │ 处理Q[0,0,0]           │ 处理Q[0,0,1]           │     │ 处理Q[0,0,N-1]                     │
// ├──────────────────────────────────────────────────────────────────────────────────────────┤
// │ bh=1 (b=0,h=1)        │ bh=1 (b=0,h=1)        │ ... │ bh=1 (b=0,h=1)                     │
// │ 处理Q[0,1,0]           │ 处理Q[0,1,1]           │     │ 处理Q[0,1,N-1]                     │
// ├──────────────────────────────────────────────────────────────────────────────────────────┤
// │ ...                   │ ...                   │ ... │ ...                                │
// ├──────────────────────────────────────────────────────────────────────────────────────────┤
// │ bh=B*H-1              │ bh=B*H-1              │ ... │ bh=B*H-1                           │
// │ 处理最后一个(batch,head)│ 处理最后一个(batch,head)│     │ 处理最后一个(batch,head)             │
// └──────────────────────────────────────────────────────────────────────────────────────────┘
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// base offset helper
__device__ __forceinline__ int idx4(
    int b, int h, int n, int d,
    int H, int N, int D)
{
    return (((b * H + h) * N + n) * D + d);
}

__inline__ __device__ float warp_reduce_sum(float v)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        if (threadIdx.x + offset < 32)
            v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

__inline__ __device__ float warp_reduce_max(float v)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        if (threadIdx.x + offset < 32)
            v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    return v;
}

__inline__ __device__ float block_reduce_sum(float v)
{
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0)
        shared[warp] = v;
    __syncthreads();

    if (warp == 0)
    {
        int warp_id = lane;
        v = (warp_id < (blockDim.x >> 5)) ? shared[warp_id] : 0.0f;
        v = warp_reduce_sum(v);
    }
    return v;
}

__inline__ __device__ float block_reduce_max(float v)
{
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_reduce_max(v);
    if (lane == 0)
        shared[warp] = v;
    __syncthreads();

    if (warp == 0)
    {
        int warp_id = lane;
        v = (warp_id < (blockDim.x >> 5)) ? shared[warp_id] : -INFINITY;
        v = warp_reduce_max(v);
    }
    return v;
}

__global__ void attn_v0_kernel(
    const half *__restrict__ q,
    const half *__restrict__ k,
    const half *__restrict__ v,
    half *__restrict__ out,
    int B, int H, int N, int D)
{
    int bh = blockIdx.x; // [0, B*H)]
    int t = blockIdx.y;  // [0, N)
    int b = bh / H;
    int h = bh % H;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    if (t >= N)
        return;

    __shared__ float scores_shared[256];
    __shared__ float exp_scores_shared[256];

    float scale = 1.0f / sqrtf((float)D);
    float local_max = -INFINITY;

    for (int s = tid; s < N; s += num_threads)
    {
        // compute scores[t,s]
        float dot = 0.0f;
        for (int d = 0; d < D; d++)
        {
            float qf = __half2float(q[idx4(b, h, t, d, H, N, D)]);
            float kf = __half2float(k[idx4(b, h, s, d, H, N, D)]);
            dot += qf * kf;
        }
        float score = dot * scale;
        local_max = fmaxf(local_max, score);
        scores_shared[tid] = score;
    }
    float block_max = block_reduce_max(local_max);

    float local_sum = 0.0f;
    for (int s = tid; s < N; s += num_threads)
    {
        float exp_score = expf(scores_shared[s] - block_max);
        exp_scores_shared[s] = exp_score;
        local_sum += exp_score;
    }
    float block_sum = block_reduce_sum(local_sum);

    if (tid == 0)
    {
        for (int d = 0; d < D; d++)
        {
            float out_val = 0.0f;
            for (int s = 0; s < N; s++)
            {
                float attn = exp_scores_shared[s] / block_sum;
                out_val += attn * __half2float(v[idx4(b, h, s, d, H, N, D)]);
            }
            out[idx4(b, h, t, d, H, N, D)] = __float2half(out_val);
        }
    }
}

void launch_attn_v0(const half *q, const half *k, const half *v, half *out,
                    int B, int H, int N, int D)
{
    dim3 grid(B * H, N);
    int threads = 256;
    attn_v0_kernel<<<grid, threads>>>(q, k, v, out, B, H, N, D);
}

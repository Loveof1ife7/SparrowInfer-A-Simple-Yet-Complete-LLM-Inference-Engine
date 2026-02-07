#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// Tiling Config
// D=64
#define TR 16 // Tile size for Q (rows)
#define TC 16 // Tile size for K/V (cols)
#define D_MODEL 64

__device__ __forceinline__ int idx(int b, int h, int n, int d, int H, int N, int D)
{
    return ((b * H + h) * N + n) * D + d;
}

// 1. 加载固定的 Q 片段到 Shared Memory (Block-shared)
// 本 block 负责的 16 行 Q，一旦加载，后面所有 K/V tile 都要复用。
// Load_Q_To_Shared();

// 2. 流水线循环：load K 和 V 进来 pre tile
// S_K/S_V：每次循环 j += TC，加载一块 16 行的 K 和 V 到 shared，算完就下一块。

// for (int j = 0; j < N; j += 16) {
// A: 大家合力把这tile的 K 和 V 搬到 Shared Memory
//     Load_K_V_To_Shared();

// B: 算出当前的 Q 和这tile K 的分数 (Score)
//     Compute_Scores();

// C: 根据分数，把这tile V 加权累加到结果里 (Online Softmax)
//     Update_Output_Accumulator();
// }

__global__ void attn_v1_tiled_kernel(
    const half *__restrict__ Q,
    const half *__restrict__ K,
    const half *__restrict__ V,
    half *__restrict__ out,
    int B, int H, int N, int D,
    float softmax_scale)
{
    int bh = blockIdx.x;     // b * H + h
    int tr_idx = blockIdx.y; // tile index
    int b = bh / H;
    int h = bh % H;

    int tid = threadIdx.x;         // lane_id [0, 31]
    int q_row_local = threadIdx.y; // [0, TR-1]

    // [row_start, row_end) for this block
    int row_start = tr_idx * TR;
    int global_q_row = row_start + q_row_local;

    bool is_valid_q = (global_q_row < N);

    // shared memory for this block
    __shared__ half S_Q[TR][D_MODEL];
    __shared__ half S_K[TC][D_MODEL];
    __shared__ half S_V[TC][D_MODEL];

    // register state
    float m_i = -1e20f; // row max
    float l_i = 0.0f;   // row sum

    // D=64, warp=32 threads, each thread computes 2 output features
    // e.g. col = tid, tid + 32
    float acc0 = 0.0f, acc1 = 0.0f;

    // load Q tile to shared memory for this block
    for (int k = 0; k < D; k += 32) // k = 0, 32
    {
        if (tid + k < D)
            S_Q[q_row_local][tid + k] = Q[idx(b, h, global_q_row, tid + k, H, N, D)];
        else
            S_Q[q_row_local][tid + k] = __float2half(0.0f);
    }
    __syncthreads(); // block threads sync

    // load K, V tiles in a loop,
    // 每次循环 j += TC，加载一块 16 行的 K 和 V 到 shared，算完就下一块。
    // e.g. N=128, TC=16, j=0,16,32,64,80,96,112
    for (int j = 0; j < N; j += TC)
    {
        // Step1: load K, V tiles to shared memory
        int thread_row_id = threadIdx.y;
        int kv_row_global = j + thread_row_id;
        // Load this K, V tile to shared memory
        for (int k = 0; k < D; k += 32)
        {
            if (tid + k < D && kv_row_global < N)
                S_K[thread_row_id][tid + k] = K[idx(b, h, kv_row_global, tid + k, H, N, D)];
            else
                S_K[thread_row_id][tid + k] = __float2half(0.0f);
        }
        for (int k = 0; k < D; k += 32)
        {
            if (tid + k < D && kv_row_global < N)
                S_V[thread_row_id][tid + k] = V[idx(b, h, kv_row_global, tid + k, H, N, D)];
            else
                S_V[thread_row_id][tid + k] = __float2half(0.0f);
        }
        __syncthreads(); // block threads sync

        // Step2: compute scores and update output accumulators
        // Each dot product of one Q row and one K row
        for (int k_idx = 0; k_idx < TC; ++k_idx)
        {
            float dot = 0.0f;               // register
            for (int k = 0; k < D; k += 32) // k = 0,32, each thread processes two elements
            {
                if (tid + k < D)
                {
                    float q_val = __half2float(S_Q[q_row_local][tid + k]); // q_row_local determine which warp, feature_dim determine which thread
                    float k_val = __half2float(S_K[k_idx][tid + k]);
                    dot += q_val * k_val;
                }
            }
            // Butterfly Reduction in warp
            for (int offset = 16; offset > 0; offset /= 2)
            {
                dot += __shfl_down_sync(0xffffffff, dot, offset);
            }
            // lane 0 has the final dot product
            float score = __shfl_sync(0xffffffff, dot, 0);
            score *= softmax_scale;

            float m_prev = m_i;
            m_i = fmaxf(m_i, score);
            float p = expf(score - m_i);
            float correction = expf(m_prev - m_i);

            // Update Denominator
            l_i = l_i * correction + p;
            // Update Numerator
            if (tid < D)
            {
                float v_val = __half2float(S_V[k_idx][tid]);
                acc0 = acc0 * correction + p * v_val;
            }
            if (tid + 32 < D)
            {
                float v_val = __half2float(S_V[k_idx][tid + 32]);
                acc1 = acc1 * correction + p * v_val;
            }
        }
        __syncthreads();
    }
    if (is_valid_q)
    {
        if (tid < D)
        {
            float out_val0 = acc0 / l_i;
            out[idx(b, h, global_q_row, tid, H, N, D)] = __float2half(out_val0);
        }
        if (tid + 32 < D)
        {
            float out_val1 = acc1 / l_i;
            out[idx(b, h, global_q_row, tid + 32, H, N, D)] = __float2half(out_val1);
        }
    }
}

void launch_attn_v1(const half *q, const half *k, const half *v, half *out,
                    int B, int H, int N, int D)
{
    // 总Block数 = Grid.x × Grid.y = (B × H) × ceil(N / 16)
    // 每个Block负责处理一个 (b, h) 的 16 行Q
    dim3 grid(B * H, (N + TR - 1) / TR);

    // x方向32线程，y方向16线程, 共512线程, 512/32=16 warps
    // 一个warp负责一行query,也就是64维度特征向量
    dim3 block(32, TR);

    float scale = 1.0f / sqrtf((float)D);
    int shared_mem_size = 0; // Static allocation used

    attn_v1_tiled_kernel<<<grid, block, shared_mem_size>>>(q, k, v, out, B, H, N, D, scale);
}
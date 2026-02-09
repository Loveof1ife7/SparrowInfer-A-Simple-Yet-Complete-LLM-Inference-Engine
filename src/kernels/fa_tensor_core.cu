// src/kernels/fa_tensor_core.cu
// 这是一个基于 Tensor Cores 的实现，利用了 NVIDIA GPU 上的专门矩阵乘法单元来加速 Attention 的计算。这个版本的性能应该是最好的，特别是在较大的输入规模下，因为它充分利用了硬件的计算能力

// 使用 Tensor Cores (HMMA) 替代 CUDA Core 计算
// fa_tiled.cu:
// dot += q * k 和 acc += p * v 的循环。这被称为 SIMT (Single Instruction Multiple Threads) 模式
// 现代 NVIDIA GPU (Volta/Ampere/Hopper) 有专门的 Tensor Cores，它们可以在一个指令周期内完成 $16 \times 8 \times 16$ 甚至更大的矩阵乘法。
// fa_tensor_core.cu：
// 使用 nvcuda::wmma (C++ API) 或 mma.sync (PTX) 指令。
// 将计算从 "向量-向量点积" 转变为 "矩阵片段 (Fragment) 乘法"。
// 你需要将 Q, K, V 的 Layout 调整为 Tensor Core 友好的格式（通常是 wmma::col_major 或 wmma::row_major），并直接在 half 精度下进行矩阵乘，只在累加器中使用 float 精度

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// Tiling Config
#define TR 16 // Q Tile size
#define TC 16 // K/V Tile size
#define D_MODEL 64

// Avoiding Bank Conflict
#define SMEM_STRIDE (D_MODEL + 16)

// WMMA Constants
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__device__ __forceinline__ int idx4(
    int b, int h, int n, int d,
    int batch_size, int num_heads, int seq_len)
{
    return ((b * num_heads + h) * seq_len + n) * D_MODEL + d;
}

__global__ void fa_tensor_core_kernel(
    const half *__restrict__ Q,
    const half *__restrict__ K,
    const half *__restrict__ V,
    half *__restrict__ out,
    int B, int H, int N, int D,
    float scale)
{
    int bh = blockIdx.x;
    int tr_idx = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    int tid = threadIdx.x; // BlockDim = 32
    int q_row_start = tr_idx * TR;

    __shared__ half S_Q[TR][SMEM_STRIDE];
    __shared__ half S_K[TC][SMEM_STRIDE];
    __shared__ half S_V[TC][SMEM_STRIDE];
    __shared__ float S_corr[TR];             // row correction
    __shared__ float S_tmp[TR][SMEM_STRIDE]; // tile PV buffer
    __shared__ float S_L[TR];                // Softmax normalization factor (l_i)
    __shared__ float S_S[TR][TC];            // score buffer
    __shared__ float S_O[TR][SMEM_STRIDE];   // output buffer

    // Fragments (寄存器片段)
    // Q 片段: 即使 D=64，我们每次 WMMA 只处理 K=16
    // 所以 Q 需要被切成 4 块 (64/16), 我们把它们都加载到寄存器里复用
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_Q[4];
    // K 片段: 用 matrix_b
    // 注意：我们要算 Q * K^T
    // 如果 K 在 shared memory 是行主序 [16, 64]，我们作为 matrix_b 加载时使用 wmma::col_major，
    // 硬件会自动帮我们转置，等效于加载了 K^T。
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_K;
    // V 片段: 用 matrix_b
    // O = P * V, P 是 16x16, V 是 16x64
    // V 作为右矩阵，Row Major 加载
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_V;
    // Accumulator fragment
    // S (Score): 16x16, float
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_S;
    // O (Output): 16x64, half, 4 个片段
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_O[4];
    // P (Probability): 16x16, half. 用于 S * V
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_P;

    // Online Softmax Stats (Register)
    float m_i = -1e20f;
    float l_i = 0.0f;

    __syncthreads();
    for (int i = 0; i < 4; i++)
        wmma::fill_fragment(frag_O[i], 0.0f);

    // 3. Load Q (SIMT Copy with Padding logic)
    for (int r = 0; r < TR; ++r)
    {
        int global_r = q_row_start + r;
        if (global_r < N)
        {
            S_Q[r][tid] = Q[idx4(b, h, global_r, tid, B, H, N)];
            S_Q[r][tid + 32] = Q[idx4(b, h, global_r, tid + 32, B, H, N)];
        }
        else
        {
            S_Q[r][tid] = __float2half(0.0f);
            S_Q[r][tid + 32] = __float2half(0.0f);
        }
        if (tid < 16)
        {
            S_Q[r][64 + tid] = __float2half(0.0f); // Padding
        }
    }
    __syncthreads();

    // load Q from Shared to Fragment
    // Q shape: [16, 64], split to 4 * [16, 16] fragments
    for (int i = 0; i < 4; ++i)
    {
        // load_matrix_sync(dst_frag, src_ptr, stride_in_elements)
        const half *tile_ptr = &S_Q[0][i * 16];
        wmma::load_matrix_sync(frag_Q[i], tile_ptr, SMEM_STRIDE);
    }

    // Step.4 main loop,  K, V Tiles
    // 1) init numerator accumulator
    for (int r = 0; r < TR; ++r)
    {
        S_O[r][tid] = 0.0f;
        S_O[r][tid + 32] = 0.0f;
        if (tid < 16)
            S_O[r][64 + tid] = 0.0f;
    }

    for (int j = 0; j < N; j += TC)
    {
        // 2) load K,V to shared (+ padding=0)
        for (int r = 0; r < TC; ++r)
        {
            int global_r = j + r;
            if (global_r < N)
            {
                S_K[r][tid] = K[idx4(b, h, global_r, tid, B, H, N)];
                S_K[r][tid + 32] = K[idx4(b, h, global_r, tid + 32, B, H, N)];
                S_V[r][tid] = V[idx4(b, h, global_r, tid, B, H, N)];
                S_V[r][tid + 32] = V[idx4(b, h, global_r, tid + 32, B, H, N)];
            }
            else
            {
                S_K[r][tid] = __float2half(0.0f);
                S_K[r][tid + 32] = __float2half(0.0f);
                S_V[r][tid] = __float2half(0.0f);
                S_V[r][tid + 32] = __float2half(0.0f);
            }
            // clean padding
            if (tid < 16)
            {
                S_K[r][64 + tid] = __float2half(0.0f);
                S_V[r][64 + tid] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // 3) scores: frag_S = Q * K^T
        wmma::fill_fragment(frag_S, 0.0f);
        // 沿 D 维度 (64) 分 4 次累加
        for (int k_idx = 0; k_idx < 4; ++k_idx)
        {
            const half *k_ptr = &S_K[0][k_idx * 16];
            wmma::load_matrix_sync(frag_K, k_ptr, SMEM_STRIDE);
            wmma::mma_sync(frag_S, frag_Q[k_idx], frag_K, frag_S); // MMA: S = Q_sub * K_sub^T + S
        }

        // 4) store scores -> S_S, compute online softmax per row
        //    produce: S_corr[row], update l_i, write S_S[row][c]=exp(s-m_new)

        // Tensor Core 算出的 S 散落在各线程寄存器里，需要把它存到 Shared Mem (S_S score buffer), 才能算出整行的 Max 和 Sum
        wmma::store_matrix_sync(&S_S[0][0], frag_S, TC, wmma::mem_row_major);
        __syncthreads();
        // TR=16，tid 0~15 工作, Warp 中前 16 个线程，每个负责 1 行ma (TR=16)
        float row_m = -1e20f; // max
        if (tid < TR)
        {
            int row = tid;
            for (int c = 0; c < TC; ++c)
            {
                S_S[row][c] *= scale;
                row_m = fmaxf(row_m, S_S[row][c]);
            }
            float m_prev = m_i;
            // m_i 和 l_i 代表行的全局状态，所以 tid=0 存的是 row0 的 m_i
            // 在 Warp 级别，每个线程应该维护自己那行的 m_i, l_i
            float m_new = fmaxf(m_prev, row_m);
            float correction = expf(m_prev - m_new);

            // 保存行校正因子到S_corr buffer，后面计算 O 时需要乘回去
            S_corr[row] = correction;
            float p_sum_local = 0.0f;

            for (int c = 0; c < TC; ++c)
            {
                float p = expf(S_S[row][c] - m_new);
                S_S[row][c] = p;
                p_sum_local += p;
            }
            // 更新当前线程负责的行的全局状态
            m_i = m_new;
            l_i = (l_i * correction) + p_sum_local; // l_i = sum_tiles sum_j exp(s_row,j - m_i)
        }
        __syncthreads(); // 等待 tid 0-15 计算完 Softmax 并写入 S_S

        // 5) write S_L[row] = l_i
        if (tid < TR)
            S_L[tid] = l_i;
        __syncthreads();

        // 6) rescale old numerator by correction: S_O [row, :] *= S_corr[row]
        for (int r = 0; r < TR; ++r)
        {
            float corr = S_corr[r];
            S_O[r][tid] *= corr;
            S_O[r][tid + 32] *= corr;
        }
        __syncthreads();

        // 7) convert P (float) -> half in shared, load frag_P
        half *P_ptr = (half *)&S_K[0][0]; // Re-use S_K buffer
        for (int i = tid; i < TR * TC; i += 32)
        {
            int r = i / TC;
            int c = i % TC;
            P_ptr[r * SMEM_STRIDE + c] = __float2half(S_S[r][c]);
        }
        __syncthreads();
        wmma::load_matrix_sync(frag_P, P_ptr, SMEM_STRIDE);

        // 8) WMMA compute PV tile into frag_T (accum=0)
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_T[4];
        for (int t = 0; t < 4; ++t)
            wmma::fill_fragment(frag_T[t], 0.0f);

        for (int k_idx = 0; k_idx < 4; ++k_idx)
        {
            // Load V (Row major)
            const half *v_ptr = &S_V[0][k_idx * 16];
            wmma::load_matrix_sync(frag_V, v_ptr, SMEM_STRIDE);
            // MMA: O[k_idx] += P * V_sub
            wmma::mma_sync(frag_T[k_idx], frag_P, frag_V, frag_T[k_idx]); // T = P*V_sub
        }
        // 9) store frag_T -> S_tmp
        for (int t = 0; t < 4; ++t)
            wmma::store_matrix_sync(&S_tmp[0][t * 16], frag_T[t], SMEM_STRIDE, wmma::mem_row_major);
        __syncthreads();

        for (int r = 0; r < TR; ++r)
        {
            S_O[r][tid] += S_tmp[r][tid];
            S_O[r][tid + 32] += S_tmp[r][tid + 32];
        }
        __syncthreads();
    }
    // 11) final write: out = (S_O / S_L) -> half
    for (int r = 0; r < TR; ++r)
    {
        int global_r = q_row_start + r;
        if (global_r < N)
        {
            float inv_l = 1.0f / S_L[r];
            if (tid < 32)
            {
                float x0 = S_O[r][tid] * inv_l;
                float x1 = S_O[r][tid + 32] * inv_l;
                out[idx4(b, h, global_r, tid, B, H, N)] = __float2half_rn(x0);
                out[idx4(b, h, global_r, tid + 32, B, H, N)] = __float2half_rn(x1);
            }
        }
    }
}

void launch_fa_tensor_core(
    const half *Q,
    const half *K,
    const half *V,
    half *out,
    int B, int H, int N, int D,
    float softmax_scale)
{
    // 每个 Block 处理 16 行 Q
    dim3 grid(B * H, (N + TR - 1) / TR);
    // 每个block 32 个线程(1 Warp)
    dim3 block(32, 1);

    float scale = 1.0f / sqrtf((float)D);
    fa_tensor_core_kernel<<<grid, block>>>(
        Q, K, V, out,
        B, H, N, D,
        scale);
}
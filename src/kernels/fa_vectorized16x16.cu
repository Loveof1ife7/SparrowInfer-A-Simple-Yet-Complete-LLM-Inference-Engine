// src/kernels/fa_vectorized.cu
// GPU 的显存总线一次处理 32 字节 (sector)，L2/L1 cache line 也是 128 字节。发起一次 2 字节的加载请求和发起一次 16 字节的请求，延迟几乎是一样的，但吞吐量差了 8 倍。
// S_Q[r][tid] = Q[idx4(...)]; // 每次搬运 1 个 half (2 bytes)
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math.h>

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

using vec16 = uint4; // 4 × 32bit = 16 bytes

__device__ __forceinline__ bool is_aligned16(const void *p)
{
    return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0;
}
__device__ __forceinline__ vec16 ld_vec16(const void *p)
{
    return *reinterpret_cast<const vec16 *>(p); // 强制告诉编译器：一次性读 16 个字节
}
__device__ __forceinline__ void st_vec16(void *p, vec16 v)
{
    *reinterpret_cast<vec16 *>(p) = v; // 在对齐成立的前提下，等价于“发出一条 128-bit 写指令”，一次性写 16 字节。
}

//  向量化加载 = 把 tile 拆成很多 16B 小块，让 warp 线程连续搬这些小块，从而减少指令数并提高内存吞吐
template <int TILE_R>
__device__ __forceinline__ void load_tile_vectorized_16x64(
    half smem[TILE_R][SMEM_STRIDE],
    const half *__restrict__ gmem, // 指向当前 batch/head 的 base
    int row_start,                 // tile 起始行
    int N,                         // seq_len
    int D,                         // head_dim (这里=64)
    int tid                        // 0..31
)
{
    // 本实现假设 D==64。如果未来要支持别的 D，需要分支处理。
    // 每行 64 half = 128B = 8 个 vec16（每 vec16=16B => 8*16=128B）
    constexpr int VEC_BYTES = 16;
    constexpr int HALF_PER_VEC = VEC_BYTES / sizeof(half); // 8 half
    constexpr int VECS_PER_ROW = 64 / HALF_PER_VEC;        // 8

    // 每个线程搬运：总 vec 数 = TILE_R * VECS_PER_ROW
    // 线程 tid 处理 vec_idx = tid, tid+32, ...
    int total_vec = TILE_R * VECS_PER_ROW;

    for (int vec_i = tid; vec_i < total_vec; vec_i += 32)
    {
        // int r = vec_i / VECS_PER_ROW; 计算开销大，使用位运算
        int r = vec_i >> 3;     // tile 内行 (vec_i / VECS_PER_ROW) --- 8=2^3
        int v = vec_i & 7;      // i 行内第几个 vec16
        int gr = row_start + r; // global 行
        int gd = v << 3;        // global 列（0,8,16,...,56）

        // 目标 shared 地址
        half *dst = &smem[r][gd];

        if (gr < N)
        {
            const half *src = gmem + gr * D + gd;

            // 尽可能走 16B 向量路径：要求 src/dst 都 16B 对齐
            // smem 通常是对齐的；gmem 是否对齐取决于 base 指针和 gd
            // if (is_aligned16(src) && is_aligned16(dst))  假设大部分情况下是对齐的，直接强转加载,真正高性能代码通常要求指针本身是对齐的，去掉运行时检查

            vec16 x = ld_vec16(src);
            st_vec16(dst, x);
            // 逻辑等价
            //      half tmp[8];
            //  tmp[0] = src[0];
            //  tmp[1] = src[1];
            //  ...
            //  tmp[7] = src[7];

            // dst[0] = tmp[0];
            // ...
            // dst[7] = tmp[7];
        }
        else
        {
            // 越界行 padding 0
            *(uint4 *)dst = make_uint4(0, 0, 0, 0);
        }
    }

    // padding 区域清零（64..SMEM_STRIDE-1），只需要 tid<16 做即可
    if (tid < 16)
    {
        for (int r = 0; r < TILE_R; ++r)
        {
            smem[r][64 + tid] = __float2half(0.0f);
        }
    }
}

__global__ void fa_tensor_core_vectorized_kernel(
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
    // [Global Memory 的基地址
    // 这样在 copy 函数里只需要做简单的偏移
    int batch_head_offset = (b * H + h) * N * D;
    const half *Q_ptr = Q + batch_head_offset;
    const half *K_ptr = K + batch_head_offset;
    const half *V_ptr = V + batch_head_offset;

    __shared__ half S_Q[TR][SMEM_STRIDE];
    __shared__ half S_K[TC][SMEM_STRIDE];
    __shared__ half S_V[TC][SMEM_STRIDE];
    __shared__ float S_corr[TR];             // row correction
    __shared__ float S_tmp[TR][SMEM_STRIDE]; // tile PV buffer
    __shared__ float S_L[TR];                // Softmax normalization factor (l_i)
    __shared__ float S_S[TR][TC];            // score buffer
    __shared__ float S_O[TR][SMEM_STRIDE];   // output buffer

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_Q[4];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_K;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_V;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_S;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_O[4];
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_P;

    float m_i = -1e20f;
    float l_i = 0.0f;

    for (int i = 0; i < 4; i++)
        wmma::fill_fragment(frag_O[i], 0.0f);

    // Load Q vectorized
    load_tile_vectorized_16x64<TR>(S_Q, Q_ptr, tr_idx * TR, N, D, tid);
    __syncthreads();

    for (int i = 0; i < 4; ++i)
    {
        // load_matrix_sync(dst_frag, src_ptr, stride_in_elements)
        const half *tile_ptr = &S_Q[0][i * 16];
        wmma::load_matrix_sync(frag_Q[i], tile_ptr, SMEM_STRIDE);
    }
    for (int r = 0; r < TR; ++r)
    {
        S_O[r][tid] = 0.0f;
        S_O[r][tid + 32] = 0.0f;
        if (tid < 16)
            S_O[r][64 + tid] = 0.0f;
    }
    __syncthreads();
    for (int j = 0; j < N; j += TC)
    {
        load_tile_vectorized_16x64<TC>(S_K, K_ptr, j, N, D, tid);
        load_tile_vectorized_16x64<TC>(S_V, V_ptr, j, N, D, tid);

        __syncthreads(); // 串行模式：加载->等待->计算

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

void launch_fa_vectorized(
    const half *Q,
    const half *K,
    const half *V,
    half *out,
    int B, int H, int N, int D,
    float softmax_scale)
{
    dim3 grid(B * H, (N + TR - 1) / TR);
    dim3 block(32, 1);
    float scale = 1.0f / sqrtf((float)D);
    fa_tensor_core_vectorized_kernel<<<grid, block>>>(
        Q, K, V, out, B, H, N, D, scale);
}

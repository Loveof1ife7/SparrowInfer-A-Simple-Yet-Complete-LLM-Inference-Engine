#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math.h>

using namespace nvcuda;

// ================= Config =================
// 1. 增大 Tile Size 到 64
#define TR 64
#define TC 64
#define D_MODEL 64

// 2. 增大 Block Size
#define BLOCK_DIM 128
#define WARPS_PER_BLOCK 4

// 3. Shared Memory Stride (64 + 8)
// 加上 padding 避免 Bank Conflict.
#define SMEM_STRIDE (D_MODEL + 8)

// WMMA Setting
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

using vec16 = uint4; // 16 bytes = 8 halfs

__device__ __forceinline__ int idx4(int b, int h, int n, int d, int H, int N)
{
    return ((b * H + h) * N + n) * D_MODEL + d;
}

// ================= Vectorized Loader (upscale version) =================
// 负责让 128 个线程协作搬运 TR*TC 的数据

template <int TILE_SIZE>
__device__ __forceinline__ void load_tile_vectorized_64x64(
    half smem[TILE_SIZE][SMEM_STRIDE],
    const half *__restrict__ gmem,
    int row_start, int N, int D, int tid)
{
    // 任务总量：64行 * 64列 (half)
    // 字节总量：64 * 64 * 2 bytes = 8192 bytes
    // 向量总量：8192 / 16 (sizeof vec16) = 512 个 vec16
    // 线程数：128
    // 每个线程负责：512 / 128 = 4 个 vec16
    const int total_vecs = TILE_SIZE * (D / 8); // 64 * 8 = 512

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        // 计算当前线程要搬运的第 i 个向量的全局索引
        // tid=0 -> 处理 0, 128, 256, 384
        int vec_idx = tid + i * BLOCK_DIM;
        if (vec_idx < total_vecs)
        {
            // 坐标映射：位运算
            // 一行有 8 个 vec16 (64 half/ 8 half)
            // row = vec_idx / 8  -> vec_idx >> 3
            // col_vec = vec_idx % 8 -> vec_idx & 7
            // col_half = col_vec * 8 -> col_vec << 3

            int r = vec_idx >> 3;
            int v = vec_idx & 7;
            int c = v << 3;
            int global_r = row_start + r;

            // 目标 Shared Memory 地址
            half *dst_ptr = &smem[r][c];

            if (global_r < N)
            {
                const half *src_ptr = gmem + global_r * D + c;
                // 强制向量加载
                *reinterpret_cast<vec16 *>(dst_ptr) = *reinterpret_cast<const vec16 *>(src_ptr);
            }
            else
            {
                // Padding zero
                *reinterpret_cast<vec16 *>(dst_ptr) = make_uint4(0, 0, 0, 0);
            }
        }
    }
}

__global__ void fa_kernel_64x64(
    const half *__restrict__ Q,
    const half *__restrict__ K,
    const half *__restrict__ V,
    half *__restrict__ out,
    int B, int H, int N, int D,
    float scale)
{
    // ================= Setup Indices =================
    int bh = blockIdx.x;
    int tr_idx = blockIdx.y; // Q 的分块索引
    int b = bh / H;
    int h = bh % H;
    int tid = threadIdx.x;

    // Warp ID: 0, 1, 2, 3
    // Warp 0: Row 0~15; Warp 1: Row 16~31; Warp 2: Row 32~47; Warp 3: Row 48~63
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int q_row_start = tr_idx * TR;
    int batch_head_offset = (b * H + h) * N * D;
    half *O_ptr = out + batch_head_offset;

    // 指针偏移
    const half *Q_ptr = Q + batch_head_offset;
    const half *K_ptr = K + batch_head_offset;
    const half *V_ptr = V + batch_head_offset;
    //  half *O_ptr = out + batch_head_offset;

    // ================= Shared Memory Allocation =================
    // Q, K, V: 64x64 half.
    // S_Q: 64 * 72 * 2 = 9216 bytes
    // S_K: 9216 bytes
    // S_V: 9216 bytes
    // S_S (Scores): 64 * 64 float (用于Softmax) -> 16KB
    // S_O (Output): 64 * 72 float (累加buffer) -> 18KB
    // Total approx: 9*3 + 16 + 18 ~= 61KB (需要在 Host 端设置 cudaFuncSetAttribute maxDynamicSharedMemorySize)
    // 如果显存不够，可以复用内存（S_K 可以复用给 P），这里为了清晰先分开定义。

    extern __shared__ char smem_buffer[];
    half(*S_Q)[SMEM_STRIDE] = (half(*)[SMEM_STRIDE])smem_buffer;
    half(*S_K)[SMEM_STRIDE] = (half(*)[SMEM_STRIDE])(S_Q + TR);
    half(*S_V)[SMEM_STRIDE] = (half(*)[SMEM_STRIDE])(S_K + TC);

    // float 缓冲区
    float *ptr_float = (float *)(S_V + TC);
    float (*S_S)[TC] = (float (*)[TC])ptr_float;
    float (*S_O)[SMEM_STRIDE] = (float (*)[SMEM_STRIDE])(S_S + TR);

    // 每个 Warp 维护的统计量，不需要 Shared Mem，直接寄存器
    // 但最后写回 Global 需要 S_L 来做归一化，S_L 放在 Shared 里方便最后统一处理
    float *S_L = (float *)(S_O + TR); // Size TR
    float *S_corr = S_L + TR;         // Size TR

    // ================= Fragments Definition =================
    // 每个 Warp 负责 output 的 16 行。
    // Q: Warp i 需要加载 Q 的 row [16*i : 16*(i+1)]
    // K: 需要加载完整的 K (所有行)
    // Q 片段: 只有 1 个 (16x16)，因为 Warp 只负责 Q 的 16 行，且沿着 K 维度(D=64)循环 4 次
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_Q[4];
    // K 片段: 4 个 (覆盖 64 列)
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_K[4];
    // V 片段: 4 个 (覆盖 64 列，注意 V 是 row major 加载)
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_V[4];
    // Accumulators
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_S[4]; // Scores: 16x64 (4 chunks)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_O[4]; // Output: 16x64 (4 chunks)
    // P fragment (Softmax 后)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_P[4];

    // ================= Initialization =================
    for (int i = 0; i < 4; ++i)
        wmma::fill_fragment(frag_O[i], 0.0f);

    // row全局状态(all tiles)
    float m_i = -1e20f;
    float l_i = 0.0f;

    // ================= Main Loop: Load Q =================
    load_tile_vectorized_64x64<TR>(S_Q, Q_ptr, q_row_start, N, D, tid);
    __syncthreads();

    // Pre-load Q into registers for this Warp
    // Warp 0 reads Q[0..15], Warp 1 reads Q[16..31]...
    // 每个 Warp 需要把 D=64 拆成 4 个 16 宽度的块读入
    int warp_row_offset = warp_id * 16;
    for (int k = 0; k < 4; ++k)
        wmma::load_matrix_sync(frag_Q[k], &S_Q[warp_row_offset][k * 16], SMEM_STRIDE);

    // ================= Loop over K/V blocks =================
    for (int j = 0; j < N; j += TC)
    {
        // 1. Load K, V Tiles (Collaboration of 128 threads)
        load_tile_vectorized_64x64<TC>(S_K, K_ptr, j, N, D, tid);
        load_tile_vectorized_64x64<TC>(S_V, V_ptr, j, N, D, tid);
        __syncthreads();

        // 2. Compute S = Q * K^T
        // 每个 Warp 计算 16x64 的结果 (4 个 16x16 块)
        for (int i = 0; i < 4; ++i)
            wmma::fill_fragment(frag_S[i], 0.0f);

        // 沿着 D 维度 (K-dim) 累加: D=64 => 4 chunks of 16
        for (int k_chunk = 0; k_chunk < 4; ++k_chunk)
        {
            // Load K sub-block.
            // K 是 col_major, 我们需要 K^T 的效果
            // S[row_chunk][col_chunk] += Q[row_chunk][k] * K[k][col_chunk]^T
            // 64 列S_K被分成 4 个 16 列的子块，每个子块对应一个
            for (int s_col = 0; s_col < 4; ++s_col)
            {
                const half *k_tile_ptr = &S_K[s_col * 16][k_chunk * 16];
                wmma::load_matrix_sync(frag_K[s_col], k_tile_ptr, SMEM_STRIDE);
                wmma::mma_sync(frag_S[s_col], frag_Q[k_chunk], frag_K[s_col], frag_S[s_col]);
            }
        }

        // RM -> SM,寄存器的私有,Softmax全局, 在 Tensor Core 计算完 $S = Q \times K^T$ 后，数值是分散在 Warp 内的 32 个线程的寄存器里的。
        // 而且wmma::fragment不规整，逻辑上的 S[0][0] 可能在 Thread 0 的寄存器 R5 里，但 S[0][1] 可能在 Thread 4 的寄存器 R2 里（仅作比喻，实际映射非常复杂且随架构变化）。
        for (int i = 0; i < 4; ++i)
        {
            float *s_ptr = &S_S[warp_row_offset][i * 16];
            wmma::store_matrix_sync(s_ptr, frag_S[i], TC, wmma::mem_row_major);
        }
        __syncthreads();

        // 4. Softmax (Row-wise) - Warp 内部并行, 16 个线程负责 16 行。循环计算 64 个元素。
        if (lane_id < 16)
        {
            int row = warp_row_offset + lane_id;
            float row_max = -1e20f;

            for (int c = 0; c < TC; ++c)
            {
                float val = S_S[row][c] * scale;
                S_S[row][c] = val;
                if (val > row_max)
                    row_max = val;
            }

            float m_prev = m_i;
            float m_new = fmaxf(m_prev, row_max);

            // 修复 NaN: 如果 m_prev 是初始极小值，强制 correction 为 0
            float correction = (m_prev <= -1e10f) ? 0.0f : expf(m_prev - m_new);

            S_corr[row] = correction;

            float p_sum = 0.0f;
            for (int c = 0; c < TC; ++c)
            {
                float p = expf(S_S[row][c] - m_new);
                S_S[row][c] = p;
                p_sum += p;
            }

            m_i = m_new;
            l_i = l_i * correction + p_sum;
            S_L[row] = l_i;
        }
        __syncthreads(); // 等待 S_S 被更新为 P，且 S_corr 准备好

        // Correct previous O, Frag_O分散在寄存器，无法与S_corr对齐，用 Shared Memory 中转 S_O
        for (int i = 0; i < 4; ++i)
            wmma::store_matrix_sync(&S_O[warp_row_offset][i * 16], frag_O[i], SMEM_STRIDE, wmma::mem_row_major);
        __syncthreads();

        // S_O = S_O * S_corr + P_TILE * V_TILE
        // Scale S_O by correction
        // tid 0..31 handles 16 rows * 64 cols = 1024 floats. 32 threads, each 32 elems.
        for (int k = 0; k < 32; ++k)
        {
            int idx = lane_id + k * 32; // 0..1023 floats to process
            if (idx < 16 * 64)
            {
                int r = idx / 64; // local row 0..15
                int c = idx % 64; // local col 0..63
                int global_r = warp_row_offset + r;
                S_O[global_r][c] *= S_corr[global_r];
            }
        }
        __syncthreads();

        // Compute P * V and add to S_O
        // Load P (from S_S buffer which now holds P) -> Half
        // Convert float P to half P in place or new buffer
        // S_S is float. WMMA needs half.
        // Convert S_S to half S_P buffer. (Reuse S_K space)
        half *S_P_half = (half *)S_K;
        for (int k = 0; k < 32; ++k) // 32 次才能覆盖 64x64 = 4096 个元素 (128线程 * 32 = 4096)
        {
            int idx = tid + k * 128; // 128 threads covering 64*64
            if (idx < 64 * 64)
            {
                int r = idx / 64;
                int c = idx % 64;
                S_P_half[r * SMEM_STRIDE + c] = __float2half(S_S[r][c]);
            }
        }
        __syncthreads();

        // Load P fragments (4 chunks along K-dim/Cols of P)
        for (int k = 0; k < 4; ++k)
        {
            // 使用显式的偏移量计算
            half *tile_ptr = S_P_half + (warp_row_offset * SMEM_STRIDE) + (k * 16);
            wmma::load_matrix_sync(frag_P[k], tile_ptr, SMEM_STRIDE);
        }

        //  Load S_O into fragment (already scaled), accumulate, store back.
        for (int i = 0; i < 4; ++i)
            wmma::load_matrix_sync(frag_O[i], &S_O[warp_row_offset][i * 16], SMEM_STRIDE, wmma::mem_row_major);

        // 外层循环：沿着“序列长度 (TC)”方向走
        // 我们要把 P 的一行和 V 的一列做点积。这需要把序列长度维度的 64 个数都乘起来加在一起。
        // k_chunk = 0 (TC 0-15), k_chunk = 1 (TC 16-31)...
        for (int k_chunk = 0; k_chunk < 4; ++k_chunk)
        {
            // 内层循环：沿着“特征维度 (D)”方向走
            // 我们要算出 Output 的所有列（所有特征）。
            // sub_col = 0 (D 0-15), sub_col = 1 (D 16-31)...

            for (int sub_col = 0; sub_col < 4; ++sub_col)
            {
                // 加载 V 的一个小块 (16x16)
                // 行 (Row) = k_chunk * 16  ->  当前处理的是哪一段 Sequence (TC)
                // 列 (Col) = sub_col * 16  ->  当前处理的是哪一段 Feature (D)
                wmma::load_matrix_sync(frag_V[sub_col], &S_V[k_chunk * 16][sub_col * 16], SMEM_STRIDE);
                // 矩阵乘加 (MMA): O += P * V
                // frag_O[sub_col] (特征 D): 累加器，位置由特征维度决定。
                // frag_P[k_chunk] (序列 TC): P 的一部分，对应当前的 TC 片段。
                // frag_V[sub_col] (混合): 刚加载进来的 V 小块。
                wmma::mma_sync(frag_O[sub_col], frag_P[k_chunk], frag_V[sub_col], frag_O[sub_col]);
            }
        }
        __syncthreads();
    }

    // ================= Final Store =================
    // Store frag_O -> S_O
    for (int i = 0; i < 4; ++i)
        wmma::store_matrix_sync(&S_O[warp_row_offset][i * 16], frag_O[i], SMEM_STRIDE, wmma::mem_row_major);
    __syncthreads();

    // Normalize and Write to Global
    // Each thread handles 4 vec16 (same as load)
    int total_vecs = TR * (D / 8);
    // 64 * 64 = 4096 half, vec16 = 8 half, 4096/8=512/vec16, 128 threads, 每线程4/vec16
    for (int i = 0; i < 4; ++i)
    {
        int vec_idx = tid + i * BLOCK_DIM;
        if (vec_idx < total_vecs)
        {
            int r = vec_idx >> 3;
            int v = vec_idx & 7;
            int c = v << 3;

            int global_r = q_row_start + r;
            if (global_r < N)
            {
                float l_val = S_L[r];
                float inv_l = 1.0f / (l_val + 1e-6f);

                // Process 8 values in the vector
                half result_buf[8];
                for (int x = 0; x < 8; ++x)
                    result_buf[x] = __float2half(S_O[r][c + x] * inv_l);

                // Vectorized Store
                half *dst = O_ptr + global_r * D + c;
                *reinterpret_cast<vec16 *>(dst) = *reinterpret_cast<vec16 *>(result_buf);
            }
        }
    }
}

// Host Launcher
void launch_fa_64x64(half *Q, half *K, half *V, half *out, int B, int H, int N, int D)
{
    int shared_mem_size = 64 * 1024; // 64KB safety
    cudaFuncSetAttribute(fa_kernel_64x64, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);

    // Grid (网格): 负责处理整个 Batch 和所有 Heads。
    // blockIdx.x: 编码了 (Batch, Head) 的组合，blockIdx.y: 负责 $Q$ 矩阵的行方向分块（TR=64）。
    dim3 grid(B * H, (N + TR - 1) / TR);
    // Block (线程块): 负责计算一个 $64 \times 64$ 的输出块。
    // 线程数: BLOCK_DMI = 128 个线程（4 个 Warps）。任务：计算Q_{tile} \times K^T \times V。
    dim3 block(BLOCK_DIM);

    fa_kernel_64x64<<<grid, block, shared_mem_size>>>(Q, K, V, out, B, H, N, D, 1.0f / sqrtf(D));
}
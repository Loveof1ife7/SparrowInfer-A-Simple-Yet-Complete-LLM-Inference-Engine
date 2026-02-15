#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math.h>

using namespace nvcuda;

// ================= Config =================
#define TR 64
#define TC 64
#define D_MODEL 64
#define BLOCK_DIM 128
#define SMEM_STRIDE (D_MODEL + 8) // Padding for bank conflict

// WMMA Shapes
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

using vec16 = uint4; // 16 bytes = 8 halfs

// ================= Helper Functions =================

// 1. Warp Reduce Max (Group size = 16)
__device__ __forceinline__ float half_warp_reduce_max(float val)
{
#pragma unroll
    for (int offset = 8; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// 2. Warp Reduce Sum (Group size = 16)
__device__ __forceinline__ float half_warp_reduce_sum(float val)
{
#pragma unroll
    for (int offset = 8; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// 3. Async Copy (Global -> Shared) 16 Bytes
__device__ __forceinline__ void cp_async4(void *smem_ptr, const void *gmem_ptr)
{
    // 将指针转换为 shared memory 的 32 位整数偏移量
    unsigned int smem_int = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(smem_int), "l"(gmem_ptr));
}
// 4. Async Copy Commit
__device__ __forceinline__ void cp_async_commit_group()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

// 5. Async Copy Wait
template <int N>
__device__ __forceinline__ void cp_async_wait_group()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// 6. Barrier Sync
__device__ __forceinline__ void barrier_sync()
{
    asm volatile("bar.sync 0;\n" ::);
}

// 7. Loader Template
template <int TILE_SIZE>
__device__ __forceinline__ void load_tile_async(
    half (*smem)[SMEM_STRIDE],
    const half *__restrict__ gmem,
    int row_start, int N, int D, int tid)
{
    const int total_vecs = TILE_SIZE * (D / 8);
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        int vec_idx = tid + i * BLOCK_DIM;
        if (vec_idx < total_vecs)
        {
            int r = vec_idx >> 3;
            int c = (vec_idx & 7) << 3;
            int global_r = row_start + r;

            void *dst = &smem[r][c];
            const void *src = gmem + global_r * D + c;

            if (global_r < N)
            {
                cp_async4(dst, src);
            }
            else
            {
                // Bounds handling: Zero padding manually
                *reinterpret_cast<uint4 *>(dst) = make_uint4(0, 0, 0, 0);
            }
        }
    }
}

// ================= Main Kernel =================

__global__ void fa_warp_double_buffer_kernel(
    const half *__restrict__ Q,
    const half *__restrict__ K,
    const half *__restrict__ V,
    half *__restrict__ out,
    int B, int H, int N, int D,
    float scale)
{
    // --- Index Calculation ---
    int bh = blockIdx.x;
    int tr_idx = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int q_row_start = tr_idx * TR;
    int batch_head_offset = (b * H + h) * N * D;

    const half *Q_ptr = Q + batch_head_offset;
    const half *K_ptr = K + batch_head_offset;
    const half *V_ptr = V + batch_head_offset;
    half *O_ptr = out + batch_head_offset;

    // --- Shared Memory Setup ---
    extern __shared__ char smem_buffer[];

    // S_Q: [64][72] (Static, no double buffer)
    half(*S_Q)[SMEM_STRIDE] = (half(*)[SMEM_STRIDE])smem_buffer;

    // S_K, S_V: [2][64][72] (Double Buffered)
    // 0: Compute Buffer, 1: Load Buffer
    half(*S_K)[TR][SMEM_STRIDE] = (half(*)[TR][SMEM_STRIDE])(S_Q + TR);
    half(*S_V)[TR][SMEM_STRIDE] = (half(*)[TR][SMEM_STRIDE])(S_K + 2);

    // Reuse Memory: After S_K is used for QK^T, we overwrite it with P
    // S_S: Stores float results of QK^T. Placed after S_V.
    float *ptr_float = (float *)(S_V + 2);
    float (*S_S)[TC] = (float (*)[TC])ptr_float;

    // S_O: Accumulator for output.
    float (*S_O)[SMEM_STRIDE] = (float (*)[SMEM_STRIDE])(S_S + TR);

    // Meta stats
    float *S_L = (float *)(S_O + TR);
    float *S_corr = S_L + TR;

    // --- Fragments ---
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_Q[4];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_K[4];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_V[4];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_S[4];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_O[4];
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_P[4];

    // Init Output Accumulator
    for (int i = 0; i < 4; ++i)
        wmma::fill_fragment(frag_O[i], 0.0f);

    float m_i = -1e20f;
    float l_i = 0.0f;

    // --- 1. Load Q (Once) ---
    load_tile_async<TR>(S_Q, Q_ptr, q_row_start, N, D, tid);
    cp_async_commit_group();
    cp_async_wait_group<0>(); // Wait strictly
    barrier_sync();

    // Load Q to Registers
    int warp_row_offset = warp_id * 16;
    for (int k = 0; k < 4; ++k)
        wmma::load_matrix_sync(frag_Q[k], &S_Q[warp_row_offset][k * 16], SMEM_STRIDE);

    // --- 2. Pipeline Prolog (Load 1st Tile) ---
    int load_stage = 0;
    int compute_stage = 0;

    load_tile_async<TC>(S_K[load_stage], K_ptr, 0, N, D, tid);
    load_tile_async<TC>(S_V[load_stage], V_ptr, 0, N, D, tid);
    cp_async_commit_group(); // Commit Batch 0

    // --- 3. Main Pipeline Loop ---
    for (int j = 0; j < N; j += TC)
    {
        // === Stage A: Pre-fetch Next Tile ===
        int next_j = j + TC;
        if (next_j < N)
        {
            int next_load_stage = 1 - load_stage;
            load_tile_async<TC>(S_K[next_load_stage], K_ptr, next_j, N, D, tid);
            load_tile_async<TC>(S_V[next_load_stage], V_ptr, next_j, N, D, tid);
            cp_async_commit_group();
            load_stage = next_load_stage;
        }

        // === Stage B: Wait for Current Tile ===
        // We need the oldest batch (compute_stage) to be ready.
        // If we prefetched, queue has 2 batches. wait_group<1> ensures oldest is done.
        cp_async_wait_group<1>();
        barrier_sync(); // Ensure S_K[compute_stage] is visible

        // === Stage C: Compute Q * K^T ===
        for (int i = 0; i < 4; ++i)
            wmma::fill_fragment(frag_S[i], 0.0f);

        for (int k_chunk = 0; k_chunk < 4; ++k_chunk)
        {
            for (int sub_col = 0; sub_col < 4; ++sub_col)
            {
                // Read from S_K[compute_stage]
                const half *k_ptr = &S_K[compute_stage][sub_col * 16][k_chunk * 16];
                wmma::load_matrix_sync(frag_K[sub_col], k_ptr, SMEM_STRIDE);
                wmma::mma_sync(frag_S[sub_col], frag_Q[k_chunk], frag_K[sub_col], frag_S[sub_col]);
            }
        }

        // Store S (float) to Shared Memory for Softmax
        for (int i = 0; i < 4; ++i)
        {
            float *s_ptr = &S_S[warp_row_offset][i * 16];
            wmma::store_matrix_sync(s_ptr, frag_S[i], TC, wmma::mem_row_major);
        }
        barrier_sync(); // Wait for S_S write

        // === Stage D: Softmax & Update Acc & Write P ===
        // 关键：复用 S_K[compute_stage] 作为 S_P 缓冲区
        half *S_P_buffer = (half *)S_K[compute_stage];

        // Apply Correction to previous O (Reading from S_O, Writing back)
        // 修正逻辑必须在计算新 P 之前完成，但只需要拿到 Correction 值
        // 为了避免复杂的依赖，我们在此处计算 Softmax，拿到 correction，立刻修正 S_O

        // Warp-level Softmax Loop
        for (int i = 0; i < 8; i++)
        {
            int row_idx_in_warp = i * 2 + (lane_id / 16);
            int row = warp_row_offset + row_idx_in_warp;
            int lane_group = lane_id % 16;

            // 1. Load S & Local Max
            float val[4];
            float local_max = -1e20f;
#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                val[k] = S_S[row][lane_group + k * 16] * scale;
                local_max = fmaxf(local_max, val[k]);
            }

            // 2. Reduce Max
            float row_max = half_warp_reduce_max(local_max);
            row_max = __shfl_sync(0xffffffff, row_max, (lane_id / 16) * 16);

            // 3. Update Global Max & Calc Correction
            float m_prev = m_i;
            float m_new = fmaxf(m_prev, row_max);
            float correction = (m_prev <= -1e10f) ? 0.0f : expf(m_prev - m_new);

            // 首领更新全局 correction (S_corr 仍需用于修正 S_O)
            if (lane_group == 0)
                S_corr[row] = correction;

            // 4. Calc Exp & Local Sum
            float local_sum = 0.0f;
#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                val[k] = expf(val[k] - m_new);
                local_sum += val[k];
            }

            // 5. Reduce Sum & Update Global Sum
            float row_sum = half_warp_reduce_sum(local_sum);
            row_sum = __shfl_sync(0xffffffff, row_sum, (lane_id / 16) * 16);

            l_i = l_i * correction + row_sum;
            m_i = m_new;
            if (lane_group == 0)
                S_L[row] = l_i;

// 6. [核心复用] Write P (half) directly to S_K[compute_stage]
#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                S_P_buffer[row * SMEM_STRIDE + (lane_group + k * 16)] = __float2half(val[k]);
            }
        }

        barrier_sync(); // 1. S_corr Ready, 2. P (in S_K) Ready

        // === Stage E: Correct S_O ===
        // 先把之前的 O 修正了，再加新的 PV
        for (int i = 0; i < 4; ++i)
            wmma::store_matrix_sync(&S_O[warp_row_offset][i * 16], frag_O[i], SMEM_STRIDE, wmma::mem_row_major);
        barrier_sync();

        for (int k = 0; k < 32; ++k)
        {
            int idx = lane_id + k * 32;
            if (idx < 16 * 64)
            {
                int r = idx / 64;
                int c = idx % 64;
                int global_r = warp_row_offset + r;
                S_O[global_r][c] *= S_corr[global_r];
            }
        }
        barrier_sync();

        // Reload Corrected O to Registers
        for (int i = 0; i < 4; ++i)
            wmma::load_matrix_sync(frag_O[i], &S_O[warp_row_offset][i * 16], SMEM_STRIDE, wmma::mem_row_major);

        // === Stage F: Compute O += P * V ===
        // 此时 S_K[compute] 存的是 P, S_V[compute] 存的是 V

        // Load P fragments (from S_K/P buffer)
        for (int k = 0; k < 4; ++k)
        {
            half *p_tile = S_P_buffer + (warp_row_offset * SMEM_STRIDE) + (k * 16);
            wmma::load_matrix_sync(frag_P[k], p_tile, SMEM_STRIDE);
        }

        // MMA Loop
        for (int k_chunk = 0; k_chunk < 4; ++k_chunk)
        {
            for (int sub_col = 0; sub_col < 4; ++sub_col)
            {
                // Read V from S_V[compute_stage]
                const half *v_ptr = &S_V[compute_stage][k_chunk * 16][sub_col * 16];
                wmma::load_matrix_sync(frag_V[sub_col], v_ptr, SMEM_STRIDE);
                // O += P_chunk * V_chunk
                wmma::mma_sync(frag_O[sub_col], frag_P[k_chunk], frag_V[sub_col], frag_O[sub_col]);
            }
        }

        // === Stage G: Release Buffer ===
        // 确保本轮所有对 compute_stage buffer 的读取都已完成
        // 然后才能释放给下一轮作为 load 目标
        barrier_sync();

        compute_stage = 1 - compute_stage;
    }

    // End of Loop: Wait for all async ops
    cp_async_wait_group<0>();
    barrier_sync();

    // --- Final Store (Normalize & Write) ---
    for (int i = 0; i < 4; ++i)
        wmma::store_matrix_sync(&S_O[warp_row_offset][i * 16], frag_O[i], SMEM_STRIDE, wmma::mem_row_major);
    barrier_sync();

    const int total_vecs = TR * (D / 8);
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
                half result_buf[8];
                for (int x = 0; x < 8; ++x)
                    result_buf[x] = __float2half(S_O[r][c + x] * inv_l);

                half *dst = O_ptr + global_r * D + c;
                *reinterpret_cast<vec16 *>(dst) = *reinterpret_cast<vec16 *>(result_buf);
            }
        }
    }
}

// Host Launcher

void launch_fa_double_buffer(half *Q, half *K, half *V, half *out, int B, int H, int N, int D)
{
    // ✅ 修改 1: 显式计算需要的 SMEM 大小 (给 96KB 安全空间)
    // 81KB 实际需求，给 96KB 是为了对齐和安全
    int shared_mem_size = 96 * 1024;

    // ✅ 修改 2: 告诉驱动这个 Kernel 需要更大的 Shared Memory
    // 默认限制是 48KB，超过必须设置 Attribute
    cudaFuncSetAttribute(
        fa_warp_double_buffer_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem_size);

    dim3 grid(B * H, (N + TR - 1) / TR);
    dim3 block(BLOCK_DIM);

    // ✅ 修改 3: 传入新的大小
    fa_warp_double_buffer_kernel<<<grid, block, shared_mem_size>>>(
        Q, K, V, out, B, H, N, D, 1.0f / sqrtf(D));
}
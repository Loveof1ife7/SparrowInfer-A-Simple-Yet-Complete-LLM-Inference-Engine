// src/kernels/fa_vectorized.cu
// GPU 的显存总线一次处理 32 字节 (sector)，L2/L1 cache line 也是 128 字节。发起一次 2 字节的加载请求和发起一次 16 字节的请求，延迟几乎是一样的，但吞吐量差了 8 倍。
// 优化方案： 使用 float4 (128-bit) 或 float2 (64-bit) 进行加载。
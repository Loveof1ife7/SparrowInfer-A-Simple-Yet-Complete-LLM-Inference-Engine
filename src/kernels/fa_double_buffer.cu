// 改进一
// 增大 Tile Size 与 多 Warp 协作(Coarsening)
// 目前的 TR = 16, TC = 16 太小了。
// 显存效率低：虽然你用了 vec16，但每次只搬运一小块，循环头部开销占比大。
// 计算强度低：Tensor Core 还没热身完，计算就结束了。
// 并行度低：你只用了 blockDim.x = 32(1个 Warp)。GPU 的一个 SM 通常需要至少 4 - 8 个 Warp 才能有效掩盖延迟。

// 改进二
// 串行 [Load 100ns] -> [Compute 50ns] -> [Load 100ns] -> [Compute 50ns] = 300ns
// 双缓冲
// [Load Tile 0]
//              [Compute Tile 0]
//              [Load Tile 1   ]  <-- 几乎免费！因为是在计算的同时加载
//                              [Compute Tile 1]
//                              [Load Tile 2   ]
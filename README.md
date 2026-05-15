# mini-flash-attention

FlashAttention kernel
→ PyTorch custom op
→ Llama attention layer
→ KV cache decode
→ PagedAttention
→ continuous batching
→ mini LLM inference runtime

## 主线

**1. FlashAttention kernel → 2. PyTorch custom op**

"封装"。手写的 CUDA/Triton kernel 封装成 PyTorch 的 `autograd.Function` 或 `torch.nn.Module`，才能被上层模型调用。

**2. PyTorch custom op → 3. Llama attention layer**

"最小可用单元"。把 custom op 放进 Llama 的 attention 模块，替换掉原本的 `scaled_dot_product_attention`。跑通单层前向，和 HuggingFace 对比误差。

**3. Llama attention layer → 4. KV Cache decode**

"推理专有优化"。训练时不需要 KV Cache（反向传播需要完整序列），但推理时每生成一个新 token 都要重新计算 attention，不缓存就是 O(n²) 的重复计算。

**4. KV Cache decode → 5. PagedAttention**

这是"工程化 KV Cache"。普通 KV Cache 是连续分配的，不同请求的序列长度不同会导致严重碎片化。PagedAttention 把 KV Cache 切成固定大小的 page，按需分配、按页表索引。

**5. PagedAttention → 6. Continuous Batching**

这是"吞吐优化"。静态 batching 的问题：一个 batch 里只要有一个请求没生成完，其他请求都得等。Continuous Batching 每步重新组 batch，完成的请求退出，新请求加入。这一步本质是调度问题——如何在每步选择最优的请求组合。

**6. Continuous Batching → 7. Mini LLM inference runtime**

前三步是算子层，4-5 是运行时层，6 是调度层。三层拼在一起，就是一个完整的推理引擎。对外暴露一个 `generate(prompt)` 接口，内部走完 prefill → decode 循环 → 采样 → 输出 token。


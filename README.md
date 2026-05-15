# SparrowInfer: A Simple Yet Complete LLM Inference Engine

SparrowInfer is a **from-scratch, component-by-component** implementation of a large language model inference engine. It exists to answer a single question with rigor: *What does it take to serve an LLM, end to end, without relying on any black-box inference framework?* The project does not aim to be production-grade. It aims to be **complete in depth**—every component is written, understood, profiled, and justified. The name comes from the Chinese proverb *麻雀虽小，五脏俱全*: a sparrow is small, but it has all five vital organs. SparrowInfer is small in scale but complete in architecture.

The engine targets **Llama-family models** (Llama-3.2-1B for development, scaling to 3B/8B for performance validation). It is built in Python with performance-critical kernels written in CUDA and Triton. The project is a personal AI infrastructure portfolio piece, designed to demonstrate end-to-end systems thinking from GPU kernel to serving scheduler.

## 主线

FlashAttention kernel
→ PyTorch custom op
→ Llama attention layer
→ KV cache decode
→ PagedAttention
→ continuous batching
→ mini LLM inference runtime

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

### Design Philosophy

**1. Depth over breadth.** SparrowInfer supports exactly one model architecture (Llama) and one inference paradigm (autoregressive decoding). This constraint frees the project from the complexity of multi-model abstraction layers and allows every component to be optimized and understood at the metal level. It is better to serve one model perfectly than ten models poorly.

**2. No black boxes.** Every dependency is either eliminated or fully understood. If the project uses a PyTorch `autograd.Function`, the underlying CUDA kernel is written by hand. If it uses a memory pool, the allocation strategy is designed from scratch. The principle is: you cannot claim to understand an inference engine if you cannot explain every layer's memory layout, launch configuration, and performance bottleneck.

**3. Numbers drive decisions.** Every design choice is accompanied by a benchmark. Kernel tiling sizes are chosen based on profiling, not convention. Page size for KV cache is justified by measuring internal fragmentation against page table overhead. Continuous batching scheduling policy is evaluated on throughput-latency tradeoff curves. The project produces evidence, not assertions.

**4. Correctness before performance.** A fast kernel that produces wrong results is worthless. Every component is tested against a reference implementation (PyTorch eager mode or HuggingFace Transformers) with explicit error tolerance thresholds. Integration tests verify end-to-end token-for-token equivalence before any optimization is attempted.

**5. Complexity is a cost.** The project adds abstraction only when the absence of abstraction would cause more pain than its presence. A flat module is preferred until the flatness becomes unmanageable. This discipline prevents the over-engineering that plagues many "learning projects" that copy production frameworks without inheriting their scale-driven necessity.

### Architectural Logic

The engine is organized into four layers, each building on the one below it. The layering reflects a fundamental dependency: schedulers need KV caches, KV caches need models, models need operators. No layer knows about the layer above it.

**Layer 1: Operators.** This is the numerical foundation. It contains hand-written GPU kernels for the compute-intensive primitives that dominate LLM inference: FlashAttention (forward), RMSNorm, Rotary Position Embedding (RoPE), and SwiGLU activation. Each kernel is wrapped as a PyTorch custom op so it can be called like any other `torch.nn.Module`. Each kernel ships with a Roofline analysis that identifies whether it is memory-bound or compute-bound, and therefore what kind of optimization would actually help. This layer answers: *are we using the hardware correctly?*

**Layer 2: Model.** This layer instantiates a complete Llama model by composing operators from Layer 1 with standard PyTorch linear layers. It includes a weight loader that reads HuggingFace-format safetensors and maps their keys to the internal model structure. The model is verified layer-by-layer against HuggingFace Transformers output. Critically, this layer implements only forward propagation—there is no backward pass, no optimizer, no training logic. This constraint simplifies the design and reflects the inference-only scope. This layer answers: *does our model compute the right logits?*

**Layer 3: Runtime.** This layer introduces the key data structure that separates inference from training: the Key-Value Cache. It implements a **paged KV cache** inspired by vLLM's PagedAttention, where the cache is divided into fixed-size pages and requests hold page tables mapping logical positions to physical pages. A pre-allocated memory pool eliminates runtime allocation. The autoregressive decode loop is implemented here, with distinct prefill (process all prompt tokens, populate cache) and decode (process one token, append to cache, sample) phases. This layer answers: *how do we manage memory so that the model can generate tokens efficiently?*

**Layer 4: Engine.** This layer implements **continuous batching**, the scheduling policy that distinguishes a serving system from a model script. Rather than waiting for an entire batch of requests to complete before admitting new ones, the scheduler re-evaluates the batch composition at every decode step: finished requests exit, waiting requests enter. This maximizes throughput by ensuring the GPU never sits idle waiting for a single slow request. The scheduler is the only component that maintains global state across requests. This layer answers: *how do we serve multiple users simultaneously without wasting compute?*


### Development Plan

The project proceeds in four phases. Each phase produces a runnable, testable artifact. Phases are sequential: each depends on the one before it. The estimated timeline assumes part-time effort (15-20 hours per week).

**Phase 1: Operators (Weeks 1-3).**
The goal is a library of hand-written GPU kernels that can replace PyTorch's built-in operations for Llama inference. This phase produces the project's technical foundation and its deepest demonstration of CUDA/Triton proficiency.

- Implement FlashAttention forward with online softmax and tiling. Verify against `torch.nn.functional.scaled_dot_product_attention` across multiple sequence lengths and batch sizes.
- Implement RMSNorm with warp-level reduction. Verify against PyTorch.
- Implement RoPE with the option to fuse into the attention kernel (stretch goal).
- Implement fused SwiGLU (SiLU activation followed by element-wise multiplication).
- For each operator: produce a Roofline plot, a latency-vs-sequence-length benchmark, and a written analysis of the bottleneck.
- Deliverable: a Python module `sparrow_infer.operators` where each operator is importable and tested. A benchmark script that compares against PyTorch native ops.

**Phase 2: Model (Weeks 4-5).**
The goal is a complete Llama model that loads real weights from HuggingFace and produces correct logits. This phase validates that the operators from Phase 1 integrate correctly into a real model structure.

- Implement `sparrow_infer.model.config.LlamaConfig` with all architectural hyperparameters.
- Implement `sparrow_infer.model.loader.WeightLoader` for safetensors, including key name mapping from HuggingFace conventions to internal names.
- Implement `sparrow_infer.model.llama.LlamaForCausalLM` composing Embedding → N×DecoderLayer → LM Head.
- Implement `sparrow_infer.model.llama.DecoderLayer` using RMSNorm, FlashAttention (from Phase 1), and SwiGLU MLP.
- Write a correctness test that loads Llama-3.2-1B from HuggingFace, runs the same input through both implementations, and asserts layer-by-layer output matching within 1e-3 tolerance.
- Deliverable: a Python script that loads a real model, runs inference on a single prompt, and outputs tokens that match HuggingFace exactly.

**Phase 3: Runtime (Weeks 6-8).**
The goal is an autoregressive decoding loop backed by a paged KV cache and a pre-allocated memory pool. This phase transforms the model from a single-prompt executor into a multi-step generator with efficient memory management.

- Design and implement `sparrow_infer.runtime.memory_pool.MemoryPool`: allocate a large contiguous block of GPU memory, partition into fixed-size pages, maintain free-list and used-list.
- Design and implement `sparrow_infer.runtime.kv_cache.PagedKVCache`: for each request, maintain a page table; store K and V tensors per layer in physical pages; support reading/writing by logical position via page table lookup.
- Implement `sparrow_infer.runtime.decode.DecodeLoop`: prefill phase (batch-process all prompt tokens, store K/V, return final logits), decode phase (process one token per step using cached K/V), sampling (temperature, top-p, top-k), stop condition (EOS token or max length).
- Justify page size selection with measurements: internal fragmentation vs. page table memory overhead vs. kernel efficiency.
- Deliverable: a script that takes a prompt, generates tokens one at a time, and matches HuggingFace's `model.generate()` output. A memory usage report showing total GPU memory consumed vs. HuggingFace baseline.

**Phase 4: Engine (Weeks 9-11).**
The goal is a continuous batching scheduler that serves multiple concurrent requests. This phase turns the project from a single-user tool into a multi-user inference server.

- Implement `sparrow_infer.engine.scheduler.Scheduler`: maintain three queues (waiting, running, finished); at each decode step, select which requests to include in the next forward pass; handle prefill-decode mixed batches; manage per-request state (tokens generated so far, stopping condition met).
- Implement batch assembly: combine multiple requests' input tokens into a single batched tensor; build a combined page table that the attention kernel can index into.
- Write a benchmark that measures throughput (tokens/second) and latency (time-to-first-token, time-per-output-token) under varying request arrival rates.
- Optional stretch goal: implement a minimal HTTP API using FastAPI so the engine can be called as a service.
- Deliverable: a script that submits 10+ requests concurrently, the scheduler processes them with continuous batching, and a performance report comparing against sequential (non-batched) execution. A written analysis of the throughput-latency tradeoff.





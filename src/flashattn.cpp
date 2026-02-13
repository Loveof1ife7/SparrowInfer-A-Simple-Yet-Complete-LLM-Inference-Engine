// PyTorch C++ binding

#include <torch/extension.h>

torch::Tensor flash_attention_forward_v1(torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor flash_attention_forward_v2(torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor flash_attention_forward_v3(torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor flash_attention_forward_v4(torch::Tensor q, torch::Tensor k, torch::Tensor v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("flash_attention_forward_v1", &flash_attention_forward_v1, "Flash Attention forward v1 (CUDA): tiled implementation");
    m.def("flash_attention_forward_v2", &flash_attention_forward_v2, "Flash Attention forward v2 (CUDA): tiled implementation with WMMA");
    m.def("flash_attention_forward_v3", &flash_attention_forward_v3, "Flash Attention forward v3 (CUDA): vectorized loads with WMMA");
    m.def("flash_attention_forward_v4", &flash_attention_forward_v4, "Flash Attention forward v4 (CUDA): scaled-up vectorized 64x64 with WMMA");
}
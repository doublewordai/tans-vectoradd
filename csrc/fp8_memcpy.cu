// Pybind module for the tANS FP8 vector-add prototype.

#include <torch/extension.h>
#include <tuple>

// Forward declarations implemented in the other translation units.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
tans_encode_interleaved(torch::Tensor, torch::Tensor, torch::Tensor);

torch::Tensor gpu_tans_decode(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, int64_t);

torch::Tensor fp8_vecadd_raw(torch::Tensor, torch::Tensor);

torch::Tensor fp8_vecadd_fused_tans(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, int64_t);

torch::Tensor fp8_vecadd_fused_tans_shared(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, int64_t);

torch::Tensor fp8_vecadd_fused_tans_register(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, int64_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tans_encode_interleaved", &tans_encode_interleaved,
          "CPU tANS encoder -> GPU-ready interleaved slab layout.");
    m.def("gpu_tans_decode", &gpu_tans_decode,
          "GPU tANS pair decoder. Returns [n_fp8_per_stream, n_streams] uint8.");
    m.def("fp8_vecadd_raw", &fp8_vecadd_raw,
          "Uncompressed FP8 vector add: C[i] = A[i] + B[i].");
    m.def("fp8_vecadd_fused_tans", &fp8_vecadd_fused_tans,
          "Fused tANS decompress + FP8 vector add selected by env.");
    m.def("fp8_vecadd_fused_tans_shared", &fp8_vecadd_fused_tans_shared,
          "Fused tANS vecadd with shared-memory decoded symbol staging.");
    m.def("fp8_vecadd_fused_tans_register", &fp8_vecadd_fused_tans_register,
          "Fused tANS vecadd with register decoded symbol staging.");
}

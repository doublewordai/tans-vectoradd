// Pybind module for the tANS FP8 vector-add prototype.
//
// Also hosts the FP8 memcpy kernel: a thin uint4 load/store that mirrors the
// access pattern of fp8_vecadd_raw_kernel but with no compute and only one
// input. Acts as the practical HBM-bandwidth ceiling for SM-launched
// kernels of this shape.

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
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

namespace {

__global__ void fp8_memcpy_kernel(
    const uint4* __restrict__ A,
    uint4*       __restrict__ C,
    int64_t n_vec)
{
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < n_vec; i += stride) {
        C[i] = A[i];
    }
}

}  // namespace

torch::Tensor fp8_memcpy(torch::Tensor A) {
    TORCH_CHECK(A.is_cuda(), "A must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kUInt8, "A must be uint8");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(A.numel() % 16 == 0, "A length must be a multiple of 16");

    int64_t n_vec = A.numel() / 16;
    auto C = torch::empty_like(A);

    int threads = 256;
    int blocks = std::min<int64_t>((n_vec + threads - 1) / threads, 512);
    blocks = std::max(blocks, 1);

    fp8_memcpy_kernel<<<blocks, threads>>>(
        reinterpret_cast<const uint4*>(A.data_ptr<uint8_t>()),
        reinterpret_cast<uint4*>(C.data_ptr<uint8_t>()),
        n_vec);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tans_encode_interleaved", &tans_encode_interleaved,
          "CPU tANS encoder -> GPU-ready interleaved slab layout.");
    m.def("gpu_tans_decode", &gpu_tans_decode,
          "GPU tANS pair decoder. Returns [n_fp8_per_stream, n_streams] uint8.");
    m.def("fp8_memcpy", &fp8_memcpy,
          "Pure HBM memcpy: C[i] = A[i] via uint4 SM kernel. "
          "Practical bandwidth ceiling reference.");
    m.def("fp8_vecadd_raw", &fp8_vecadd_raw,
          "Uncompressed FP8 vector add: C[i] = A[i] + B[i].");
    m.def("fp8_vecadd_fused_tans", &fp8_vecadd_fused_tans,
          "Fused tANS decompress + FP8 vector add.");
}

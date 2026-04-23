// Pybind module + HBM-read-only memcpy reference.

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void fp8_memcpy_digest_kernel(
    const uint4* __restrict__ src,
    uint8_t*     __restrict__ digest_out,
    int64_t n_vec)
{
    int64_t tid    = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x  * blockDim.x;
    uint4 acc; acc.x = acc.y = acc.z = acc.w = 0;
    for (int64_t i = tid; i < n_vec; i += stride) {
        uint4 v = src[i];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    uint32_t folded = acc.x ^ acc.y ^ acc.z ^ acc.w;
    folded ^= folded >> 16;
    folded ^= folded >> 8;
    digest_out[tid] = (uint8_t)(folded & 0xFF);
}

torch::Tensor fp8_memcpy_digest(torch::Tensor src) {
    TORCH_CHECK(src.is_cuda() && src.dtype() == torch::kUInt8,
                "uint8 CUDA tensor required");
    TORCH_CHECK(src.is_contiguous(), "contiguous required");
    TORCH_CHECK(src.numel() % 16 == 0, "numel must be a multiple of 16");
    int64_t n_vec = src.numel() / 16;

    const int threads = 256;
    int blocks = 512;
    int64_t cap = (n_vec + threads - 1) / threads;
    if ((int64_t)blocks > cap) blocks = (int)cap;
    if (blocks < 1) blocks = 1;

    auto digest = torch::empty(
        {(int64_t)blocks * threads},
        torch::TensorOptions().dtype(torch::kUInt8).device(src.device()));

    fp8_memcpy_digest_kernel<<<blocks, threads>>>(
        reinterpret_cast<const uint4*>(src.data_ptr<uint8_t>()),
        digest.data_ptr<uint8_t>(),
        n_vec);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return digest;
}

// Forward declarations
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    rans_encode_interleaved(torch::Tensor, torch::Tensor, int64_t);
torch::Tensor gpu_rans_decode(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
    torch::Tensor);
torch::Tensor gpu_rans_decode_dump(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
    torch::Tensor);
torch::Tensor fp8_vecadd_raw(torch::Tensor, torch::Tensor);
torch::Tensor fp8_vecadd_fused(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, int64_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rans_encode_interleaved", &rans_encode_interleaved,
          "CPU rANS encoder → GPU-ready interleaved slab layout.");
    m.def("gpu_rans_decode", &gpu_rans_decode,
          "GPU rANS pair decoder. Returns [n_fp8_per_stream, n_streams] uint8.");
    m.def("gpu_rans_decode_dump", &gpu_rans_decode_dump,
          "Benchmark variant: XOR digest instead of full output.");
    m.def("fp8_memcpy_digest", &fp8_memcpy_digest,
          "HBM-read-only memcpy reference (XOR digest).");
    m.def("fp8_vecadd_raw", &fp8_vecadd_raw,
          "Uncompressed FP8 vector add: C[i] = A[i] + B[i].");
    m.def("fp8_vecadd_fused", &fp8_vecadd_fused,
          "Fused decompress + FP8 vector add from compressed A and B.");
}

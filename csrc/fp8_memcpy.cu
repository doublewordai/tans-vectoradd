// Pybind module + HBM-read-only memcpy reference.
//
// fp8_memcpy_digest: read N fp8 bytes, XOR-fold them into a per-thread
// digest, write 1 byte per thread. This is the apples-to-apples memcpy
// reference for compressed decoders whose decoded bytes stay in SMEM
// (or registers) for downstream MMA consumption — same HBM access
// pattern as a "load into SMEM" path, without paying for an artificial
// HBM write-back.

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

// Forward declarations of functions defined in the sibling TUs.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    rans_encode_interleaved(torch::Tensor, torch::Tensor, int64_t);
torch::Tensor gpu_rans_decode_fp8(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
    torch::Tensor);
torch::Tensor gpu_rans_decode_fp8_dump(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
    torch::Tensor);
torch::Tensor gpu_rans_decode_fp8_ldg(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
    torch::Tensor);
torch::Tensor gpu_rans_decode_fp8_ldg_dump(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
    torch::Tensor);
torch::Tensor gpu_rans_decode_fp8_pair_ldg(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
    torch::Tensor);
torch::Tensor gpu_rans_decode_fp8_pair_ldg_dump(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
    torch::Tensor);
torch::Tensor gpu_tans_decode_fp8_pair(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, int64_t, int64_t);
torch::Tensor gpu_tans_decode_fp8_pair_dump(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, int64_t, int64_t);
torch::Tensor gpu_tans_decode_fp8(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, int64_t, int64_t);
torch::Tensor gpu_tans_decode_fp8_dump(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, int64_t, int64_t);
torch::Tensor gpu_rans_decode_fp8_triple(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
    torch::Tensor, int64_t);
torch::Tensor gpu_rans_decode_fp8_triple_dump(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
    torch::Tensor, int64_t);
torch::Tensor gpu_rans_decode_fp8_pair_bl(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
    torch::Tensor);
torch::Tensor gpu_rans_decode_fp8_pair_bl_dump(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
    torch::Tensor);
torch::Tensor gpu_rans_decode_fp8_pair_ldg_q4(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
    torch::Tensor);
torch::Tensor gpu_rans_decode_fp8_pair_ldg_q4_dump(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
    torch::Tensor);
torch::Tensor gpu_rans_decode_fp8_regscan(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
    torch::Tensor);
torch::Tensor gpu_rans_decode_fp8_regscan_dump(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t,
    torch::Tensor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rans_encode_interleaved", &rans_encode_interleaved,
          "CPU encoder → GPU-ready interleaved layout. Inputs: symbols "
          "[K, N], freqs [n_alphabet], block_streams. Returns "
          "(compressed, final_states[K], block_offsets[n_blocks+1]).");
    m.def("gpu_rans_decode_fp8", &gpu_rans_decode_fp8,
          "GPU FP8 decompressor. rANS-decodes exponent pairs and composes "
          "with packed sign+mantissa nibbles into FP8 E4M3 bytes. Returns "
          "decoded bytes as [n_fp8_per_stream, n_streams] uint8.");
    m.def("gpu_rans_decode_fp8_dump", &gpu_rans_decode_fp8_dump,
          "Benchmark variant of gpu_rans_decode_fp8: folds decoded bytes "
          "into a per-thread XOR digest (tiny HBM write), for measuring "
          "throughput when the downstream consumer lives in SMEM.");
    m.def("gpu_rans_decode_fp8_ldg", &gpu_rans_decode_fp8_ldg,
          "L1-cache decoder: same sfc table as exact, but read via __ldg "
          "through L1 instead of SMEM. Returns [n_fp8_per_stream, n_streams].");
    m.def("gpu_rans_decode_fp8_ldg_dump", &gpu_rans_decode_fp8_ldg_dump,
          "Bench-only dump variant of gpu_rans_decode_fp8_ldg.");
    m.def("gpu_rans_decode_fp8_pair_ldg", &gpu_rans_decode_fp8_pair_ldg,
          "Pair-alphabet decoder: 256-symbol M=4096, two nibbles per step, "
          "__ldg through L1. Returns [n_fp8_per_stream, n_streams] uint8.");
    m.def("gpu_rans_decode_fp8_pair_ldg_dump", &gpu_rans_decode_fp8_pair_ldg_dump,
          "Bench-only dump variant of gpu_rans_decode_fp8_pair_ldg.");
    m.def("gpu_tans_decode_fp8_pair", &gpu_tans_decode_fp8_pair,
          "Pair-tANS decoder: 256-symbol pairs, tANS table. Returns [n_fp8, n_streams].");
    m.def("gpu_tans_decode_fp8_pair_dump", &gpu_tans_decode_fp8_pair_dump,
          "Bench-only dump variant of pair-tANS decoder.");
    m.def("gpu_tans_decode_fp8", &gpu_tans_decode_fp8,
          "tANS decoder: table lookup, no multiply. Returns [n_fp8, n_streams].");
    m.def("gpu_tans_decode_fp8_dump", &gpu_tans_decode_fp8_dump,
          "Bench-only dump variant of tANS decoder.");
    m.def("gpu_rans_decode_fp8_triple", &gpu_rans_decode_fp8_triple,
          "Joint triple decoder: 4096 symbols, uint64 sfc entries, 3 nibbles/step.");
    m.def("gpu_rans_decode_fp8_triple_dump", &gpu_rans_decode_fp8_triple_dump,
          "Bench-only dump variant of joint triple decoder.");
    m.def("gpu_rans_decode_fp8_pair_bl", &gpu_rans_decode_fp8_pair_bl,
          "Branchless-renorm pair decoder. Returns [n_fp8, n_streams] uint8.");
    m.def("gpu_rans_decode_fp8_pair_bl_dump", &gpu_rans_decode_fp8_pair_bl_dump,
          "Bench-only dump variant of branchless-renorm pair decoder.");
    m.def("gpu_rans_decode_fp8_pair_ldg_q4", &gpu_rans_decode_fp8_pair_ldg_q4,
          "Quad-stream pair decoder: 4 streams/thread for more ILP.");
    m.def("gpu_rans_decode_fp8_pair_ldg_q4_dump", &gpu_rans_decode_fp8_pair_ldg_q4_dump,
          "Bench-only dump variant of quad-stream pair decoder.");
    m.def("gpu_rans_decode_fp8_regscan", &gpu_rans_decode_fp8_regscan,
          "Register-scan decoder: replaces the 8 KB sfc SMEM table with a "
          "17-entry cum table in registers. Branchless linear scan, zero "
          "SMEM bank conflicts. Returns [n_fp8_per_stream, n_streams] uint8.");
    m.def("gpu_rans_decode_fp8_regscan_dump", &gpu_rans_decode_fp8_regscan_dump,
          "Bench-only dump variant of gpu_rans_decode_fp8_regscan.");
    m.def("fp8_memcpy_digest", &fp8_memcpy_digest,
          "HBM-read-only memcpy reference: fold N fp8 bytes into a "
          "per-thread XOR digest. Apples-to-apples baseline for "
          "compressed decoders staging into SMEM.");
}

// Fused decompress + FP8 vector add.
//
// Reads compressed A and B tensors, decodes both via rANS, adds
// element-wise as FP8 E4M3, writes result C. Compared against an
// uncompressed baseline that reads raw A + raw B + writes C.
//
// If decompress throughput exceeds the HBM savings from compression,
// the fused kernel is faster than the uncompressed baseline — this
// is "bandwidth amplification."

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <stdint.h>

// ─── Uncompressed FP8 vector add baseline ────────────────────────────

__global__ void fp8_vecadd_raw_kernel(
    const __nv_fp8_e4m3* __restrict__ A,
    const __nv_fp8_e4m3* __restrict__ B,
    __nv_fp8_e4m3*       __restrict__ C,
    int64_t n)
{
    int64_t tid    = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x  * blockDim.x;
    for (int64_t i = tid; i < n; i += stride) {
        C[i] = __nv_fp8_e4m3(float(A[i]) + float(B[i]));
    }
}

torch::Tensor fp8_vecadd_raw(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kUInt8);
    TORCH_CHECK(B.is_cuda() && B.dtype() == torch::kUInt8);
    TORCH_CHECK(A.numel() == B.numel());
    int64_t n = A.numel();

    auto C = torch::empty_like(A);
    const int threads = 256;
    int blocks = std::min((int)((n + threads - 1) / threads), 512);

    fp8_vecadd_raw_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(A.data_ptr<uint8_t>()),
        reinterpret_cast<const __nv_fp8_e4m3*>(B.data_ptr<uint8_t>()),
        reinterpret_cast<__nv_fp8_e4m3*>(C.data_ptr<uint8_t>()), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}

// ─── Fused decompress + vector add ───────────────────────────────────
//
// Each thread decodes one stream from A and one from B. Per pair
// decode step: decode pair_a (2 nibbles), decode pair_b (2 nibbles),
// compose 2 fp8 bytes from each, add pairwise, write 2 result bytes.
//
// Both A and B share the same sfc table (same distribution).

// Import the decoder's constants and context
constexpr int      M_LOG         = 12;
constexpr uint32_t M_SIZE        = 1u << M_LOG;
constexpr uint32_t L_LOW         = 1u << 16;
constexpr int      BLOCK_STREAMS = 128;

struct RansCtx {
    uint32_t x;
    int      slab_idx;
    uint32_t buf_hi;
    uint32_t buf_lo;
    int      buf_avail;
    int32_t  block_base_u32;
};

// Inline decode step (same as rans_decode.cu)
#define DECODE_STEP(CTX, SFC, COMPRESSED, TID, SYM) do {               \
    uint32_t slot_ = CTX.x & (M_SIZE-1);                               \
    uint32_t entry_ = __ldg(&SFC[slot_]);                              \
    SYM = (uint8_t)(entry_ & 0xFF);                                    \
    uint32_t f_=(entry_>>8)&0xFFF, c_=(entry_>>20)&0xFFF;             \
    CTX.x = (CTX.x >> M_LOG) * f_ + (slot_ - c_);                     \
    bool need_ = (CTX.x < L_LOW);                                      \
    if (need_ && CTX.buf_avail == 0) {                                 \
        CTX.slab_idx -= 2;                                             \
        int32_t off_ = CTX.block_base_u32                              \
            + CTX.slab_idx*(int)blockDim.x + (int)threadIdx.x;         \
        CTX.buf_lo=COMPRESSED[off_];                                   \
        CTX.buf_hi=COMPRESSED[off_+(int)blockDim.x];                  \
        CTX.buf_avail=4;                                               \
    }                                                                   \
    uint32_t xr_=(CTX.x<<16)|(CTX.buf_hi>>16);                        \
    uint32_t bh_=(CTX.buf_hi<<16)|(CTX.buf_lo>>16);                   \
    uint32_t bl_=CTX.buf_lo<<16; int av_=CTX.buf_avail-1;             \
    CTX.x=need_?xr_:CTX.x; CTX.buf_hi=need_?bh_:CTX.buf_hi;         \
    CTX.buf_lo=need_?bl_:CTX.buf_lo; CTX.buf_avail=need_?av_:CTX.buf_avail; \
} while(0)

__global__ void fp8_vecadd_fused_kernel(
    // A's compressed data
    const uint32_t* __restrict__ a_compressed,
    const int32_t*  __restrict__ a_offsets,
    const uint32_t* __restrict__ a_states,
    const uint8_t*  __restrict__ a_sm,
    // B's compressed data
    const uint32_t* __restrict__ b_compressed,
    const int32_t*  __restrict__ b_offsets,
    const uint32_t* __restrict__ b_states,
    const uint8_t*  __restrict__ b_sm,
    // Shared sfc table
    const uint32_t* __restrict__ sfc,
    // Output
    uint8_t*        __restrict__ output,
    int64_t n_fp8_per_stream,
    int64_t n_streams)
{
    int tid = threadIdx.x;
    // Each block handles one encoder block — each thread decodes one
    // stream from A and the corresponding stream from B.
    int enc = blockIdx.x;
    int64_t stream_id = (int64_t)enc * blockDim.x + tid;
    if (stream_id >= n_streams) return;

    // Init contexts for A and B
    RansCtx A, B;
    {
        int32_t base = a_offsets[enc], next = a_offsets[enc+1];
        int G = (next-base)/((int)blockDim.x*4);
        A.x = a_states[stream_id]; A.slab_idx = G;
        A.buf_hi=0; A.buf_lo=0; A.buf_avail=0;
        A.block_base_u32 = base/4;
    }
    {
        int32_t base = b_offsets[enc], next = b_offsets[enc+1];
        int G = (next-base)/((int)blockDim.x*4);
        B.x = b_states[stream_id]; B.slab_idx = G;
        B.buf_hi=0; B.buf_lo=0; B.buf_avail=0;
        B.block_base_u32 = base/4;
    }

    int64_t n_pairs = n_fp8_per_stream / 2;
    for (int64_t i = 0; i < n_pairs; i++) {
        // Decode one pair from A and one from B
        uint8_t sym_a, sym_b;
        DECODE_STEP(A, sfc, a_compressed, tid, sym_a);
        DECODE_STEP(B, sfc, b_compressed, tid, sym_b);

        // Read sign+mantissa for both
        uint8_t sm_a = a_sm[i * n_streams + stream_id];
        uint8_t sm_b = b_sm[i * n_streams + stream_id];

        // Compose FP8 bytes from decoded exponent + sign/mantissa
        uint8_t ea0 = sym_a >> 4, ea1 = sym_a & 0xF;
        uint8_t sa0 = sm_a & 0xF, sa1 = sm_a >> 4;
        uint8_t ba0 = ((sa0&8)<<4) | (ea0<<3) | (sa0&7);
        uint8_t ba1 = ((sa1&8)<<4) | (ea1<<3) | (sa1&7);

        uint8_t eb0 = sym_b >> 4, eb1 = sym_b & 0xF;
        uint8_t sb0 = sm_b & 0xF, sb1 = sm_b >> 4;
        uint8_t bb0 = ((sb0&8)<<4) | (eb0<<3) | (sb0&7);
        uint8_t bb1 = ((sb1&8)<<4) | (eb1<<3) | (sb1&7);

        // Reinterpret as FP8, add natively, write
        __nv_fp8_e4m3 fa0, fa1, fb0, fb1;
        memcpy(&fa0, &ba0, 1); memcpy(&fa1, &ba1, 1);
        memcpy(&fb0, &bb0, 1); memcpy(&fb1, &bb1, 1);

        __nv_fp8_e4m3 c0(float(fa0) + float(fb0));
        __nv_fp8_e4m3 c1(float(fa1) + float(fb1));
        memcpy(&output[(2*i)*n_streams + stream_id], &c0, 1);
        memcpy(&output[(2*i+1)*n_streams + stream_id], &c1, 1);
    }
}

// Host launcher for fused kernel
torch::Tensor fp8_vecadd_fused(
    torch::Tensor a_comp, torch::Tensor a_offsets, torch::Tensor a_states,
    torch::Tensor a_sm, torch::Tensor b_comp, torch::Tensor b_offsets,
    torch::Tensor b_states, torch::Tensor b_sm,
    torch::Tensor sfc_table,
    int64_t n_fp8_per_stream)
{
    int64_t n_streams = a_states.numel();
    int64_t n_enc_blocks = (n_streams + BLOCK_STREAMS - 1) / BLOCK_STREAMS;

    auto output = torch::empty(
        {n_fp8_per_stream, n_streams},
        torch::TensorOptions().dtype(torch::kUInt8).device(a_comp.device()));

    fp8_vecadd_fused_kernel<<<n_enc_blocks, BLOCK_STREAMS>>>(
        reinterpret_cast<const uint32_t*>(a_comp.data_ptr<uint8_t>()),
        a_offsets.data_ptr<int32_t>(),
        reinterpret_cast<const uint32_t*>(a_states.data_ptr<int32_t>()),
        a_sm.data_ptr<uint8_t>(),
        reinterpret_cast<const uint32_t*>(b_comp.data_ptr<uint8_t>()),
        b_offsets.data_ptr<int32_t>(),
        reinterpret_cast<const uint32_t*>(b_states.data_ptr<int32_t>()),
        b_sm.data_ptr<uint8_t>(),
        reinterpret_cast<const uint32_t*>(sfc_table.data_ptr<int32_t>()),
        output.data_ptr<uint8_t>(),
        n_fp8_per_stream, n_streams);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

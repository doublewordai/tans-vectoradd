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
//
// Vectorized: loads 16 bytes at a time (uint4), processes 16 FP8
// elements per vector load, writes 16 bytes. Maximizes HBM bandwidth.

__global__ void fp8_vecadd_raw_kernel(
    const uint4* __restrict__ A,
    const uint4* __restrict__ B,
    uint4*       __restrict__ C,
    int64_t n_vec)
{
    int64_t tid    = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x  * blockDim.x;
    for (int64_t i = tid; i < n_vec; i += stride) {
        uint4 va = A[i];
        uint4 vb = B[i];

        // Process 16 FP8 bytes (4 uint32s × 4 bytes each)
        uint4 vc;
        #pragma unroll
        for (int w = 0; w < 4; w++) {
            uint32_t wa = (&va.x)[w];
            uint32_t wb = (&vb.x)[w];
            uint32_t wc = 0;
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                uint8_t ba = (wa >> (b*8)) & 0xFF;
                uint8_t bb = (wb >> (b*8)) & 0xFF;
                __nv_fp8_e4m3 fa, fb;
                memcpy(&fa, &ba, 1);
                memcpy(&fb, &bb, 1);
                __nv_fp8_e4m3 fc(float(fa) + float(fb));
                uint8_t bc;
                memcpy(&bc, &fc, 1);
                wc |= ((uint32_t)bc << (b*8));
            }
            (&vc.x)[w] = wc;
        }
        C[i] = vc;
    }
}

torch::Tensor fp8_vecadd_raw(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kUInt8);
    TORCH_CHECK(B.is_cuda() && B.dtype() == torch::kUInt8);
    TORCH_CHECK(A.numel() == B.numel());
    TORCH_CHECK(A.numel() % 16 == 0, "numel must be multiple of 16");
    int64_t n = A.numel();
    int64_t n_vec = n / 16;

    auto C = torch::empty_like(A);
    const int threads = 256;
    int blocks = std::min((int)((n_vec + threads - 1) / threads), 512);

    fp8_vecadd_raw_kernel<<<blocks, threads>>>(
        reinterpret_cast<const uint4*>(A.data_ptr<uint8_t>()),
        reinterpret_cast<const uint4*>(B.data_ptr<uint8_t>()),
        reinterpret_cast<uint4*>(C.data_ptr<uint8_t>()), n_vec);
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

// Compose one fp8 byte from decoded exponent nibble + sign/mantissa nibble
__device__ __forceinline__ __nv_fp8_e4m3 compose_fp8(uint8_t exp_nib, uint8_t sm_nib) {
    uint8_t byte = ((sm_nib & 8) << 4) | (exp_nib << 3) | (sm_nib & 7);
    __nv_fp8_e4m3 result;
    memcpy(&result, &byte, 1);
    return result;
}

// Multi-stream fused kernel. Each CUDA block handles NS encoder blocks.
// Each thread decodes NS stream pairs (A_q, B_q), issuing 2*NS __ldg
// reads per iteration before consuming any — same interleaving pattern
// as the standalone decoder.
template <int NS>
__global__ void fp8_vecadd_fused_kernel(
    const uint32_t* __restrict__ a_compressed,
    const int32_t*  __restrict__ a_offsets,
    const uint32_t* __restrict__ a_states,
    const uint8_t*  __restrict__ a_sm,
    const uint32_t* __restrict__ b_compressed,
    const int32_t*  __restrict__ b_offsets,
    const uint32_t* __restrict__ b_states,
    const uint8_t*  __restrict__ b_sm,
    const uint32_t* __restrict__ sfc,
    uint8_t*        __restrict__ output,
    int64_t n_fp8_per_stream,
    int64_t n_streams)
{
    int tid = threadIdx.x;
    RansCtx ctxA[NS], ctxB[NS];
    int32_t sid[NS];  // stream IDs
    bool have[NS];

    #pragma unroll
    for (int q = 0; q < NS; q++) {
        int enc = NS * blockIdx.x + q;
        sid[q] = enc * blockDim.x + tid;
        have[q] = sid[q] < n_streams;
        if (have[q]) {
            int32_t ba = a_offsets[enc], na = a_offsets[enc+1];
            int Ga = (na-ba)/((int)blockDim.x*4);
            ctxA[q].x = a_states[sid[q]]; ctxA[q].slab_idx = Ga;
            ctxA[q].buf_hi=0; ctxA[q].buf_lo=0; ctxA[q].buf_avail=0;
            ctxA[q].block_base_u32 = ba/4;

            int32_t bb = b_offsets[enc], nb = b_offsets[enc+1];
            int Gb = (nb-bb)/((int)blockDim.x*4);
            ctxB[q].x = b_states[sid[q]]; ctxB[q].slab_idx = Gb;
            ctxB[q].buf_hi=0; ctxB[q].buf_lo=0; ctxB[q].buf_avail=0;
            ctxB[q].block_base_u32 = bb/4;
        }
    }

    bool any = false;
    #pragma unroll
    for (int q = 0; q < NS; q++) any |= have[q];
    if (!any) return;

    int64_t n_pairs = n_fp8_per_stream / 2;
    for (int64_t i = 0; i < n_pairs; i++) {
        // Phase A: issue ALL 2*NS sfc LDGs before consuming any.
        // Also issue sign_mantissa reads here — they're HBM reads that
        // can overlap with the L1 sfc reads.
        uint32_t entA[NS], entB[NS];
        uint8_t smA[NS], smB[NS];
        #pragma unroll
        for (int q = 0; q < NS; q++) {
            if (have[q]) {
                entA[q] = __ldg(&sfc[ctxA[q].x & (M_SIZE-1)]);
                entB[q] = __ldg(&sfc[ctxB[q].x & (M_SIZE-1)]);
                smA[q] = a_sm[i * n_streams + sid[q]];
                smB[q] = b_sm[i * n_streams + sid[q]];
            }
        }

        // Phase B: consume A, consume B, add, write — per stream pair.
        // By now the sfc reads have had time to return.
        #pragma unroll
        for (int q = 0; q < NS; q++) {
            if (have[q]) {
                // Consume A
                uint32_t slA = ctxA[q].x & (M_SIZE-1);
                uint8_t symA = entA[q] & 0xFF;
                uint32_t fA=(entA[q]>>8)&0xFFF, cA=(entA[q]>>20)&0xFFF;
                ctxA[q].x = (ctxA[q].x >> M_LOG) * fA + (slA - cA);
                bool needA = (ctxA[q].x < L_LOW);
                if (needA && ctxA[q].buf_avail == 0) {
                    ctxA[q].slab_idx -= 2;
                    int32_t off = ctxA[q].block_base_u32 + ctxA[q].slab_idx*(int)blockDim.x + tid;
                    ctxA[q].buf_lo=a_compressed[off]; ctxA[q].buf_hi=a_compressed[off+(int)blockDim.x];
                    ctxA[q].buf_avail=4;
                }
                uint32_t xrA=(ctxA[q].x<<16)|(ctxA[q].buf_hi>>16);
                uint32_t bhA=(ctxA[q].buf_hi<<16)|(ctxA[q].buf_lo>>16);
                uint32_t blA=ctxA[q].buf_lo<<16; int avA=ctxA[q].buf_avail-1;
                ctxA[q].x=needA?xrA:ctxA[q].x; ctxA[q].buf_hi=needA?bhA:ctxA[q].buf_hi;
                ctxA[q].buf_lo=needA?blA:ctxA[q].buf_lo; ctxA[q].buf_avail=needA?avA:ctxA[q].buf_avail;

                // Consume B
                uint32_t slB = ctxB[q].x & (M_SIZE-1);
                uint8_t symB = entB[q] & 0xFF;
                uint32_t fB=(entB[q]>>8)&0xFFF, cB=(entB[q]>>20)&0xFFF;
                ctxB[q].x = (ctxB[q].x >> M_LOG) * fB + (slB - cB);
                bool needB = (ctxB[q].x < L_LOW);
                if (needB && ctxB[q].buf_avail == 0) {
                    ctxB[q].slab_idx -= 2;
                    int32_t off = ctxB[q].block_base_u32 + ctxB[q].slab_idx*(int)blockDim.x + tid;
                    ctxB[q].buf_lo=b_compressed[off]; ctxB[q].buf_hi=b_compressed[off+(int)blockDim.x];
                    ctxB[q].buf_avail=4;
                }
                uint32_t xrB=(ctxB[q].x<<16)|(ctxB[q].buf_hi>>16);
                uint32_t bhB=(ctxB[q].buf_hi<<16)|(ctxB[q].buf_lo>>16);
                uint32_t blB=ctxB[q].buf_lo<<16; int avB=ctxB[q].buf_avail-1;
                ctxB[q].x=needB?xrB:ctxB[q].x; ctxB[q].buf_hi=needB?bhB:ctxB[q].buf_hi;
                ctxB[q].buf_lo=needB?blB:ctxB[q].buf_lo; ctxB[q].buf_avail=needB?avB:ctxB[q].buf_avail;

                // Compose + add + write
                __nv_fp8_e4m3 a0 = compose_fp8(symA >> 4, smA[q] & 0xF);
                __nv_fp8_e4m3 b0 = compose_fp8(symB >> 4, smB[q] & 0xF);
                __nv_fp8_e4m3 c0(float(a0) + float(b0));

                __nv_fp8_e4m3 a1 = compose_fp8(symA & 0xF, smA[q] >> 4);
                __nv_fp8_e4m3 b1 = compose_fp8(symB & 0xF, smB[q] >> 4);
                __nv_fp8_e4m3 c1(float(a1) + float(b1));

                memcpy(&output[(2*i)*n_streams + sid[q]], &c0, 1);
                memcpy(&output[(2*i+1)*n_streams + sid[q]], &c1, 1);
            }
        }
    }
}

// Host launcher — tries NS=1,2,4 via template dispatch
torch::Tensor fp8_vecadd_fused(
    torch::Tensor a_comp, torch::Tensor a_offsets, torch::Tensor a_states,
    torch::Tensor a_sm, torch::Tensor b_comp, torch::Tensor b_offsets,
    torch::Tensor b_states, torch::Tensor b_sm,
    torch::Tensor sfc_table,
    int64_t n_fp8_per_stream,
    int64_t ns = 1)
{
    int64_t n_streams = a_states.numel();
    int64_t n_enc_blocks = (n_streams + BLOCK_STREAMS - 1) / BLOCK_STREAMS;

    auto output = torch::empty(
        {n_fp8_per_stream, n_streams},
        torch::TensorOptions().dtype(torch::kUInt8).device(a_comp.device()));

    #define LAUNCH_FUSED(N) \
        fp8_vecadd_fused_kernel<N><<<(n_enc_blocks+N-1)/N, BLOCK_STREAMS>>>( \
            reinterpret_cast<const uint32_t*>(a_comp.data_ptr<uint8_t>()), \
            a_offsets.data_ptr<int32_t>(), \
            reinterpret_cast<const uint32_t*>(a_states.data_ptr<int32_t>()), \
            a_sm.data_ptr<uint8_t>(), \
            reinterpret_cast<const uint32_t*>(b_comp.data_ptr<uint8_t>()), \
            b_offsets.data_ptr<int32_t>(), \
            reinterpret_cast<const uint32_t*>(b_states.data_ptr<int32_t>()), \
            b_sm.data_ptr<uint8_t>(), \
            reinterpret_cast<const uint32_t*>(sfc_table.data_ptr<int32_t>()), \
            output.data_ptr<uint8_t>(), \
            n_fp8_per_stream, n_streams)

    switch (ns) {
        case 1: LAUNCH_FUSED(1); break;
        case 2: LAUNCH_FUSED(2); break;
        case 3: LAUNCH_FUSED(3); break;
        case 4: LAUNCH_FUSED(4); break;
        default: TORCH_CHECK(false, "ns must be 1-4");
    }
    #undef LAUNCH_FUSED
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

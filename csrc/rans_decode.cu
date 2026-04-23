// GPU rANS decoder for FP8 E4M3 — pair alphabet (256 symbols, M=4096).
//
// Decodes pairs of exponent nibbles per rANS step via __ldg through L1.
// Triple-stream interleaved: issues all 3 streams' __ldg reads before
// consuming any, maximizing L1 pipeline overlap.
// Branchless renorm: predicated state/buffer update, zero warp divergence.

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ─── Configuration ───────────────────────────────────────────────────

constexpr int      M_LOG         = 12;
constexpr uint32_t M_SIZE        = 1u << M_LOG;   // 4096
constexpr uint32_t L_LOW         = 1u << 16;      // 65536
constexpr int      MAX_ALPHA     = 256;            // pair alphabet
constexpr int      BLOCK_STREAMS = 128;            // threads per block
constexpr int      N_STREAMS     = 2;

// ─── Per-stream decode context ───────────────────────────────────────

struct RansCtx {
    uint32_t x;
    int      slab_idx;
    uint32_t buf_hi;
    uint32_t buf_lo;
    int      buf_avail;
    int32_t  block_base_u32;  // int32 suffices for data < 8 GB
};

// ─── sfc table builder ───────────────────────────────────────────────

static torch::Tensor build_sfc_table(
    torch::Tensor compressed,
    torch::Tensor block_offsets,
    torch::Tensor final_states,
    torch::Tensor sign_mantissa,
    int64_t n_fp8_per_stream,
    torch::Tensor freqs,
    int64_t& n_streams_out,
    int64_t& n_enc_blocks_out)
{
    TORCH_CHECK(compressed.is_cuda() && compressed.dtype() == torch::kUInt8);
    TORCH_CHECK(block_offsets.is_cuda() && block_offsets.dtype() == torch::kInt32);
    TORCH_CHECK(final_states.is_cuda() && final_states.dtype() == torch::kInt32);
    TORCH_CHECK(sign_mantissa.is_cuda() && sign_mantissa.dtype() == torch::kUInt8);
    TORCH_CHECK(freqs.dtype() == torch::kInt32);
    TORCH_CHECK(n_fp8_per_stream % 2 == 0);
    TORCH_CHECK(compressed.numel() % 4 == 0);

    int64_t n_streams  = final_states.numel();
    int     n_alphabet = (int)freqs.numel();
    TORCH_CHECK(n_alphabet > 0 && n_alphabet <= MAX_ALPHA);
    TORCH_CHECK(sign_mantissa.dim() == 2
                && sign_mantissa.size(0) == n_fp8_per_stream / 2
                && sign_mantissa.size(1) == n_streams);

    auto freqs_cpu = freqs.cpu().contiguous();
    const int32_t* f_ptr = freqs_cpu.data_ptr<int32_t>();

    std::vector<uint32_t> cum_vec(n_alphabet);
    uint32_t acc = 0;
    for (int i = 0; i < n_alphabet; i++) {
        TORCH_CHECK(f_ptr[i] >= 0);
        cum_vec[i] = acc;
        acc += (uint32_t)f_ptr[i];
    }
    TORCH_CHECK(acc == M_SIZE, "freqs must sum to ", M_SIZE, "; got ", acc);

    auto sfc_cpu = torch::zeros({(int64_t)M_SIZE}, torch::kInt32);
    uint32_t* sfc_ptr = reinterpret_cast<uint32_t*>(sfc_cpu.data_ptr<int32_t>());
    uint32_t pos = 0;
    for (int s = 0; s < n_alphabet; s++) {
        uint32_t f = (uint32_t)f_ptr[s];
        uint32_t c = cum_vec[s];
        TORCH_CHECK(f < 4096u && c < 4096u);
        uint32_t entry = ((uint32_t)s & 0xFFu)
                       | ((f & 0xFFFu) << 8)
                       | ((c & 0xFFFu) << 20);
        for (uint32_t j = 0; j < f; j++) sfc_ptr[pos + j] = entry;
        pos += f;
    }

    int64_t n_enc_blocks = (n_streams + BLOCK_STREAMS - 1) / BLOCK_STREAMS;
    if (n_enc_blocks < 1) n_enc_blocks = 1;
    TORCH_CHECK(block_offsets.numel() == n_enc_blocks + 1);

    n_streams_out    = n_streams;
    n_enc_blocks_out = n_enc_blocks;
    return sfc_cpu.to(compressed.device());
}

// ─── Inline consume: unpack entry, state update, branchless renorm ──

// Consume entry + compose fp8 + fold into digest, all in one scope.
// slot is recomputed (x hasn't changed for this stream yet), sym is a
// local scalar that dies immediately. No arrays needed.
#define DECODE_AND_DIGEST(CTX, ENTRY, TID_S, I, DIGEST) do {           \
    uint32_t slot_ = CTX.x & (M_SIZE-1);                               \
    uint8_t sym_ = (uint8_t)(ENTRY & 0xFF);                            \
    uint32_t f_=(ENTRY>>8)&0xFFF, c_=(ENTRY>>20)&0xFFF;               \
    CTX.x = (CTX.x >> M_LOG) * f_ + (slot_ - c_);                     \
    bool need_ = (CTX.x < L_LOW);                                      \
    if (need_ && CTX.buf_avail == 0) {                                 \
        CTX.slab_idx -= 2;                                             \
        int32_t off_ = CTX.block_base_u32                              \
            + CTX.slab_idx*(int)blockDim.x + (threadIdx.x);                     \
        CTX.buf_lo=compressed_u32[off_];                               \
        CTX.buf_hi=compressed_u32[off_+(int)blockDim.x];              \
        CTX.buf_avail=4;                                               \
    }                                                                   \
    uint32_t xr_=(CTX.x<<16)|(CTX.buf_hi>>16);                        \
    uint32_t bh_=(CTX.buf_hi<<16)|(CTX.buf_lo>>16);                   \
    uint32_t bl_=CTX.buf_lo<<16; int av_=CTX.buf_avail-1;             \
    CTX.x=need_?xr_:CTX.x; CTX.buf_hi=need_?bh_:CTX.buf_hi;         \
    CTX.buf_lo=need_?bl_:CTX.buf_lo; CTX.buf_avail=need_?av_:CTX.buf_avail; \
    uint8_t sm_=sign_mantissa[(I)*n_streams+TID_S];                    \
    uint8_t e0_=sym_>>4,e1_=sym_&0xF,s0_=sm_&0xF,s1_=sm_>>4;        \
    DIGEST ^= ((s0_&8)<<4)|(e0_<<3)|(s0_&7);                          \
    DIGEST ^= ((s1_&8)<<4)|(e1_<<3)|(s1_&7);                          \
} while(0)

// Same but writes output instead of digest.
#define DECODE_AND_WRITE(CTX, ENTRY, TID_S, I, OUT) do {               \
    uint32_t slot_ = CTX.x & (M_SIZE-1);                               \
    uint8_t sym_ = (uint8_t)(ENTRY & 0xFF);                            \
    uint32_t f_=(ENTRY>>8)&0xFFF, c_=(ENTRY>>20)&0xFFF;               \
    CTX.x = (CTX.x >> M_LOG) * f_ + (slot_ - c_);                     \
    bool need_ = (CTX.x < L_LOW);                                      \
    if (need_ && CTX.buf_avail == 0) {                                 \
        CTX.slab_idx -= 2;                                             \
        int32_t off_ = CTX.block_base_u32                              \
            + CTX.slab_idx*(int)blockDim.x + (threadIdx.x);                     \
        CTX.buf_lo=compressed_u32[off_];                               \
        CTX.buf_hi=compressed_u32[off_+(int)blockDim.x];              \
        CTX.buf_avail=4;                                               \
    }                                                                   \
    uint32_t xr_=(CTX.x<<16)|(CTX.buf_hi>>16);                        \
    uint32_t bh_=(CTX.buf_hi<<16)|(CTX.buf_lo>>16);                   \
    uint32_t bl_=CTX.buf_lo<<16; int av_=CTX.buf_avail-1;             \
    CTX.x=need_?xr_:CTX.x; CTX.buf_hi=need_?bh_:CTX.buf_hi;         \
    CTX.buf_lo=need_?bl_:CTX.buf_lo; CTX.buf_avail=need_?av_:CTX.buf_avail; \
    uint8_t sm_=sign_mantissa[(I)*n_streams+TID_S];                    \
    uint8_t e0_=sym_>>4,e1_=sym_&0xF,s0_=sm_&0xF,s1_=sm_>>4;        \
    OUT[(2*(I))*n_streams+TID_S]=((s0_&8)<<4)|(e0_<<3)|(s0_&7);      \
    OUT[(2*(I)+1)*n_streams+TID_S]=((s1_&8)<<4)|(e1_<<3)|(s1_&7);    \
} while(0)

// ─── Triple-interleaved kernels ──────────────────────────────────────

__global__ void rans_decode_kernel(
    const uint32_t* __restrict__ compressed_u32,
    const int32_t*  __restrict__ block_offsets,
    const uint32_t* __restrict__ final_states,
    const uint32_t* __restrict__ sfc_global,
    const uint8_t*  __restrict__ sign_mantissa,
    uint8_t*        __restrict__ output,
    int64_t n_fp8_per_stream,
    int64_t n_streams)
{
    int tid = threadIdx.x;
    RansCtx ctx[N_STREAMS];
    int32_t tid_s[N_STREAMS];
    bool have[N_STREAMS];

    #pragma unroll
    for (int q = 0; q < N_STREAMS; q++) {
        int enc = N_STREAMS * blockIdx.x + q;
        tid_s[q] = (int64_t)enc * blockDim.x + tid;
        have[q] = tid_s[q] < n_streams;
        if (have[q]) {
            int32_t base = block_offsets[enc], next = block_offsets[enc+1];
            int G = (next-base)/((int)blockDim.x*4);
            ctx[q].x = final_states[tid_s[q]]; ctx[q].slab_idx = G;
            ctx[q].buf_hi=0; ctx[q].buf_lo=0; ctx[q].buf_avail=0;
            ctx[q].block_base_u32 = base/4;
        }
    }
    bool any = false;
    #pragma unroll
    for (int q = 0; q < N_STREAMS; q++) any |= have[q];
    if (!any) return;

    int64_t n_pairs = n_fp8_per_stream / 2;
    for (int64_t i = 0; i < n_pairs; i++) {
        // Phase A: issue all LDGs before consuming any
        uint32_t entry[N_STREAMS];
        #pragma unroll
        for (int q = 0; q < N_STREAMS; q++)
            if (have[q])
                entry[q] = __ldg(&sfc_global[ctx[q].x & (M_SIZE-1)]);

        // Phase B: consume + compose per stream (separate loop for ILP)
        #pragma unroll
        for (int q = 0; q < N_STREAMS; q++)
            if (have[q])
                DECODE_AND_WRITE(ctx[q], entry[q], tid_s[q], i, output);
    }
}

__global__ void rans_decode_dump_kernel(
    const uint32_t* __restrict__ compressed_u32,
    const int32_t*  __restrict__ block_offsets,
    const uint32_t* __restrict__ final_states,
    const uint32_t* __restrict__ sfc_global,
    const uint8_t*  __restrict__ sign_mantissa,
    uint8_t*        __restrict__ digest_out,
    int64_t n_fp8_per_stream,
    int64_t n_streams)
{
    int tid = threadIdx.x;
    RansCtx ctx[N_STREAMS];
    int32_t tid_s[N_STREAMS];
    bool have[N_STREAMS];

    #pragma unroll
    for (int q = 0; q < N_STREAMS; q++) {
        int enc = N_STREAMS * blockIdx.x + q;
        tid_s[q] = (int64_t)enc * blockDim.x + tid;
        have[q] = tid_s[q] < n_streams;
        if (have[q]) {
            int32_t base = block_offsets[enc], next = block_offsets[enc+1];
            int G = (next-base)/((int)blockDim.x*4);
            ctx[q].x = final_states[tid_s[q]]; ctx[q].slab_idx = G;
            ctx[q].buf_hi=0; ctx[q].buf_lo=0; ctx[q].buf_avail=0;
            ctx[q].block_base_u32 = base/4;
        }
    }
    bool any = false;
    #pragma unroll
    for (int q = 0; q < N_STREAMS; q++) any |= have[q];
    if (!any) return;

    uint8_t digest = 0;
    int64_t n_pairs = n_fp8_per_stream / 2;
    for (int64_t i = 0; i < n_pairs; i++) {
        // ── Stage 1: issue all 3 LDGs ──
        uint32_t e0=0, e1=0, e2=0;
        if (have[0]) e0 = __ldg(&sfc_global[ctx[0].x & (M_SIZE-1)]);
        if (have[1]) e1 = __ldg(&sfc_global[ctx[1].x & (M_SIZE-1)]);
        if (have[2]) e2 = __ldg(&sfc_global[ctx[2].x & (M_SIZE-1)]);

        // ── Stage 2: unpack all 3 (f, c extracted; sym overwrites entry) ──
        uint32_t f0=0,c0=0, f1=0,c1=0, f2=0,c2=0;
        if (have[0]) { f0=(e0>>8)&0xFFF; c0=(e0>>20)&0xFFF; e0=e0&0xFF; }
        if (have[1]) { f1=(e1>>8)&0xFFF; c1=(e1>>20)&0xFFF; e1=e1&0xFF; }
        if (have[2]) { f2=(e2>>8)&0xFFF; c2=(e2>>20)&0xFFF; e2=e2&0xFF; }

        // ── Stage 3: state update (slot recomputed inline from x) ──
        if (have[0]) ctx[0].x = (ctx[0].x >> M_LOG) * f0 + ((ctx[0].x & (M_SIZE-1)) - c0);
        if (have[1]) ctx[1].x = (ctx[1].x >> M_LOG) * f1 + ((ctx[1].x & (M_SIZE-1)) - c1);
        if (have[2]) ctx[2].x = (ctx[2].x >> M_LOG) * f2 + ((ctx[2].x & (M_SIZE-1)) - c2);

        // ── Stage 4: branchless renorm all 3 (interleaved) ──
        #pragma unroll
        for (int q = 0; q < 3; q++) {
            if (have[q]) {
                bool need = (ctx[q].x < L_LOW);
                if (need && ctx[q].buf_avail == 0) {
                    ctx[q].slab_idx -= 2;
                    int32_t off = ctx[q].block_base_u32
                        + ctx[q].slab_idx*(int)blockDim.x + (int)threadIdx.x;
                    ctx[q].buf_lo = compressed_u32[off];
                    ctx[q].buf_hi = compressed_u32[off+(int)blockDim.x];
                    ctx[q].buf_avail = 4;
                }
                uint32_t xr=(ctx[q].x<<16)|(ctx[q].buf_hi>>16);
                uint32_t bh=(ctx[q].buf_hi<<16)|(ctx[q].buf_lo>>16);
                uint32_t bl=ctx[q].buf_lo<<16; int av=ctx[q].buf_avail-1;
                ctx[q].x=need?xr:ctx[q].x; ctx[q].buf_hi=need?bh:ctx[q].buf_hi;
                ctx[q].buf_lo=need?bl:ctx[q].buf_lo; ctx[q].buf_avail=need?av:ctx[q].buf_avail;
            }
        }

        // ── Stage 5: compose fp8 + digest (e0/e1/e2 hold sym) ──
        if (have[0]) {
            uint8_t sm=sign_mantissa[i*n_streams+tid_s[0]];
            uint8_t s0=sm&0xF,s1=sm>>4;
            digest ^= ((s0&8)<<4)|((e0>>4)<<3)|(s0&7);
            digest ^= ((s1&8)<<4)|((e0&0xF)<<3)|(s1&7);
        }
        if (have[1]) {
            uint8_t sm=sign_mantissa[i*n_streams+tid_s[1]];
            uint8_t s0=sm&0xF,s1=sm>>4;
            digest ^= ((s0&8)<<4)|((e1>>4)<<3)|(s0&7);
            digest ^= ((s1&8)<<4)|((e1&0xF)<<3)|(s1&7);
        }
        if (have[2]) {
            uint8_t sm=sign_mantissa[i*n_streams+tid_s[2]];
            uint8_t s0=sm&0xF,s1=sm>>4;
            digest ^= ((s0&8)<<4)|((e2>>4)<<3)|(s0&7);
            digest ^= ((s1&8)<<4)|((e2&0xF)<<3)|(s1&7);
        }
    }
    digest_out[(int64_t)blockIdx.x * blockDim.x + tid] = digest;
}

#undef CONSUME

// ─── Host launchers ──────────────────────────────────────────────────

torch::Tensor gpu_rans_decode(
    torch::Tensor compressed, torch::Tensor block_offsets,
    torch::Tensor final_states, torch::Tensor sign_mantissa,
    int64_t n_fp8_per_stream, torch::Tensor freqs)
{
    int64_t n_streams, n_enc_blocks;
    auto sfc_gpu = build_sfc_table(compressed, block_offsets, final_states,
                                   sign_mantissa, n_fp8_per_stream, freqs,
                                   n_streams, n_enc_blocks);
    auto output = torch::empty(
        {n_fp8_per_stream, n_streams},
        torch::TensorOptions().dtype(torch::kUInt8).device(compressed.device()));
    int64_t blocks = (n_enc_blocks + N_STREAMS - 1) / N_STREAMS;
    rans_decode_kernel<<<blocks, BLOCK_STREAMS>>>(
        reinterpret_cast<const uint32_t*>(compressed.data_ptr<uint8_t>()),
        block_offsets.data_ptr<int32_t>(),
        reinterpret_cast<const uint32_t*>(final_states.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(sfc_gpu.data_ptr<int32_t>()),
        sign_mantissa.data_ptr<uint8_t>(),
        output.data_ptr<uint8_t>(),
        n_fp8_per_stream, n_streams);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor gpu_rans_decode_dump(
    torch::Tensor compressed, torch::Tensor block_offsets,
    torch::Tensor final_states, torch::Tensor sign_mantissa,
    int64_t n_fp8_per_stream, torch::Tensor freqs)
{
    int64_t n_streams, n_enc_blocks;
    auto sfc_gpu = build_sfc_table(compressed, block_offsets, final_states,
                                   sign_mantissa, n_fp8_per_stream, freqs,
                                   n_streams, n_enc_blocks);
    int64_t blocks = (n_enc_blocks + N_STREAMS - 1) / N_STREAMS;
    auto digest = torch::empty(
        {blocks * BLOCK_STREAMS},
        torch::TensorOptions().dtype(torch::kUInt8).device(compressed.device()));
    rans_decode_dump_kernel<<<blocks, BLOCK_STREAMS>>>(
        reinterpret_cast<const uint32_t*>(compressed.data_ptr<uint8_t>()),
        block_offsets.data_ptr<int32_t>(),
        reinterpret_cast<const uint32_t*>(final_states.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(sfc_gpu.data_ptr<int32_t>()),
        sign_mantissa.data_ptr<uint8_t>(),
        digest.data_ptr<uint8_t>(),
        n_fp8_per_stream, n_streams);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return digest;
}

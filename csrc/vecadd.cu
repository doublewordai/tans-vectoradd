// Fused tANS decompress + FP8 vector add.
//
// Two kernels live here:
//   - fp8_vecadd_raw: uncompressed FP8 baseline.
//   - fp8_vecadd_fused_tans: tANS-decode-and-add fused into one kernel.
//
// The fused kernel decodes two compressed FP8 tensors, adds the decoded
// values as E4M3, and writes the tiled output.  The tANS stream stores FP8
// exponent pairs entropy-coded and keeps sign+mantissa nibbles uncompressed.

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <stdint.h>

namespace {

constexpr int kTansL = 4096;
constexpr int kBlockStreams = 128;
constexpr int kDefaultStreamsPerBlock = 8;
constexpr int kTilePairs = 8;

__device__ __forceinline__ uint8_t add_fp8_bytes(uint8_t a, uint8_t b) {
    __nv_fp8_e4m3 fa, fb;
    memcpy(&fa, &a, 1);
    memcpy(&fb, &b, 1);
    __nv_fp8_e4m3 fc(float(fa) + float(fb));
    uint8_t out;
    memcpy(&out, &fc, 1);
    return out;
}

__device__ __forceinline__ uint8_t compose_fp8_byte(uint8_t exp, uint8_t sm) {
    return (uint8_t)(((sm & 8u) << 4) | ((exp & 0xFu) << 3) | (sm & 7u));
}

__global__ void fp8_vecadd_raw_kernel(
    const uint4* __restrict__ A,
    const uint4* __restrict__ B,
    uint4*       __restrict__ C,
    int64_t n_vec)
{
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = tid; i < n_vec; i += stride) {
        uint4 va = A[i];
        uint4 vb = B[i];
        uint4 vc;

        #pragma unroll
        for (int w = 0; w < 4; w++) {
            uint32_t wa = (&va.x)[w];
            uint32_t wb = (&vb.x)[w];
            uint32_t wc = 0;

            #pragma unroll
            for (int b = 0; b < 4; b++) {
                uint8_t ba = (uint8_t)((wa >> (b * 8)) & 0xFFu);
                uint8_t bb = (uint8_t)((wb >> (b * 8)) & 0xFFu);
                wc |= (uint32_t)add_fp8_bytes(ba, bb) << (b * 8);
            }

            (&vc.x)[w] = wc;
        }

        C[i] = vc;
    }
}

struct TansCtx {
    uint32_t x;
    uint32_t buf_hi;       // Current slab; MSB end is consumed next.
    uint32_t buf_lo;       // Reserve slab promoted to buf_hi on advance.
    uint32_t cnt;          // Bits already consumed from buf_hi, in [0, 31].
    int slab_idx;          // Next slab to load into buf_lo on advance, or -1.
    int32_t block_base_u32;
};

__device__ __forceinline__ void init_tans_ctx(
    TansCtx& ctx,
    const int32_t* __restrict__ offsets,
    const uint16_t* __restrict__ states,
    const uint8_t* __restrict__ partial_cnts,
    const uint32_t* __restrict__ compressed,
    int enc,
    int sid,
    int tid)
{
    int32_t base = offsets[enc];
    int32_t next = offsets[enc + 1];
    int slabs_per_stream = (next - base) / (kBlockStreams * 4);

    ctx.x = states[sid];
    ctx.block_base_u32 = base / 4;

    int top = slabs_per_stream - 1;
    int next_top = slabs_per_stream - 2;
    int extra_top = slabs_per_stream - 3;

    ctx.buf_hi = __ldg(&compressed[ctx.block_base_u32 + top * kBlockStreams + tid]);
    ctx.buf_lo = next_top >= 0
        ? __ldg(&compressed[ctx.block_base_u32 + next_top * kBlockStreams + tid])
        : 0u;
    ctx.slab_idx = extra_top;

    uint32_t partial = partial_cnts[sid];
    ctx.cnt = partial == 0u ? 0u : 32u - partial;
}

template <bool kPrefetch>
__device__ __forceinline__ uint8_t tans_step(
    TansCtx& ctx,
    const uint32_t* __restrict__ compressed,
    int tid,
    const uint32_t* __restrict__ decode_tbl)
{
    uint32_t entry = decode_tbl[ctx.x - kTansL];
    uint32_t nb = (entry >> 8) & 0xFu;
    uint32_t base_state = (entry >> 16) & 0xFFFFu;
    uint8_t sym = (uint8_t)(entry & 0xFFu);

    uint32_t shifted = __funnelshift_l(ctx.buf_lo, ctx.buf_hi, ctx.cnt);
    uint32_t bits = nb == 0u ? 0u : shifted >> (32u - nb);
    ctx.x = base_state | bits;

    uint32_t cnt_new = ctx.cnt + nb;
    bool advance = cnt_new >= 32u;
    uint32_t new_xtra = 0u;
    if (advance && ctx.slab_idx >= 0) {
        new_xtra = __ldg(&compressed[
            ctx.block_base_u32 + ctx.slab_idx * kBlockStreams + tid]);
    }

    if constexpr (kPrefetch) {
        int prefetch_idx = ctx.slab_idx - 1;
        if (prefetch_idx >= 0) {
            asm volatile("prefetch.global.L2 [%0];" :: "l"(
                &compressed[ctx.block_base_u32
                    + prefetch_idx * kBlockStreams + tid]));
        }
    }

    ctx.buf_hi = advance ? ctx.buf_lo : ctx.buf_hi;
    ctx.buf_lo = advance ? new_xtra : ctx.buf_lo;
    ctx.cnt = advance ? cnt_new - 32u : cnt_new;
    ctx.slab_idx = advance ? ctx.slab_idx - 1 : ctx.slab_idx;

    return sym;
}

template <int kStreamsPerBlock>
__global__ void fp8_vecadd_fused_tans_kernel(
    const uint32_t* __restrict__ a_compressed,
    const int32_t*  __restrict__ a_offsets,
    const uint16_t* __restrict__ a_states,
    const uint8_t*  __restrict__ a_partial,
    const uint8_t*  __restrict__ a_sm_tiled,
    const uint32_t* __restrict__ b_compressed,
    const int32_t*  __restrict__ b_offsets,
    const uint16_t* __restrict__ b_states,
    const uint8_t*  __restrict__ b_partial,
    const uint8_t*  __restrict__ b_sm_tiled,
    const uint32_t* __restrict__ decode_tbl,
    uint8_t*        __restrict__ output_tiled,
    int64_t n_fp8_per_stream,
    int64_t n_streams)
{
    extern __shared__ uint8_t smem_raw[];
    uint32_t* tbl_smem = reinterpret_cast<uint32_t*>(smem_raw);
    uint8_t* decoded_sym_a = smem_raw + kTansL * sizeof(uint32_t);
    uint8_t* decoded_sym_b = decoded_sym_a + kTilePairs * blockDim.x;

    for (int i = threadIdx.x; i < kTansL; i += blockDim.x) {
        tbl_smem[i] = decode_tbl[i];
    }
    __syncthreads();

    int enc_group = threadIdx.x / kBlockStreams;
    int tid = threadIdx.x % kBlockStreams;
    int enc = kStreamsPerBlock * blockIdx.x + enc_group;
    int sid = enc * kBlockStreams + tid;
    bool active = sid < n_streams;

    TansCtx ctx_a, ctx_b;
    if (active) {
        init_tans_ctx(ctx_a, a_offsets, a_states, a_partial, a_compressed,
                      enc, sid, tid);
        init_tans_ctx(ctx_b, b_offsets, b_states, b_partial, b_compressed,
                      enc, sid, tid);
    }

    int64_t n_tiles = (n_fp8_per_stream / 2) / kTilePairs;

    for (int64_t tile_idx = 0; tile_idx < n_tiles; tile_idx++) {
        uint64_t sm_a;
        uint64_t sm_b;

        if (active) {
            int64_t sm_base = (tile_idx * n_streams + sid) * kTilePairs;
            sm_a = *reinterpret_cast<const uint64_t*>(a_sm_tiled + sm_base);
            sm_b = *reinterpret_cast<const uint64_t*>(b_sm_tiled + sm_base);

            #pragma unroll
            for (int t = 0; t < kTilePairs; t++) {
                decoded_sym_a[t * blockDim.x + threadIdx.x] =
                    tans_step<true>(ctx_a, a_compressed, tid, tbl_smem);
                decoded_sym_b[t * blockDim.x + threadIdx.x] =
                    tans_step<true>(ctx_b, b_compressed, tid, tbl_smem);
            }
        }
        __syncthreads();

        if (active) {
            uint8_t out_buf[kTilePairs * 2];

            #pragma unroll
            for (int t = 0; t < kTilePairs; t++) {
                uint8_t sym_a = decoded_sym_a[t * blockDim.x + threadIdx.x];
                uint8_t sym_b = decoded_sym_b[t * blockDim.x + threadIdx.x];
                uint8_t sm_a_val = (uint8_t)(sm_a >> (t * 8));
                uint8_t sm_b_val = (uint8_t)(sm_b >> (t * 8));

                uint8_t a0 = compose_fp8_byte(sym_a >> 4, sm_a_val & 0xFu);
                uint8_t a1 = compose_fp8_byte(sym_a & 0xFu, sm_a_val >> 4);
                uint8_t b0 = compose_fp8_byte(sym_b >> 4, sm_b_val & 0xFu);
                uint8_t b1 = compose_fp8_byte(sym_b & 0xFu, sm_b_val >> 4);

                out_buf[t * 2] = add_fp8_bytes(a0, b0);
                out_buf[t * 2 + 1] = add_fp8_bytes(a1, b1);
            }

            int64_t out_base = (tile_idx * n_streams + sid) * (kTilePairs * 2);
            *reinterpret_cast<uint4*>(output_tiled + out_base) =
                *reinterpret_cast<const uint4*>(out_buf);
        }
        __syncthreads();
    }
}

int streams_per_block_from_env() {
    const char* value = std::getenv("TANS_STREAMS_PER_BLOCK");
    if (value == nullptr || value[0] == '\0') {
        return kDefaultStreamsPerBlock;
    }

    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    TORCH_CHECK(end != value && *end == '\0',
                "TANS_STREAMS_PER_BLOCK must be an integer");
    TORCH_CHECK(parsed == 1 || parsed == 2 || parsed == 4 || parsed == 8,
                "TANS_STREAMS_PER_BLOCK must be one of 1, 2, 4, or 8");
    return static_cast<int>(parsed);
}

void check_u8_cuda_contiguous(const char* name, const torch::Tensor& t) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.dtype() == torch::kUInt8, name, " must be uint8");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor fp8_vecadd_raw(torch::Tensor A, torch::Tensor B) {
    check_u8_cuda_contiguous("A", A);
    check_u8_cuda_contiguous("B", B);
    TORCH_CHECK(A.numel() == B.numel(), "A and B must have the same length");
    TORCH_CHECK(A.numel() % 16 == 0, "A/B length must be a multiple of 16");

    int64_t n_vec = A.numel() / 16;
    auto C = torch::empty_like(A);

    int threads = 256;
    int blocks = std::min<int64_t>((n_vec + threads - 1) / threads, 512);
    blocks = std::max(blocks, 1);

    fp8_vecadd_raw_kernel<<<blocks, threads>>>(
        reinterpret_cast<const uint4*>(A.data_ptr<uint8_t>()),
        reinterpret_cast<const uint4*>(B.data_ptr<uint8_t>()),
        reinterpret_cast<uint4*>(C.data_ptr<uint8_t>()),
        n_vec);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}

torch::Tensor fp8_vecadd_fused_tans(
    torch::Tensor a_comp, torch::Tensor a_offsets, torch::Tensor a_states,
    torch::Tensor a_partial, torch::Tensor a_sm,
    torch::Tensor b_comp, torch::Tensor b_offsets, torch::Tensor b_states,
    torch::Tensor b_partial, torch::Tensor b_sm,
    torch::Tensor decode_tbl,
    int64_t n_fp8_per_stream)
{
    check_u8_cuda_contiguous("a_comp", a_comp);
    check_u8_cuda_contiguous("a_partial", a_partial);
    check_u8_cuda_contiguous("a_sm", a_sm);
    check_u8_cuda_contiguous("b_comp", b_comp);
    check_u8_cuda_contiguous("b_partial", b_partial);
    check_u8_cuda_contiguous("b_sm", b_sm);

    TORCH_CHECK(a_offsets.is_cuda() && a_offsets.dtype() == torch::kInt32
                && a_offsets.is_contiguous(), "a_offsets must be contiguous CUDA int32");
    TORCH_CHECK(b_offsets.is_cuda() && b_offsets.dtype() == torch::kInt32
                && b_offsets.is_contiguous(), "b_offsets must be contiguous CUDA int32");
    TORCH_CHECK(a_states.is_cuda() && a_states.dtype() == torch::kUInt16
                && a_states.is_contiguous(), "a_states must be contiguous CUDA uint16");
    TORCH_CHECK(b_states.is_cuda() && b_states.dtype() == torch::kUInt16
                && b_states.is_contiguous(), "b_states must be contiguous CUDA uint16");
    TORCH_CHECK(decode_tbl.is_cuda() && decode_tbl.dtype() == torch::kInt32
                && decode_tbl.is_contiguous(), "decode_tbl must be contiguous CUDA int32");

    TORCH_CHECK(a_states.numel() == b_states.numel(),
                "A/B stream counts must match");
    TORCH_CHECK(a_partial.numel() == a_states.numel()
                && b_partial.numel() == b_states.numel(),
                "partial-count tensors must match state tensors");
    TORCH_CHECK(a_comp.numel() % 4 == 0 && b_comp.numel() % 4 == 0,
                "compressed tensors must have byte lengths divisible by 4");
    TORCH_CHECK(decode_tbl.numel() == kTansL,
                "decode_tbl must have ", kTansL, " entries");
    TORCH_CHECK(n_fp8_per_stream % (2 * kTilePairs) == 0,
                "n_fp8_per_stream must be divisible by ", 2 * kTilePairs);

    int64_t n_streams = a_states.numel();
    int64_t n_enc_blocks = (n_streams + kBlockStreams - 1) / kBlockStreams;
    TORCH_CHECK(a_offsets.numel() == n_enc_blocks + 1
                && b_offsets.numel() == n_enc_blocks + 1,
                "offset tensors must have n_encoder_blocks + 1 entries");

    int64_t n_tiles = (n_fp8_per_stream / 2) / kTilePairs;
    int64_t expected_sm = n_tiles * n_streams * kTilePairs;
    TORCH_CHECK(a_sm.numel() == expected_sm && b_sm.numel() == expected_sm,
                "sign/mantissa tensors have the wrong tiled size");

    auto output = torch::empty(
        {n_tiles * n_streams * kTilePairs * 2},
        torch::TensorOptions().dtype(torch::kUInt8).device(a_comp.device()));

    int streams_per_block = streams_per_block_from_env();

#define LAUNCH_FUSED_TANS(SPB)                                                \
    do {                                                                       \
        int64_t blocks = (n_enc_blocks + (SPB) - 1) / (SPB);                   \
        int threads = kBlockStreams * (SPB);                                   \
        C10_CUDA_CHECK(cudaFuncSetCacheConfig(                                 \
            fp8_vecadd_fused_tans_kernel<SPB>,                                 \
            cudaFuncCachePreferL1));                                           \
        int smem_bytes = kTansL * 4 + 2 * kTilePairs * threads;                \
        fp8_vecadd_fused_tans_kernel<SPB><<<                                   \
            blocks, threads, smem_bytes>>>(                                    \
            reinterpret_cast<const uint32_t*>(a_comp.data_ptr<uint8_t>()),     \
            a_offsets.data_ptr<int32_t>(),                                     \
            a_states.data_ptr<uint16_t>(),                                     \
            a_partial.data_ptr<uint8_t>(),                                     \
            a_sm.data_ptr<uint8_t>(),                                          \
            reinterpret_cast<const uint32_t*>(b_comp.data_ptr<uint8_t>()),     \
            b_offsets.data_ptr<int32_t>(),                                     \
            b_states.data_ptr<uint16_t>(),                                     \
            b_partial.data_ptr<uint8_t>(),                                     \
            b_sm.data_ptr<uint8_t>(),                                          \
            reinterpret_cast<const uint32_t*>(                                 \
                decode_tbl.data_ptr<int32_t>()),                               \
            output.data_ptr<uint8_t>(),                                        \
            n_fp8_per_stream,                                                  \
            n_streams);                                                        \
    } while (0)

    switch (streams_per_block) {
        case 1:
            LAUNCH_FUSED_TANS(1);
            break;
        case 2:
            LAUNCH_FUSED_TANS(2);
            break;
        case 4:
            LAUNCH_FUSED_TANS(4);
            break;
        case 8:
            LAUNCH_FUSED_TANS(8);
            break;
        default:
            TORCH_CHECK(false, "unreachable TANS_STREAMS_PER_BLOCK value");
    }

#undef LAUNCH_FUSED_TANS

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

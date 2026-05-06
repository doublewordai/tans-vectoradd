// Standalone GPU tANS decoder for FP8 E4M3 pair streams.

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

namespace {

constexpr int kTansL = 4096;
constexpr int kBlockStreams = 128;

struct TansCtx {
    uint32_t x;
    uint32_t buf_hi;
    uint32_t buf_lo;
    uint32_t cnt;
    int slab_idx;
    int32_t block_base_u32;
};

torch::Tensor build_decode_table(torch::Tensor spread, torch::Tensor freqs) {
    TORCH_CHECK(spread.dtype() == torch::kInt32);
    TORCH_CHECK(spread.numel() == kTansL);
    TORCH_CHECK(freqs.dtype() == torch::kInt32);

    auto spread_cpu = spread.cpu().contiguous();
    auto freqs_cpu = freqs.cpu().contiguous();
    const int32_t* sp = spread_cpu.data_ptr<int32_t>();
    const int32_t* fr = freqs_cpu.data_ptr<int32_t>();
    int n_alphabet = (int)freqs.numel();

    auto table_cpu = torch::zeros({kTansL}, torch::kInt32);
    uint32_t* table = reinterpret_cast<uint32_t*>(table_cpu.data_ptr<int32_t>());
    std::vector<int32_t> counts(n_alphabet, 0);

    for (int slot = 0; slot < kTansL; slot++) {
        int sym = sp[slot];
        TORCH_CHECK(sym >= 0 && sym < n_alphabet);
        int rank = counts[sym]++;
        int freq = fr[sym];
        int prev = freq + rank;

        int nb = 0;
        int base_state = prev;
        while (base_state < kTansL) {
            base_state <<= 1;
            nb++;
        }
        TORCH_CHECK(base_state < 2 * kTansL);

        table[slot] = ((uint32_t)sym & 0xFFu)
                    | (((uint32_t)nb & 0xFu) << 8)
                    | (((uint32_t)base_state & 0xFFFFu) << 16);
    }

    return table_cpu.to(spread.device());
}

__device__ __forceinline__ uint8_t compose_fp8_byte(uint8_t exp, uint8_t sm) {
    return (uint8_t)(((sm & 8u) << 4) | ((exp & 0xFu) << 3) | (sm & 7u));
}

__device__ __forceinline__ void init_tans_ctx(
    TansCtx& ctx,
    const uint32_t* __restrict__ compressed,
    const int32_t* __restrict__ offsets,
    const uint16_t* __restrict__ states,
    const uint8_t* __restrict__ partial_cnts,
    int enc_block,
    int sid,
    int tid)
{
    int32_t base = offsets[enc_block];
    int32_t next = offsets[enc_block + 1];
    int slabs_per_stream = (next - base) / (kBlockStreams * 4);

    ctx.x = states[sid];
    ctx.block_base_u32 = base / 4;

    int top = slabs_per_stream - 1;
    int next_top = slabs_per_stream - 2;
    ctx.buf_hi = compressed[ctx.block_base_u32 + top * kBlockStreams + tid];
    ctx.buf_lo = next_top >= 0
        ? compressed[ctx.block_base_u32 + next_top * kBlockStreams + tid]
        : 0u;
    ctx.slab_idx = next_top - 1;

    uint32_t partial = partial_cnts[sid];
    ctx.cnt = partial == 0u ? 0u : 32u - partial;
}

__device__ __forceinline__ uint8_t tans_step(
    TansCtx& ctx,
    const uint32_t* __restrict__ compressed,
    int tid,
    const uint32_t* __restrict__ decode_table)
{
    uint32_t entry = __ldg(&decode_table[ctx.x - kTansL]);
    uint32_t nb = (entry >> 8) & 0xFu;
    uint32_t base_state = (entry >> 16) & 0xFFFFu;
    uint8_t sym = (uint8_t)(entry & 0xFFu);

    uint32_t shifted = __funnelshift_l(ctx.buf_lo, ctx.buf_hi, ctx.cnt);
    uint32_t bits = nb == 0u ? 0u : shifted >> (32u - nb);
    ctx.x = base_state | bits;

    ctx.cnt += nb;
    if (ctx.cnt >= 32u) {
        ctx.cnt -= 32u;
        ctx.buf_hi = ctx.buf_lo;
        if (ctx.slab_idx >= 0) {
            ctx.buf_lo = compressed[
                ctx.block_base_u32 + ctx.slab_idx * kBlockStreams + tid];
            ctx.slab_idx -= 1;
        } else {
            ctx.buf_lo = 0u;
        }
    }

    return sym;
}

__global__ void tans_decode_kernel(
    const uint32_t* __restrict__ compressed,
    const int32_t* __restrict__ offsets,
    const uint16_t* __restrict__ states,
    const uint8_t* __restrict__ partial_cnts,
    const uint32_t* __restrict__ decode_table,
    const uint8_t* __restrict__ sign_mantissa,
    uint8_t* __restrict__ output,
    int64_t n_fp8_per_stream,
    int64_t n_streams)
{
    int tid = threadIdx.x;
    int enc_block = blockIdx.x;
    int sid = enc_block * kBlockStreams + tid;
    if (sid >= n_streams) return;

    TansCtx ctx;
    init_tans_ctx(ctx, compressed, offsets, states, partial_cnts,
                  enc_block, sid, tid);

    int64_t n_pairs = n_fp8_per_stream / 2;
    for (int64_t i = 0; i < n_pairs; i++) {
        uint8_t sym = tans_step(ctx, compressed, tid, decode_table);
        uint8_t sm = sign_mantissa[i * n_streams + sid];

        output[(2 * i) * n_streams + sid] =
            compose_fp8_byte(sym >> 4, sm & 0xFu);
        output[(2 * i + 1) * n_streams + sid] =
            compose_fp8_byte(sym & 0xFu, sm >> 4);
    }
}

}  // namespace

torch::Tensor gpu_tans_decode(
    torch::Tensor compressed,
    torch::Tensor offsets,
    torch::Tensor states,
    torch::Tensor partial_cnts,
    torch::Tensor sign_mantissa,
    torch::Tensor spread,
    torch::Tensor freqs,
    int64_t n_fp8_per_stream)
{
    TORCH_CHECK(compressed.is_cuda() && compressed.dtype() == torch::kUInt8
                && compressed.is_contiguous(), "compressed must be contiguous CUDA uint8");
    TORCH_CHECK(offsets.is_cuda() && offsets.dtype() == torch::kInt32
                && offsets.is_contiguous(), "offsets must be contiguous CUDA int32");
    TORCH_CHECK(states.is_cuda() && states.dtype() == torch::kUInt16
                && states.is_contiguous(), "states must be contiguous CUDA uint16");
    TORCH_CHECK(partial_cnts.is_cuda() && partial_cnts.dtype() == torch::kUInt8
                && partial_cnts.is_contiguous(), "partial_cnts must be contiguous CUDA uint8");
    TORCH_CHECK(sign_mantissa.is_cuda() && sign_mantissa.dtype() == torch::kUInt8
                && sign_mantissa.is_contiguous(), "sign_mantissa must be contiguous CUDA uint8");
    TORCH_CHECK(n_fp8_per_stream % 2 == 0,
                "n_fp8_per_stream must be even");
    TORCH_CHECK(compressed.numel() % 4 == 0,
                "compressed length must be divisible by 4 bytes");

    int64_t n_streams = states.numel();
    int64_t n_blocks = (n_streams + kBlockStreams - 1) / kBlockStreams;
    TORCH_CHECK(offsets.numel() == n_blocks + 1,
                "offsets must have n_blocks + 1 entries");
    TORCH_CHECK(partial_cnts.numel() == n_streams,
                "partial_cnts must match states");
    TORCH_CHECK(sign_mantissa.numel() == (n_fp8_per_stream / 2) * n_streams,
                "sign_mantissa has the wrong shape");

    auto decode_table = build_decode_table(spread, freqs);
    auto output = torch::empty(
        {n_fp8_per_stream, n_streams},
        torch::TensorOptions().dtype(torch::kUInt8).device(compressed.device()));

    tans_decode_kernel<<<n_blocks, kBlockStreams>>>(
        reinterpret_cast<const uint32_t*>(compressed.data_ptr<uint8_t>()),
        offsets.data_ptr<int32_t>(),
        states.data_ptr<uint16_t>(),
        partial_cnts.data_ptr<uint8_t>(),
        reinterpret_cast<const uint32_t*>(decode_table.data_ptr<int32_t>()),
        sign_mantissa.data_ptr<uint8_t>(),
        output.data_ptr<uint8_t>(),
        n_fp8_per_stream,
        n_streams);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

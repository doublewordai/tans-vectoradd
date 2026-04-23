// GPU rANS decoder for FP8 E4M3 — 16-symbol exponent alphabet.
//
// Input layout (matches rans_encode_interleaved):
//   Block b owns streams [b*128, b*128+128).
//   Block b's compressed data starts at byte offset block_offsets[b].
//   Within a block there are G_b "slabs", each 128 * 4 = 512 bytes.
//   Slab s holds, for each stream tid in the block, 4 bytes of that
//   stream's compressed data at position 4*tid within the slab.
//   Each stream is right-aligned: a stream with G_s slabs of real data
//   occupies slabs [G_b - G_s, G_b - 1]. Leading slabs are zero-padded.
//
// Decode: each iteration produces one exponent nibble (sym in [0,16))
// and recomposes an FP8 byte with the corresponding sign+mantissa
// nibble from the packed sm tensor.
//
// Stream radix b = 2^16: renorm pulls 16 bits per step, state lives in
// [L, bL) = [2^16, 2^32), which fills the uint32.

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace rans_gpu {
constexpr int      M_LOG         = 11;
constexpr uint32_t M_SIZE        = 1u << M_LOG;   // 2048
constexpr uint32_t L_LOW         = 1u << 16;      // 65536
constexpr int      MAX_ALPHA     = 16;            // exponent nibble alphabet
constexpr int      BLOCK_STREAMS = 128;           // must match blockDim.x
}

struct RansCtx {
    uint32_t x;
    int      slab_idx;
    uint32_t buf_hi;        // newer uint16 chunks (MSB consumed first)
    uint32_t buf_lo;        // older uint16 chunks
    int      buf_avail;     // 0..4 uint16 chunks
    int64_t  block_base_u32;
};

// One rANS decode step: read the combined slot→(sym, f, c) entry from
// SMEM, advance state, pull renorm uint16 chunks as needed.
//
// sfc_smem entry packing (uint32 per slot):
//   bits  0..7  : symbol (0..15 for this alphabet)
//   bits  8..19 : frequency f (12 bits)
//   bits 20..31 : cumulative offset c (12 bits)
//
// Renorm refill: when buf is empty, load TWO uint32s (= 4 uint16 chunks)
// from consecutive slabs. Halves refill frequency vs 1-slab-at-a-time,
// which reduces per-lane divergence for global-memory coalescing.
__device__ __forceinline__ uint8_t decode_step(
    RansCtx& ctx,
    const uint32_t* sfc_smem,
    const uint32_t* __restrict__ compressed_u32,
    int tid_in_block)
{
    uint32_t slot  = ctx.x & (rans_gpu::M_SIZE - 1);
    uint32_t entry = sfc_smem[slot];
    uint8_t  sym   = (uint8_t)(entry & 0xFF);
    uint32_t f     = (entry >> 8) & 0xFFF;
    uint32_t c     = (entry >> 20) & 0xFFF;
    ctx.x = (ctx.x >> rans_gpu::M_LOG) * f + (slot - c);

    while (ctx.x < rans_gpu::L_LOW) {
        if (ctx.buf_avail == 0) {
            ctx.slab_idx -= 2;
            int64_t off = ctx.block_base_u32
                + (int64_t)ctx.slab_idx * blockDim.x + tid_in_block;
            ctx.buf_lo = compressed_u32[off];              // older slab
            ctx.buf_hi = compressed_u32[off + blockDim.x]; // newer slab
            ctx.buf_avail = 4;
        }
        ctx.x = (ctx.x << 16) | (ctx.buf_hi >> 16);
        ctx.buf_hi = (ctx.buf_hi << 16) | (ctx.buf_lo >> 16);
        ctx.buf_lo <<= 16;
        ctx.buf_avail--;
    }
    return sym;
}

// Dual-stream kernel: each kernel block covers TWO encoder blocks (256
// streams). Each thread decodes 2 independent streams; the compiler
// inlines decode_step and interleaves SMEM loads across streams for ILP.
__global__ void rans_decode_fp8_kernel(
    const uint32_t* __restrict__ compressed_u32,
    const int32_t*  __restrict__ block_offsets,
    const uint32_t* __restrict__ final_states,
    const uint32_t* __restrict__ sfc_global,       // [M_SIZE] packed (sym,f,c)
    const uint8_t*  __restrict__ sign_mantissa,    // [n_fp8/2, n_streams]
    uint8_t*        __restrict__ output,           // [n_fp8, n_streams]
    int64_t n_fp8_per_stream,
    int64_t n_streams)
{
    __shared__ uint32_t sfc_smem[rans_gpu::M_SIZE];

    for (int i = threadIdx.x; i < (int)rans_gpu::M_SIZE; i += blockDim.x) {
        sfc_smem[i] = sfc_global[i];
    }
    __syncthreads();

    int tid_in_block = threadIdx.x;
    int enc_block_a  = 2 * blockIdx.x;
    int enc_block_b  = 2 * blockIdx.x + 1;
    int64_t tid_a    = (int64_t)enc_block_a * blockDim.x + tid_in_block;
    int64_t tid_b    = (int64_t)enc_block_b * blockDim.x + tid_in_block;

    bool have_a = tid_a < n_streams;
    bool have_b = tid_b < n_streams;
    if (!have_a && !have_b) return;

    RansCtx A, B;
    if (have_a) {
        int32_t base_bytes_a = block_offsets[enc_block_a];
        int32_t next_bytes_a = block_offsets[enc_block_a + 1];
        int     G_a          = (next_bytes_a - base_bytes_a)
                               / ((int)blockDim.x * 4);
        A.x              = final_states[tid_a];
        A.slab_idx       = G_a;
        A.buf_hi         = 0;
        A.buf_lo         = 0;
        A.buf_avail      = 0;
        A.block_base_u32 = (int64_t)base_bytes_a / 4;
    }
    if (have_b) {
        int32_t base_bytes_b = block_offsets[enc_block_b];
        int32_t next_bytes_b = block_offsets[enc_block_b + 1];
        int     G_b          = (next_bytes_b - base_bytes_b)
                               / ((int)blockDim.x * 4);
        B.x              = final_states[tid_b];
        B.slab_idx       = G_b;
        B.buf_hi         = 0;
        B.buf_lo         = 0;
        B.buf_avail      = 0;
        B.block_base_u32 = (int64_t)base_bytes_b / 4;
    }

    // sm_packed holds 2 nibbles per byte (one byte per fp8 pair).
    // Cache the current byte and select the appropriate nibble each iter.
    uint8_t sm_byte_a = 0, sm_byte_b = 0;

    for (int64_t i = 0; i < n_fp8_per_stream; i++) {
        if ((i & 1) == 0) {
            if (have_a) sm_byte_a = sign_mantissa[(i >> 1) * n_streams + tid_a];
            if (have_b) sm_byte_b = sign_mantissa[(i >> 1) * n_streams + tid_b];
        }
        uint8_t exp_a = 0, exp_b = 0;
        if (have_a) exp_a = decode_step(A, sfc_smem, compressed_u32, tid_in_block);
        if (have_b) exp_b = decode_step(B, sfc_smem, compressed_u32, tid_in_block);

        if (have_a) {
            uint8_t sm_nib = (i & 1) ? (sm_byte_a >> 4) : (sm_byte_a & 0xF);
            uint8_t fp8    = ((sm_nib & 0x8) << 4) | (exp_a << 3) | (sm_nib & 0x7);
            output[i * n_streams + tid_a] = fp8;
        }
        if (have_b) {
            uint8_t sm_nib = (i & 1) ? (sm_byte_b >> 4) : (sm_byte_b & 0xF);
            uint8_t fp8    = ((sm_nib & 0x8) << 4) | (exp_b << 3) | (sm_nib & 0x7);
            output[i * n_streams + tid_b] = fp8;
        }
    }
}

// Dump variant: same decode work but folds decoded bytes into a
// per-thread XOR digest instead of writing them back to HBM. For
// throughput measurement when the downstream consumer lives in SMEM
// (no HBM write-back scaffolding). Returned tensor is a digest, not
// decoded bytes.
__global__ void rans_decode_fp8_dump_kernel(
    const uint32_t* __restrict__ compressed_u32,
    const int32_t*  __restrict__ block_offsets,
    const uint32_t* __restrict__ final_states,
    const uint32_t* __restrict__ sfc_global,
    const uint8_t*  __restrict__ sign_mantissa,
    uint8_t*        __restrict__ digest_out,
    int64_t n_fp8_per_stream,
    int64_t n_streams)
{
    __shared__ uint32_t sfc_smem[rans_gpu::M_SIZE];

    for (int i = threadIdx.x; i < (int)rans_gpu::M_SIZE; i += blockDim.x) {
        sfc_smem[i] = sfc_global[i];
    }
    __syncthreads();

    int tid_in_block = threadIdx.x;
    int enc_block_a  = 2 * blockIdx.x;
    int enc_block_b  = 2 * blockIdx.x + 1;
    int64_t tid_a    = (int64_t)enc_block_a * blockDim.x + tid_in_block;
    int64_t tid_b    = (int64_t)enc_block_b * blockDim.x + tid_in_block;

    bool have_a = tid_a < n_streams;
    bool have_b = tid_b < n_streams;
    if (!have_a && !have_b) return;

    RansCtx A, B;
    if (have_a) {
        int32_t base_bytes_a = block_offsets[enc_block_a];
        int32_t next_bytes_a = block_offsets[enc_block_a + 1];
        int     G_a          = (next_bytes_a - base_bytes_a)
                               / ((int)blockDim.x * 4);
        A.x              = final_states[tid_a];
        A.slab_idx       = G_a;
        A.buf_hi         = 0;
        A.buf_lo         = 0;
        A.buf_avail      = 0;
        A.block_base_u32 = (int64_t)base_bytes_a / 4;
    }
    if (have_b) {
        int32_t base_bytes_b = block_offsets[enc_block_b];
        int32_t next_bytes_b = block_offsets[enc_block_b + 1];
        int     G_b          = (next_bytes_b - base_bytes_b)
                               / ((int)blockDim.x * 4);
        B.x              = final_states[tid_b];
        B.slab_idx       = G_b;
        B.buf_hi         = 0;
        B.buf_lo         = 0;
        B.buf_avail      = 0;
        B.block_base_u32 = (int64_t)base_bytes_b / 4;
    }

    uint8_t digest = 0;
    uint8_t sm_byte_a = 0, sm_byte_b = 0;

    for (int64_t i = 0; i < n_fp8_per_stream; i++) {
        if ((i & 1) == 0) {
            if (have_a) sm_byte_a = sign_mantissa[(i >> 1) * n_streams + tid_a];
            if (have_b) sm_byte_b = sign_mantissa[(i >> 1) * n_streams + tid_b];
        }
        uint8_t exp_a = 0, exp_b = 0;
        if (have_a) exp_a = decode_step(A, sfc_smem, compressed_u32, tid_in_block);
        if (have_b) exp_b = decode_step(B, sfc_smem, compressed_u32, tid_in_block);

        if (have_a) {
            uint8_t sm_nib = (i & 1) ? (sm_byte_a >> 4) : (sm_byte_a & 0xF);
            uint8_t fp8    = ((sm_nib & 0x8) << 4) | (exp_a << 3) | (sm_nib & 0x7);
            digest ^= fp8;
        }
        if (have_b) {
            uint8_t sm_nib = (i & 1) ? (sm_byte_b >> 4) : (sm_byte_b & 0xF);
            uint8_t fp8    = ((sm_nib & 0x8) << 4) | (exp_b << 3) | (sm_nib & 0x7);
            digest ^= fp8;
        }
    }

    digest_out[(int64_t)blockIdx.x * blockDim.x + tid_in_block] = digest;
}

// Shared setup: validate args, build the sfc table on CPU, upload.
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
    TORCH_CHECK(compressed.is_cuda() && compressed.dtype() == torch::kUInt8,
                "compressed must be uint8 CUDA");
    TORCH_CHECK(block_offsets.is_cuda() && block_offsets.dtype() == torch::kInt32,
                "block_offsets must be int32 CUDA");
    TORCH_CHECK(final_states.is_cuda() && final_states.dtype() == torch::kInt32,
                "final_states must be int32 CUDA");
    TORCH_CHECK(sign_mantissa.is_cuda() && sign_mantissa.dtype() == torch::kUInt8,
                "sign_mantissa must be uint8 CUDA");
    TORCH_CHECK(freqs.dtype() == torch::kInt32, "freqs must be int32");
    TORCH_CHECK(n_fp8_per_stream % 2 == 0,
                "n_fp8_per_stream must be even (sign+mantissa packing)");
    TORCH_CHECK(compressed.numel() % 4 == 0,
                "compressed byte length must be a multiple of 4");

    int64_t n_streams  = final_states.numel();
    int     n_alphabet = (int)freqs.numel();
    TORCH_CHECK(n_alphabet > 0 && n_alphabet <= rans_gpu::MAX_ALPHA,
                "alphabet size must be in [1, ", rans_gpu::MAX_ALPHA, "]");
    TORCH_CHECK(sign_mantissa.dim() == 2
                && sign_mantissa.size(0) == n_fp8_per_stream / 2
                && sign_mantissa.size(1) == n_streams,
                "sign_mantissa shape must be [n_fp8/2, n_streams]; got ",
                sign_mantissa.sizes());

    auto freqs_cpu = freqs.cpu().contiguous();
    const int32_t* f_ptr = freqs_cpu.data_ptr<int32_t>();

    std::vector<uint32_t> cum_vec(n_alphabet);
    uint32_t acc = 0;
    for (int i = 0; i < n_alphabet; i++) {
        TORCH_CHECK(f_ptr[i] >= 0, "freqs must be non-negative");
        cum_vec[i] = acc;
        acc += (uint32_t)f_ptr[i];
    }
    TORCH_CHECK(acc == rans_gpu::M_SIZE,
                "freqs must sum to ", rans_gpu::M_SIZE, "; got ", acc);

    auto sfc_cpu = torch::zeros({(int64_t)rans_gpu::M_SIZE}, torch::kInt32);
    uint32_t* sfc_ptr = reinterpret_cast<uint32_t*>(sfc_cpu.data_ptr<int32_t>());
    uint32_t pos = 0;
    for (int s = 0; s < n_alphabet; s++) {
        uint32_t f = (uint32_t)f_ptr[s];
        uint32_t c = cum_vec[s];
        TORCH_CHECK(f < 4096u && c < 4096u, "f and c must fit in 12 bits each");
        uint32_t entry = ((uint32_t)s & 0xFFu)
                       | ((f & 0xFFFu) << 8)
                       | ((c & 0xFFFu) << 20);
        for (uint32_t j = 0; j < f; j++) sfc_ptr[pos + j] = entry;
        pos += f;
    }

    int64_t n_enc_blocks = (n_streams + rans_gpu::BLOCK_STREAMS - 1)
                           / rans_gpu::BLOCK_STREAMS;
    if (n_enc_blocks < 1) n_enc_blocks = 1;
    TORCH_CHECK(block_offsets.numel() == n_enc_blocks + 1,
                "block_offsets length must be n_enc_blocks + 1 (",
                n_enc_blocks + 1, "); got ", block_offsets.numel());

    n_streams_out    = n_streams;
    n_enc_blocks_out = n_enc_blocks;
    return sfc_cpu.to(compressed.device());
}

// Returns [n_fp8_per_stream, n_streams] uint8 decoded bytes.
torch::Tensor gpu_rans_decode_fp8(
    torch::Tensor compressed,       // uint8 CUDA
    torch::Tensor block_offsets,    // int32 CUDA [n_enc_blocks + 1]
    torch::Tensor final_states,     // int32 CUDA [n_streams]
    torch::Tensor sign_mantissa,    // uint8 CUDA [n_fp8/2, n_streams]
    int64_t n_fp8_per_stream,
    torch::Tensor freqs)            // int32 [≤ 16]
{
    int64_t n_streams, n_enc_blocks;
    auto sfc_gpu = build_sfc_table(compressed, block_offsets, final_states,
                                   sign_mantissa, n_fp8_per_stream, freqs,
                                   n_streams, n_enc_blocks);

    auto output = torch::empty(
        {n_fp8_per_stream, n_streams},
        torch::TensorOptions().dtype(torch::kUInt8).device(compressed.device()));

    const int threads = rans_gpu::BLOCK_STREAMS;
    int64_t blocks = (n_enc_blocks + 1) / 2;   // dual-stream

    rans_decode_fp8_kernel<<<blocks, threads>>>(
        reinterpret_cast<const uint32_t*>(compressed.data_ptr<uint8_t>()),
        block_offsets.data_ptr<int32_t>(),
        reinterpret_cast<const uint32_t*>(final_states.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(sfc_gpu.data_ptr<int32_t>()),
        sign_mantissa.data_ptr<uint8_t>(),
        output.data_ptr<uint8_t>(),
        n_fp8_per_stream,
        n_streams);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

// Benchmark variant: identical decode, but emits a per-thread digest
// instead of the full decoded tensor. See gpu_rans_decode_fp8_dump
// kernel comment.
torch::Tensor gpu_rans_decode_fp8_dump(
    torch::Tensor compressed,
    torch::Tensor block_offsets,
    torch::Tensor final_states,
    torch::Tensor sign_mantissa,
    int64_t n_fp8_per_stream,
    torch::Tensor freqs)
{
    int64_t n_streams, n_enc_blocks;
    auto sfc_gpu = build_sfc_table(compressed, block_offsets, final_states,
                                   sign_mantissa, n_fp8_per_stream, freqs,
                                   n_streams, n_enc_blocks);

    const int threads = rans_gpu::BLOCK_STREAMS;
    int64_t blocks = (n_enc_blocks + 1) / 2;

    auto digest = torch::empty(
        {blocks * threads},
        torch::TensorOptions().dtype(torch::kUInt8).device(compressed.device()));

    rans_decode_fp8_dump_kernel<<<blocks, threads>>>(
        reinterpret_cast<const uint32_t*>(compressed.data_ptr<uint8_t>()),
        block_offsets.data_ptr<int32_t>(),
        reinterpret_cast<const uint32_t*>(final_states.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(sfc_gpu.data_ptr<int32_t>()),
        sign_mantissa.data_ptr<uint8_t>(),
        digest.data_ptr<uint8_t>(),
        n_fp8_per_stream,
        n_streams);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return digest;
}

// ─── L1-cache variant ────────────────────────────────────────────────
//
// Same O(1) sfc table lookup as the SMEM variant, but reads through L1
// via __ldg instead of shared memory. Hypothesis: L1's cache-line-based
// access path may handle random reads better than SMEM's 32-bank direct
// addressing, which suffers 3.4× replay on uniformly random slots.
// The 8 KB table fits in ~6% of the 128 KB L1.

__device__ __forceinline__ uint8_t decode_step_ldg(
    RansCtx& ctx,
    const uint32_t* __restrict__ sfc_ptr,
    const uint32_t* __restrict__ compressed_u32,
    int tid_in_block)
{
    uint32_t slot  = ctx.x & (rans_gpu::M_SIZE - 1);
    uint32_t entry = __ldg(&sfc_ptr[slot]);
    uint8_t  sym   = (uint8_t)(entry & 0xFF);
    uint32_t f     = (entry >> 8) & 0xFFF;
    uint32_t c     = (entry >> 20) & 0xFFF;
    ctx.x = (ctx.x >> rans_gpu::M_LOG) * f + (slot - c);

    while (ctx.x < rans_gpu::L_LOW) {
        if (ctx.buf_avail == 0) {
            ctx.slab_idx -= 2;
            int64_t off = ctx.block_base_u32
                + (int64_t)ctx.slab_idx * blockDim.x + tid_in_block;
            ctx.buf_lo = compressed_u32[off];
            ctx.buf_hi = compressed_u32[off + blockDim.x];
            ctx.buf_avail = 4;
        }
        ctx.x = (ctx.x << 16) | (ctx.buf_hi >> 16);
        ctx.buf_hi = (ctx.buf_hi << 16) | (ctx.buf_lo >> 16);
        ctx.buf_lo <<= 16;
        ctx.buf_avail--;
    }
    return sym;
}

__global__ void rans_decode_fp8_ldg_kernel(
    const uint32_t* __restrict__ compressed_u32,
    const int32_t*  __restrict__ block_offsets,
    const uint32_t* __restrict__ final_states,
    const uint32_t* __restrict__ sfc_global,
    const uint8_t*  __restrict__ sign_mantissa,
    uint8_t*        __restrict__ output,
    int64_t n_fp8_per_stream,
    int64_t n_streams)
{
    int tid_in_block = threadIdx.x;
    int enc_block_a  = 2 * blockIdx.x;
    int enc_block_b  = 2 * blockIdx.x + 1;
    int64_t tid_a    = (int64_t)enc_block_a * blockDim.x + tid_in_block;
    int64_t tid_b    = (int64_t)enc_block_b * blockDim.x + tid_in_block;

    bool have_a = tid_a < n_streams;
    bool have_b = tid_b < n_streams;
    if (!have_a && !have_b) return;

    RansCtx A, B;
    if (have_a) {
        int32_t base = block_offsets[enc_block_a];
        int32_t next = block_offsets[enc_block_a + 1];
        int G = (next - base) / ((int)blockDim.x * 4);
        A.x = final_states[tid_a]; A.slab_idx = G;
        A.buf_hi = 0; A.buf_lo = 0; A.buf_avail = 0;
        A.block_base_u32 = (int64_t)base / 4;
    }
    if (have_b) {
        int32_t base = block_offsets[enc_block_b];
        int32_t next = block_offsets[enc_block_b + 1];
        int G = (next - base) / ((int)blockDim.x * 4);
        B.x = final_states[tid_b]; B.slab_idx = G;
        B.buf_hi = 0; B.buf_lo = 0; B.buf_avail = 0;
        B.block_base_u32 = (int64_t)base / 4;
    }

    uint8_t sm_byte_a = 0, sm_byte_b = 0;
    for (int64_t i = 0; i < n_fp8_per_stream; i++) {
        if ((i & 1) == 0) {
            if (have_a) sm_byte_a = sign_mantissa[(i >> 1) * n_streams + tid_a];
            if (have_b) sm_byte_b = sign_mantissa[(i >> 1) * n_streams + tid_b];
        }
        uint8_t exp_a = 0, exp_b = 0;
        if (have_a) exp_a = decode_step_ldg(A, sfc_global, compressed_u32, tid_in_block);
        if (have_b) exp_b = decode_step_ldg(B, sfc_global, compressed_u32, tid_in_block);

        if (have_a) {
            uint8_t sm_nib = (i & 1) ? (sm_byte_a >> 4) : (sm_byte_a & 0xF);
            uint8_t fp8 = ((sm_nib & 0x8) << 4) | (exp_a << 3) | (sm_nib & 0x7);
            output[i * n_streams + tid_a] = fp8;
        }
        if (have_b) {
            uint8_t sm_nib = (i & 1) ? (sm_byte_b >> 4) : (sm_byte_b & 0xF);
            uint8_t fp8 = ((sm_nib & 0x8) << 4) | (exp_b << 3) | (sm_nib & 0x7);
            output[i * n_streams + tid_b] = fp8;
        }
    }
}

__global__ void rans_decode_fp8_ldg_dump_kernel(
    const uint32_t* __restrict__ compressed_u32,
    const int32_t*  __restrict__ block_offsets,
    const uint32_t* __restrict__ final_states,
    const uint32_t* __restrict__ sfc_global,
    const uint8_t*  __restrict__ sign_mantissa,
    uint8_t*        __restrict__ digest_out,
    int64_t n_fp8_per_stream,
    int64_t n_streams)
{
    int tid_in_block = threadIdx.x;
    int enc_block_a  = 2 * blockIdx.x;
    int enc_block_b  = 2 * blockIdx.x + 1;
    int64_t tid_a    = (int64_t)enc_block_a * blockDim.x + tid_in_block;
    int64_t tid_b    = (int64_t)enc_block_b * blockDim.x + tid_in_block;

    bool have_a = tid_a < n_streams;
    bool have_b = tid_b < n_streams;
    if (!have_a && !have_b) return;

    RansCtx A, B;
    if (have_a) {
        int32_t base = block_offsets[enc_block_a];
        int32_t next = block_offsets[enc_block_a + 1];
        int G = (next - base) / ((int)blockDim.x * 4);
        A.x = final_states[tid_a]; A.slab_idx = G;
        A.buf_hi = 0; A.buf_lo = 0; A.buf_avail = 0;
        A.block_base_u32 = (int64_t)base / 4;
    }
    if (have_b) {
        int32_t base = block_offsets[enc_block_b];
        int32_t next = block_offsets[enc_block_b + 1];
        int G = (next - base) / ((int)blockDim.x * 4);
        B.x = final_states[tid_b]; B.slab_idx = G;
        B.buf_hi = 0; B.buf_lo = 0; B.buf_avail = 0;
        B.block_base_u32 = (int64_t)base / 4;
    }

    uint8_t digest = 0, sm_byte_a = 0, sm_byte_b = 0;
    for (int64_t i = 0; i < n_fp8_per_stream; i++) {
        if ((i & 1) == 0) {
            if (have_a) sm_byte_a = sign_mantissa[(i >> 1) * n_streams + tid_a];
            if (have_b) sm_byte_b = sign_mantissa[(i >> 1) * n_streams + tid_b];
        }
        uint8_t exp_a = 0, exp_b = 0;
        if (have_a) exp_a = decode_step_ldg(A, sfc_global, compressed_u32, tid_in_block);
        if (have_b) exp_b = decode_step_ldg(B, sfc_global, compressed_u32, tid_in_block);

        if (have_a) {
            uint8_t sm_nib = (i & 1) ? (sm_byte_a >> 4) : (sm_byte_a & 0xF);
            digest ^= ((sm_nib & 0x8) << 4) | (exp_a << 3) | (sm_nib & 0x7);
        }
        if (have_b) {
            uint8_t sm_nib = (i & 1) ? (sm_byte_b >> 4) : (sm_byte_b & 0xF);
            digest ^= ((sm_nib & 0x8) << 4) | (exp_b << 3) | (sm_nib & 0x7);
        }
    }
    digest_out[(int64_t)blockIdx.x * blockDim.x + tid_in_block] = digest;
}

torch::Tensor gpu_rans_decode_fp8_ldg(
    torch::Tensor compressed,
    torch::Tensor block_offsets,
    torch::Tensor final_states,
    torch::Tensor sign_mantissa,
    int64_t n_fp8_per_stream,
    torch::Tensor freqs)
{
    int64_t n_streams, n_enc_blocks;
    auto sfc_gpu = build_sfc_table(compressed, block_offsets, final_states,
                                   sign_mantissa, n_fp8_per_stream, freqs,
                                   n_streams, n_enc_blocks);

    auto output = torch::empty(
        {n_fp8_per_stream, n_streams},
        torch::TensorOptions().dtype(torch::kUInt8).device(compressed.device()));

    const int threads = rans_gpu::BLOCK_STREAMS;
    int64_t blocks = (n_enc_blocks + 1) / 2;

    rans_decode_fp8_ldg_kernel<<<blocks, threads>>>(
        reinterpret_cast<const uint32_t*>(compressed.data_ptr<uint8_t>()),
        block_offsets.data_ptr<int32_t>(),
        reinterpret_cast<const uint32_t*>(final_states.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(sfc_gpu.data_ptr<int32_t>()),
        sign_mantissa.data_ptr<uint8_t>(),
        output.data_ptr<uint8_t>(),
        n_fp8_per_stream,
        n_streams);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor gpu_rans_decode_fp8_ldg_dump(
    torch::Tensor compressed,
    torch::Tensor block_offsets,
    torch::Tensor final_states,
    torch::Tensor sign_mantissa,
    int64_t n_fp8_per_stream,
    torch::Tensor freqs)
{
    int64_t n_streams, n_enc_blocks;
    auto sfc_gpu = build_sfc_table(compressed, block_offsets, final_states,
                                   sign_mantissa, n_fp8_per_stream, freqs,
                                   n_streams, n_enc_blocks);

    const int threads = rans_gpu::BLOCK_STREAMS;
    int64_t blocks = (n_enc_blocks + 1) / 2;

    auto digest = torch::empty(
        {blocks * threads},
        torch::TensorOptions().dtype(torch::kUInt8).device(compressed.device()));

    rans_decode_fp8_ldg_dump_kernel<<<blocks, threads>>>(
        reinterpret_cast<const uint32_t*>(compressed.data_ptr<uint8_t>()),
        block_offsets.data_ptr<int32_t>(),
        reinterpret_cast<const uint32_t*>(final_states.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(sfc_gpu.data_ptr<int32_t>()),
        sign_mantissa.data_ptr<uint8_t>(),
        digest.data_ptr<uint8_t>(),
        n_fp8_per_stream,
        n_streams);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return digest;
}

// ─── Register-scan variant ───────────────────────────────────────────
//
// Replaces the 2048-entry random-indexed sfc table with a 17-entry cum
// table loaded into per-thread registers. The sfc table causes 71%
// excess SMEM wavefronts due to bank conflicts (ncu: 3.4× replay on
// random-indexed 2048-entry table). With only 16 symbols, a branchless
// linear scan over 17 cumulative frequencies finds the symbol in ~60
// ALU instructions with zero SMEM access in the hot loop.
//
// Register cost: 17 uint32 cum values kept live through the inner loop.
// Total ~55 regs/thread (vs 38 for sfc variant), well under the 255
// limit. SMEM drops from 8 KB (sfc) to 68 bytes (cum staging), so
// the block-per-SM limit shifts from SMEM-bound to register-bound.

__device__ __forceinline__ uint8_t decode_step_regscan(
    RansCtx& ctx,
    const uint32_t (&cum_r)[17],
    const uint32_t* __restrict__ compressed_u32,
    int tid_in_block)
{
    uint32_t slot = ctx.x & (rans_gpu::M_SIZE - 1);

    // Branchless linear scan: find sym where cum[sym] <= slot < cum[sym+1].
    // cum is sorted, so the last true `slot >= cum[s]` gives the answer.
    // All indices are compile-time constants (#pragma unroll) → registers.
    int sym = 0;
    uint32_t c = cum_r[0], c_next = cum_r[1];

    #pragma unroll
    for (int s = 1; s < 16; s++) {
        if (slot >= cum_r[s]) {
            sym = s;
            c = cum_r[s];
            c_next = cum_r[s + 1];
        }
    }

    uint32_t f = c_next - c;
    ctx.x = (ctx.x >> rans_gpu::M_LOG) * f + (slot - c);

    while (ctx.x < rans_gpu::L_LOW) {
        if (ctx.buf_avail == 0) {
            ctx.slab_idx -= 2;
            int64_t off = ctx.block_base_u32
                + (int64_t)ctx.slab_idx * blockDim.x + tid_in_block;
            ctx.buf_lo = compressed_u32[off];
            ctx.buf_hi = compressed_u32[off + blockDim.x];
            ctx.buf_avail = 4;
        }
        ctx.x = (ctx.x << 16) | (ctx.buf_hi >> 16);
        ctx.buf_hi = (ctx.buf_hi << 16) | (ctx.buf_lo >> 16);
        ctx.buf_lo <<= 16;
        ctx.buf_avail--;
    }
    return (uint8_t)sym;
}

__global__ void rans_decode_fp8_regscan_kernel(
    const uint32_t* __restrict__ compressed_u32,
    const int32_t*  __restrict__ block_offsets,
    const uint32_t* __restrict__ final_states,
    const uint32_t* __restrict__ cum_global,
    const uint8_t*  __restrict__ sign_mantissa,
    uint8_t*        __restrict__ output,
    int64_t n_fp8_per_stream,
    int64_t n_streams)
{
    __shared__ uint32_t cum_smem[17];
    if (threadIdx.x < 17) cum_smem[threadIdx.x] = cum_global[threadIdx.x];
    __syncthreads();

    uint32_t cum_r[17];
    #pragma unroll
    for (int i = 0; i < 17; i++) cum_r[i] = cum_smem[i];

    int tid_in_block = threadIdx.x;
    int enc_block_a  = 2 * blockIdx.x;
    int enc_block_b  = 2 * blockIdx.x + 1;
    int64_t tid_a    = (int64_t)enc_block_a * blockDim.x + tid_in_block;
    int64_t tid_b    = (int64_t)enc_block_b * blockDim.x + tid_in_block;

    bool have_a = tid_a < n_streams;
    bool have_b = tid_b < n_streams;
    if (!have_a && !have_b) return;

    RansCtx A, B;
    if (have_a) {
        int32_t base = block_offsets[enc_block_a];
        int32_t next = block_offsets[enc_block_a + 1];
        int G = (next - base) / ((int)blockDim.x * 4);
        A.x = final_states[tid_a]; A.slab_idx = G;
        A.buf_hi = 0; A.buf_lo = 0; A.buf_avail = 0;
        A.block_base_u32 = (int64_t)base / 4;
    }
    if (have_b) {
        int32_t base = block_offsets[enc_block_b];
        int32_t next = block_offsets[enc_block_b + 1];
        int G = (next - base) / ((int)blockDim.x * 4);
        B.x = final_states[tid_b]; B.slab_idx = G;
        B.buf_hi = 0; B.buf_lo = 0; B.buf_avail = 0;
        B.block_base_u32 = (int64_t)base / 4;
    }

    uint8_t sm_byte_a = 0, sm_byte_b = 0;
    for (int64_t i = 0; i < n_fp8_per_stream; i++) {
        if ((i & 1) == 0) {
            if (have_a) sm_byte_a = sign_mantissa[(i >> 1) * n_streams + tid_a];
            if (have_b) sm_byte_b = sign_mantissa[(i >> 1) * n_streams + tid_b];
        }
        uint8_t exp_a = 0, exp_b = 0;
        if (have_a) exp_a = decode_step_regscan(A, cum_r, compressed_u32, tid_in_block);
        if (have_b) exp_b = decode_step_regscan(B, cum_r, compressed_u32, tid_in_block);

        if (have_a) {
            uint8_t sm_nib = (i & 1) ? (sm_byte_a >> 4) : (sm_byte_a & 0xF);
            uint8_t fp8 = ((sm_nib & 0x8) << 4) | (exp_a << 3) | (sm_nib & 0x7);
            output[i * n_streams + tid_a] = fp8;
        }
        if (have_b) {
            uint8_t sm_nib = (i & 1) ? (sm_byte_b >> 4) : (sm_byte_b & 0xF);
            uint8_t fp8 = ((sm_nib & 0x8) << 4) | (exp_b << 3) | (sm_nib & 0x7);
            output[i * n_streams + tid_b] = fp8;
        }
    }
}

__global__ void rans_decode_fp8_regscan_dump_kernel(
    const uint32_t* __restrict__ compressed_u32,
    const int32_t*  __restrict__ block_offsets,
    const uint32_t* __restrict__ final_states,
    const uint32_t* __restrict__ cum_global,
    const uint8_t*  __restrict__ sign_mantissa,
    uint8_t*        __restrict__ digest_out,
    int64_t n_fp8_per_stream,
    int64_t n_streams)
{
    __shared__ uint32_t cum_smem[17];
    if (threadIdx.x < 17) cum_smem[threadIdx.x] = cum_global[threadIdx.x];
    __syncthreads();

    uint32_t cum_r[17];
    #pragma unroll
    for (int i = 0; i < 17; i++) cum_r[i] = cum_smem[i];

    int tid_in_block = threadIdx.x;
    int enc_block_a  = 2 * blockIdx.x;
    int enc_block_b  = 2 * blockIdx.x + 1;
    int64_t tid_a    = (int64_t)enc_block_a * blockDim.x + tid_in_block;
    int64_t tid_b    = (int64_t)enc_block_b * blockDim.x + tid_in_block;

    bool have_a = tid_a < n_streams;
    bool have_b = tid_b < n_streams;
    if (!have_a && !have_b) return;

    RansCtx A, B;
    if (have_a) {
        int32_t base = block_offsets[enc_block_a];
        int32_t next = block_offsets[enc_block_a + 1];
        int G = (next - base) / ((int)blockDim.x * 4);
        A.x = final_states[tid_a]; A.slab_idx = G;
        A.buf_hi = 0; A.buf_lo = 0; A.buf_avail = 0;
        A.block_base_u32 = (int64_t)base / 4;
    }
    if (have_b) {
        int32_t base = block_offsets[enc_block_b];
        int32_t next = block_offsets[enc_block_b + 1];
        int G = (next - base) / ((int)blockDim.x * 4);
        B.x = final_states[tid_b]; B.slab_idx = G;
        B.buf_hi = 0; B.buf_lo = 0; B.buf_avail = 0;
        B.block_base_u32 = (int64_t)base / 4;
    }

    uint8_t digest = 0, sm_byte_a = 0, sm_byte_b = 0;
    for (int64_t i = 0; i < n_fp8_per_stream; i++) {
        if ((i & 1) == 0) {
            if (have_a) sm_byte_a = sign_mantissa[(i >> 1) * n_streams + tid_a];
            if (have_b) sm_byte_b = sign_mantissa[(i >> 1) * n_streams + tid_b];
        }
        uint8_t exp_a = 0, exp_b = 0;
        if (have_a) exp_a = decode_step_regscan(A, cum_r, compressed_u32, tid_in_block);
        if (have_b) exp_b = decode_step_regscan(B, cum_r, compressed_u32, tid_in_block);

        if (have_a) {
            uint8_t sm_nib = (i & 1) ? (sm_byte_a >> 4) : (sm_byte_a & 0xF);
            digest ^= ((sm_nib & 0x8) << 4) | (exp_a << 3) | (sm_nib & 0x7);
        }
        if (have_b) {
            uint8_t sm_nib = (i & 1) ? (sm_byte_b >> 4) : (sm_byte_b & 0xF);
            digest ^= ((sm_nib & 0x8) << 4) | (exp_b << 3) | (sm_nib & 0x7);
        }
    }
    digest_out[(int64_t)blockIdx.x * blockDim.x + tid_in_block] = digest;
}

// Host helper: build 17-entry cumulative table [cum[0], ..., cum[15], M].
static torch::Tensor build_cum_table(
    torch::Tensor compressed,
    torch::Tensor block_offsets,
    torch::Tensor final_states,
    torch::Tensor sign_mantissa,
    int64_t n_fp8_per_stream,
    torch::Tensor freqs,
    int64_t& n_streams_out,
    int64_t& n_enc_blocks_out)
{
    TORCH_CHECK(compressed.is_cuda() && compressed.dtype() == torch::kUInt8,
                "compressed must be uint8 CUDA");
    TORCH_CHECK(block_offsets.is_cuda() && block_offsets.dtype() == torch::kInt32,
                "block_offsets must be int32 CUDA");
    TORCH_CHECK(final_states.is_cuda() && final_states.dtype() == torch::kInt32,
                "final_states must be int32 CUDA");
    TORCH_CHECK(sign_mantissa.is_cuda() && sign_mantissa.dtype() == torch::kUInt8,
                "sign_mantissa must be uint8 CUDA");
    TORCH_CHECK(freqs.dtype() == torch::kInt32, "freqs must be int32");
    TORCH_CHECK(n_fp8_per_stream % 2 == 0,
                "n_fp8_per_stream must be even (sign+mantissa packing)");
    TORCH_CHECK(compressed.numel() % 4 == 0,
                "compressed byte length must be a multiple of 4");

    int64_t n_streams  = final_states.numel();
    int     n_alphabet = (int)freqs.numel();
    TORCH_CHECK(n_alphabet > 0 && n_alphabet <= rans_gpu::MAX_ALPHA,
                "alphabet size must be in [1, ", rans_gpu::MAX_ALPHA, "]");
    TORCH_CHECK(sign_mantissa.dim() == 2
                && sign_mantissa.size(0) == n_fp8_per_stream / 2
                && sign_mantissa.size(1) == n_streams,
                "sign_mantissa shape must be [n_fp8/2, n_streams]");

    auto freqs_cpu = freqs.cpu().contiguous();
    const int32_t* f_ptr = freqs_cpu.data_ptr<int32_t>();

    auto cum_cpu = torch::zeros({17}, torch::kInt32);
    auto* cum_ptr = reinterpret_cast<uint32_t*>(cum_cpu.data_ptr<int32_t>());
    uint32_t acc = 0;
    for (int i = 0; i < n_alphabet; i++) {
        TORCH_CHECK(f_ptr[i] >= 0, "freqs must be non-negative");
        cum_ptr[i] = acc;
        acc += (uint32_t)f_ptr[i];
    }
    cum_ptr[n_alphabet] = acc;
    for (int i = n_alphabet + 1; i <= 16; i++) cum_ptr[i] = acc;
    TORCH_CHECK(acc == rans_gpu::M_SIZE,
                "freqs must sum to ", rans_gpu::M_SIZE, "; got ", acc);

    int64_t n_enc_blocks = (n_streams + rans_gpu::BLOCK_STREAMS - 1)
                           / rans_gpu::BLOCK_STREAMS;
    if (n_enc_blocks < 1) n_enc_blocks = 1;
    TORCH_CHECK(block_offsets.numel() == n_enc_blocks + 1,
                "block_offsets length must be n_enc_blocks + 1");

    n_streams_out    = n_streams;
    n_enc_blocks_out = n_enc_blocks;
    return cum_cpu.to(compressed.device());
}

torch::Tensor gpu_rans_decode_fp8_regscan(
    torch::Tensor compressed,
    torch::Tensor block_offsets,
    torch::Tensor final_states,
    torch::Tensor sign_mantissa,
    int64_t n_fp8_per_stream,
    torch::Tensor freqs)
{
    int64_t n_streams, n_enc_blocks;
    auto cum_gpu = build_cum_table(compressed, block_offsets, final_states,
                                   sign_mantissa, n_fp8_per_stream, freqs,
                                   n_streams, n_enc_blocks);

    auto output = torch::empty(
        {n_fp8_per_stream, n_streams},
        torch::TensorOptions().dtype(torch::kUInt8).device(compressed.device()));

    const int threads = rans_gpu::BLOCK_STREAMS;
    int64_t blocks = (n_enc_blocks + 1) / 2;

    rans_decode_fp8_regscan_kernel<<<blocks, threads>>>(
        reinterpret_cast<const uint32_t*>(compressed.data_ptr<uint8_t>()),
        block_offsets.data_ptr<int32_t>(),
        reinterpret_cast<const uint32_t*>(final_states.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(cum_gpu.data_ptr<int32_t>()),
        sign_mantissa.data_ptr<uint8_t>(),
        output.data_ptr<uint8_t>(),
        n_fp8_per_stream,
        n_streams);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor gpu_rans_decode_fp8_regscan_dump(
    torch::Tensor compressed,
    torch::Tensor block_offsets,
    torch::Tensor final_states,
    torch::Tensor sign_mantissa,
    int64_t n_fp8_per_stream,
    torch::Tensor freqs)
{
    int64_t n_streams, n_enc_blocks;
    auto cum_gpu = build_cum_table(compressed, block_offsets, final_states,
                                   sign_mantissa, n_fp8_per_stream, freqs,
                                   n_streams, n_enc_blocks);

    const int threads = rans_gpu::BLOCK_STREAMS;
    int64_t blocks = (n_enc_blocks + 1) / 2;

    auto digest = torch::empty(
        {blocks * threads},
        torch::TensorOptions().dtype(torch::kUInt8).device(compressed.device()));

    rans_decode_fp8_regscan_dump_kernel<<<blocks, threads>>>(
        reinterpret_cast<const uint32_t*>(compressed.data_ptr<uint8_t>()),
        block_offsets.data_ptr<int32_t>(),
        reinterpret_cast<const uint32_t*>(final_states.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(cum_gpu.data_ptr<int32_t>()),
        sign_mantissa.data_ptr<uint8_t>(),
        digest.data_ptr<uint8_t>(),
        n_fp8_per_stream,
        n_streams);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return digest;
}

// ─── Pair-alphabet variant (M=4096, 256 symbols, __ldg) ─────────────
//
// Decodes TWO exponent nibbles per rANS step using a 256-symbol pair
// alphabet with M=4096. Halves the number of decode steps per fp8 byte.
// Table: 4096 × 4 = 16 KB, read via __ldg (fits in L1's 128 KB).
// Same sfc packing: sym(8) | f(12) | c(12) = 32 bits per entry.

namespace rans_gpu_pair {
constexpr int      M_LOG         = 12;
constexpr uint32_t M_SIZE        = 1u << M_LOG;   // 4096
constexpr uint32_t L_LOW         = 1u << 16;
constexpr int      BLOCK_STREAMS = 128;
}

// Branched renorm (original) — used by correctness-checked full-output kernels.
__device__ __forceinline__ uint8_t decode_step_pair_ldg(
    RansCtx& ctx,
    const uint32_t* __restrict__ sfc_ptr,
    const uint32_t* __restrict__ compressed_u32,
    int tid_in_block)
{
    uint32_t slot  = ctx.x & (rans_gpu_pair::M_SIZE - 1);
    uint32_t entry = __ldg(&sfc_ptr[slot]);
    uint8_t  sym   = (uint8_t)(entry & 0xFF);
    uint32_t f     = (entry >> 8) & 0xFFF;
    uint32_t c     = (entry >> 20) & 0xFFF;
    ctx.x = (ctx.x >> rans_gpu_pair::M_LOG) * f + (slot - c);

    while (ctx.x < rans_gpu_pair::L_LOW) {
        if (ctx.buf_avail == 0) {
            ctx.slab_idx -= 2;
            int64_t off = ctx.block_base_u32
                + (int64_t)ctx.slab_idx * blockDim.x + tid_in_block;
            ctx.buf_lo = compressed_u32[off];
            ctx.buf_hi = compressed_u32[off + blockDim.x];
            ctx.buf_avail = 4;
        }
        ctx.x = (ctx.x << 16) | (ctx.buf_hi >> 16);
        ctx.buf_hi = (ctx.buf_hi << 16) | (ctx.buf_lo >> 16);
        ctx.buf_lo <<= 16;
        ctx.buf_avail--;
    }
    return sym;
}

// Branchless renorm variant. With M=4096, x_min after decode = L/M = 16.
// One renorm step gives x >= 16×2^16 = 1M >> L_LOW, so at most 1 step
// is ever needed. We replace the while+if with predicated operations:
// all threads compute the renormed values; only threads with x < L_LOW
// apply them. Zero warp divergence on the renorm path.
//
// The buffer refill (buf_avail < 0 after predicated decrement) remains
// branched but fires ~10× less often than renorm (~every 11 decode steps).
__device__ __forceinline__ uint8_t decode_step_pair_ldg_branchless(
    RansCtx& ctx,
    const uint32_t* __restrict__ sfc_ptr,
    const uint32_t* __restrict__ compressed_u32,
    int tid_in_block)
{
    uint32_t slot  = ctx.x & (rans_gpu_pair::M_SIZE - 1);
    uint32_t entry = __ldg(&sfc_ptr[slot]);
    uint8_t  sym   = (uint8_t)(entry & 0xFF);
    uint32_t f     = (entry >> 8) & 0xFFF;
    uint32_t c     = (entry >> 20) & 0xFFF;
    ctx.x = (ctx.x >> rans_gpu_pair::M_LOG) * f + (slot - c);

    bool need = (ctx.x < rans_gpu_pair::L_LOW);

    // Buffer refill BEFORE consume — branched, but fires only when a
    // thread needs renorm AND its buffer is empty (~9% of steps).
    if (need && ctx.buf_avail == 0) {
        ctx.slab_idx -= 2;
        int64_t off = ctx.block_base_u32
            + (int64_t)ctx.slab_idx * blockDim.x + tid_in_block;
        ctx.buf_lo = compressed_u32[off];
        ctx.buf_hi = compressed_u32[off + blockDim.x];
        ctx.buf_avail = 4;
    }

    // Predicated consume — no branch divergence on the renorm path.
    uint32_t x_renormed  = (ctx.x << 16) | (ctx.buf_hi >> 16);
    uint32_t bh_shifted  = (ctx.buf_hi << 16) | (ctx.buf_lo >> 16);
    uint32_t bl_shifted  = ctx.buf_lo << 16;
    int      avail_dec   = ctx.buf_avail - 1;

    ctx.x          = need ? x_renormed : ctx.x;
    ctx.buf_hi     = need ? bh_shifted  : ctx.buf_hi;
    ctx.buf_lo     = need ? bl_shifted  : ctx.buf_lo;
    ctx.buf_avail  = need ? avail_dec   : ctx.buf_avail;
    return sym;
}

__global__ void rans_decode_fp8_pair_ldg_kernel(
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
    int enc_a = 2 * blockIdx.x;
    int enc_b = 2 * blockIdx.x + 1;
    int64_t tid_a = (int64_t)enc_a * blockDim.x + tid;
    int64_t tid_b = (int64_t)enc_b * blockDim.x + tid;

    bool have_a = tid_a < n_streams;
    bool have_b = tid_b < n_streams;
    if (!have_a && !have_b) return;

    RansCtx A, B;
    if (have_a) {
        int32_t base = block_offsets[enc_a];
        int32_t next = block_offsets[enc_a + 1];
        int G = (next - base) / ((int)blockDim.x * 4);
        A.x = final_states[tid_a]; A.slab_idx = G;
        A.buf_hi = 0; A.buf_lo = 0; A.buf_avail = 0;
        A.block_base_u32 = (int64_t)base / 4;
    }
    if (have_b) {
        int32_t base = block_offsets[enc_b];
        int32_t next = block_offsets[enc_b + 1];
        int G = (next - base) / ((int)blockDim.x * 4);
        B.x = final_states[tid_b]; B.slab_idx = G;
        B.buf_hi = 0; B.buf_lo = 0; B.buf_avail = 0;
        B.block_base_u32 = (int64_t)base / 4;
    }

    int64_t n_pairs = n_fp8_per_stream / 2;
    for (int64_t i = 0; i < n_pairs; i++) {
        uint8_t pair_a = 0, pair_b = 0;
        if (have_a) pair_a = decode_step_pair_ldg(A, sfc_global, compressed_u32, tid);
        if (have_b) pair_b = decode_step_pair_ldg(B, sfc_global, compressed_u32, tid);

        uint8_t sm_a = 0, sm_b = 0;
        if (have_a) sm_a = sign_mantissa[i * n_streams + tid_a];
        if (have_b) sm_b = sign_mantissa[i * n_streams + tid_b];

        if (have_a) {
            uint8_t e0 = pair_a >> 4, e1 = pair_a & 0xF;
            uint8_t s0 = sm_a & 0xF, s1 = sm_a >> 4;
            output[(2*i)   * n_streams + tid_a] = ((s0&8)<<4) | (e0<<3) | (s0&7);
            output[(2*i+1) * n_streams + tid_a] = ((s1&8)<<4) | (e1<<3) | (s1&7);
        }
        if (have_b) {
            uint8_t e0 = pair_b >> 4, e1 = pair_b & 0xF;
            uint8_t s0 = sm_b & 0xF, s1 = sm_b >> 4;
            output[(2*i)   * n_streams + tid_b] = ((s0&8)<<4) | (e0<<3) | (s0&7);
            output[(2*i+1) * n_streams + tid_b] = ((s1&8)<<4) | (e1<<3) | (s1&7);
        }
    }
}

__global__ void rans_decode_fp8_pair_ldg_dump_kernel(
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
    int enc_a = 2 * blockIdx.x;
    int enc_b = 2 * blockIdx.x + 1;
    int64_t tid_a = (int64_t)enc_a * blockDim.x + tid;
    int64_t tid_b = (int64_t)enc_b * blockDim.x + tid;

    bool have_a = tid_a < n_streams;
    bool have_b = tid_b < n_streams;
    if (!have_a && !have_b) return;

    RansCtx A, B;
    if (have_a) {
        int32_t base = block_offsets[enc_a];
        int32_t next = block_offsets[enc_a + 1];
        int G = (next - base) / ((int)blockDim.x * 4);
        A.x = final_states[tid_a]; A.slab_idx = G;
        A.buf_hi = 0; A.buf_lo = 0; A.buf_avail = 0;
        A.block_base_u32 = (int64_t)base / 4;
    }
    if (have_b) {
        int32_t base = block_offsets[enc_b];
        int32_t next = block_offsets[enc_b + 1];
        int G = (next - base) / ((int)blockDim.x * 4);
        B.x = final_states[tid_b]; B.slab_idx = G;
        B.buf_hi = 0; B.buf_lo = 0; B.buf_avail = 0;
        B.block_base_u32 = (int64_t)base / 4;
    }

    uint8_t digest = 0;
    int64_t n_pairs = n_fp8_per_stream / 2;
    for (int64_t i = 0; i < n_pairs; i++) {
        uint8_t pair_a = 0, pair_b = 0;
        if (have_a) pair_a = decode_step_pair_ldg(A, sfc_global, compressed_u32, tid);
        if (have_b) pair_b = decode_step_pair_ldg(B, sfc_global, compressed_u32, tid);

        uint8_t sm_a = 0, sm_b = 0;
        if (have_a) sm_a = sign_mantissa[i * n_streams + tid_a];
        if (have_b) sm_b = sign_mantissa[i * n_streams + tid_b];

        if (have_a) {
            uint8_t e0 = pair_a >> 4, e1 = pair_a & 0xF;
            uint8_t s0 = sm_a & 0xF, s1 = sm_a >> 4;
            digest ^= ((s0&8)<<4) | (e0<<3) | (s0&7);
            digest ^= ((s1&8)<<4) | (e1<<3) | (s1&7);
        }
        if (have_b) {
            uint8_t e0 = pair_b >> 4, e1 = pair_b & 0xF;
            uint8_t s0 = sm_b & 0xF, s1 = sm_b >> 4;
            digest ^= ((s0&8)<<4) | (e0<<3) | (s0&7);
            digest ^= ((s1&8)<<4) | (e1<<3) | (s1&7);
        }
    }
    digest_out[(int64_t)blockIdx.x * blockDim.x + tid] = digest;
}

static torch::Tensor build_sfc_table_pair(
    torch::Tensor compressed, torch::Tensor block_offsets,
    torch::Tensor final_states, torch::Tensor sign_mantissa,
    int64_t n_fp8_per_stream, torch::Tensor freqs,
    int64_t& n_streams_out, int64_t& n_enc_blocks_out)
{
    TORCH_CHECK(compressed.is_cuda() && compressed.dtype() == torch::kUInt8);
    TORCH_CHECK(n_fp8_per_stream % 2 == 0);
    TORCH_CHECK(compressed.numel() % 4 == 0);
    TORCH_CHECK(freqs.dtype() == torch::kInt32);

    int64_t n_streams  = final_states.numel();
    int     n_alphabet = (int)freqs.numel();
    TORCH_CHECK(n_alphabet > 0 && n_alphabet <= 256);
    TORCH_CHECK(sign_mantissa.dim() == 2
                && sign_mantissa.size(0) == n_fp8_per_stream / 2
                && sign_mantissa.size(1) == n_streams);

    auto freqs_cpu = freqs.cpu().contiguous();
    const int32_t* f_ptr = freqs_cpu.data_ptr<int32_t>();

    std::vector<uint32_t> cum_vec(n_alphabet);
    uint32_t acc = 0;
    for (int i = 0; i < n_alphabet; i++) {
        cum_vec[i] = acc;
        acc += (uint32_t)f_ptr[i];
    }
    TORCH_CHECK(acc == rans_gpu_pair::M_SIZE,
                "pair freqs must sum to ", rans_gpu_pair::M_SIZE);

    auto sfc_cpu = torch::zeros({(int64_t)rans_gpu_pair::M_SIZE}, torch::kInt32);
    auto* sfc_ptr = reinterpret_cast<uint32_t*>(sfc_cpu.data_ptr<int32_t>());
    uint32_t pos = 0;
    for (int s = 0; s < n_alphabet; s++) {
        uint32_t f = (uint32_t)f_ptr[s];
        uint32_t c = cum_vec[s];
        uint32_t entry = ((uint32_t)s & 0xFFu)
                       | ((f & 0xFFFu) << 8)
                       | ((c & 0xFFFu) << 20);
        for (uint32_t j = 0; j < f; j++) sfc_ptr[pos + j] = entry;
        pos += f;
    }

    int64_t n_enc_blocks = (n_streams + rans_gpu_pair::BLOCK_STREAMS - 1)
                           / rans_gpu_pair::BLOCK_STREAMS;
    if (n_enc_blocks < 1) n_enc_blocks = 1;
    TORCH_CHECK(block_offsets.numel() == n_enc_blocks + 1);

    n_streams_out    = n_streams;
    n_enc_blocks_out = n_enc_blocks;
    return sfc_cpu.to(compressed.device());
}

torch::Tensor gpu_rans_decode_fp8_pair_ldg(
    torch::Tensor compressed, torch::Tensor block_offsets,
    torch::Tensor final_states, torch::Tensor sign_mantissa,
    int64_t n_fp8_per_stream, torch::Tensor freqs)
{
    int64_t n_streams, n_enc_blocks;
    auto sfc_gpu = build_sfc_table_pair(compressed, block_offsets, final_states,
                                        sign_mantissa, n_fp8_per_stream, freqs,
                                        n_streams, n_enc_blocks);
    auto output = torch::empty(
        {n_fp8_per_stream, n_streams},
        torch::TensorOptions().dtype(torch::kUInt8).device(compressed.device()));

    const int threads = rans_gpu_pair::BLOCK_STREAMS;
    int64_t blocks = (n_enc_blocks + 1) / 2;

    rans_decode_fp8_pair_ldg_kernel<<<blocks, threads>>>(
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

torch::Tensor gpu_rans_decode_fp8_pair_ldg_dump(
    torch::Tensor compressed, torch::Tensor block_offsets,
    torch::Tensor final_states, torch::Tensor sign_mantissa,
    int64_t n_fp8_per_stream, torch::Tensor freqs)
{
    int64_t n_streams, n_enc_blocks;
    auto sfc_gpu = build_sfc_table_pair(compressed, block_offsets, final_states,
                                        sign_mantissa, n_fp8_per_stream, freqs,
                                        n_streams, n_enc_blocks);
    const int threads = rans_gpu_pair::BLOCK_STREAMS;
    int64_t blocks = (n_enc_blocks + 1) / 2;

    auto digest = torch::empty(
        {blocks * threads},
        torch::TensorOptions().dtype(torch::kUInt8).device(compressed.device()));

    rans_decode_fp8_pair_ldg_dump_kernel<<<blocks, threads>>>(
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

// ─── Quad-stream pair variant ────────────────────────────────────────
//
// Each CUDA block handles 4 encoder blocks (4 streams per thread) instead
// of 2. Doubles the ILP available to the warp scheduler for hiding L1
// and DRAM latency. The pair-ldg profile showed IPC 2.05 with both
// compute (50%) and memory (46%) unsaturated — classic latency-limited.
// More independent decode chains per thread gives the scheduler more
// instructions to issue while any one chain stalls on memory.

__global__ void rans_decode_fp8_pair_ldg_q4_kernel(
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
    int64_t tid_s[4];
    bool have[4];
    RansCtx ctx[4];

    #pragma unroll
    for (int q = 0; q < 4; q++) {
        int enc = 4 * blockIdx.x + q;
        tid_s[q] = (int64_t)enc * blockDim.x + tid;
        have[q] = tid_s[q] < n_streams;
        if (have[q]) {
            int32_t base = block_offsets[enc];
            int32_t next = block_offsets[enc + 1];
            int G = (next - base) / ((int)blockDim.x * 4);
            ctx[q].x = final_states[tid_s[q]]; ctx[q].slab_idx = G;
            ctx[q].buf_hi = 0; ctx[q].buf_lo = 0; ctx[q].buf_avail = 0;
            ctx[q].block_base_u32 = (int64_t)base / 4;
        }
    }

    if (!have[0] && !have[1] && !have[2] && !have[3]) return;

    int64_t n_pairs = n_fp8_per_stream / 2;
    for (int64_t i = 0; i < n_pairs; i++) {
        uint8_t pair[4], sm[4];
        #pragma unroll
        for (int q = 0; q < 4; q++)
            pair[q] = have[q] ? decode_step_pair_ldg(ctx[q], sfc_global,
                                                     compressed_u32, tid) : 0;
        #pragma unroll
        for (int q = 0; q < 4; q++)
            sm[q] = have[q] ? sign_mantissa[i * n_streams + tid_s[q]] : 0;

        #pragma unroll
        for (int q = 0; q < 4; q++) {
            if (have[q]) {
                uint8_t e0 = pair[q] >> 4, e1 = pair[q] & 0xF;
                uint8_t s0 = sm[q] & 0xF, s1 = sm[q] >> 4;
                output[(2*i)   * n_streams + tid_s[q]] = ((s0&8)<<4)|(e0<<3)|(s0&7);
                output[(2*i+1) * n_streams + tid_s[q]] = ((s1&8)<<4)|(e1<<3)|(s1&7);
            }
        }
    }
}

__global__ void rans_decode_fp8_pair_ldg_q4_dump_kernel(
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
    int64_t tid_s[4];
    bool have[4];
    RansCtx ctx[4];

    #pragma unroll
    for (int q = 0; q < 4; q++) {
        int enc = 4 * blockIdx.x + q;
        tid_s[q] = (int64_t)enc * blockDim.x + tid;
        have[q] = tid_s[q] < n_streams;
        if (have[q]) {
            int32_t base = block_offsets[enc];
            int32_t next = block_offsets[enc + 1];
            int G = (next - base) / ((int)blockDim.x * 4);
            ctx[q].x = final_states[tid_s[q]]; ctx[q].slab_idx = G;
            ctx[q].buf_hi = 0; ctx[q].buf_lo = 0; ctx[q].buf_avail = 0;
            ctx[q].block_base_u32 = (int64_t)base / 4;
        }
    }

    if (!have[0] && !have[1] && !have[2] && !have[3]) return;

    uint8_t digest = 0;
    int64_t n_pairs = n_fp8_per_stream / 2;
    for (int64_t i = 0; i < n_pairs; i++) {
        uint8_t pair[4], sm[4];
        #pragma unroll
        for (int q = 0; q < 4; q++)
            pair[q] = have[q] ? decode_step_pair_ldg(ctx[q], sfc_global,
                                                     compressed_u32, tid) : 0;
        #pragma unroll
        for (int q = 0; q < 4; q++)
            sm[q] = have[q] ? sign_mantissa[i * n_streams + tid_s[q]] : 0;

        #pragma unroll
        for (int q = 0; q < 4; q++) {
            if (have[q]) {
                uint8_t e0 = pair[q] >> 4, e1 = pair[q] & 0xF;
                uint8_t s0 = sm[q] & 0xF, s1 = sm[q] >> 4;
                digest ^= ((s0&8)<<4)|(e0<<3)|(s0&7);
                digest ^= ((s1&8)<<4)|(e1<<3)|(s1&7);
            }
        }
    }
    digest_out[(int64_t)blockIdx.x * blockDim.x + tid] = digest;
}

torch::Tensor gpu_rans_decode_fp8_pair_ldg_q4(
    torch::Tensor compressed, torch::Tensor block_offsets,
    torch::Tensor final_states, torch::Tensor sign_mantissa,
    int64_t n_fp8_per_stream, torch::Tensor freqs)
{
    int64_t n_streams, n_enc_blocks;
    auto sfc_gpu = build_sfc_table_pair(compressed, block_offsets, final_states,
                                        sign_mantissa, n_fp8_per_stream, freqs,
                                        n_streams, n_enc_blocks);
    auto output = torch::empty(
        {n_fp8_per_stream, n_streams},
        torch::TensorOptions().dtype(torch::kUInt8).device(compressed.device()));

    const int threads = rans_gpu_pair::BLOCK_STREAMS;
    int64_t blocks = (n_enc_blocks + 3) / 4;

    rans_decode_fp8_pair_ldg_q4_kernel<<<blocks, threads>>>(
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

torch::Tensor gpu_rans_decode_fp8_pair_ldg_q4_dump(
    torch::Tensor compressed, torch::Tensor block_offsets,
    torch::Tensor final_states, torch::Tensor sign_mantissa,
    int64_t n_fp8_per_stream, torch::Tensor freqs)
{
    int64_t n_streams, n_enc_blocks;
    auto sfc_gpu = build_sfc_table_pair(compressed, block_offsets, final_states,
                                        sign_mantissa, n_fp8_per_stream, freqs,
                                        n_streams, n_enc_blocks);
    const int threads = rans_gpu_pair::BLOCK_STREAMS;
    int64_t blocks = (n_enc_blocks + 3) / 4;

    auto digest = torch::empty(
        {blocks * threads},
        torch::TensorOptions().dtype(torch::kUInt8).device(compressed.device()));

    rans_decode_fp8_pair_ldg_q4_dump_kernel<<<blocks, threads>>>(
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

// ─── Branchless-renorm pair variants ─────────────────────────────────

// Full output (for correctness testing).
__global__ void rans_decode_fp8_pair_bl_kernel(
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
    int enc_a = 2 * blockIdx.x, enc_b = 2 * blockIdx.x + 1;
    int64_t tid_a = (int64_t)enc_a * blockDim.x + tid;
    int64_t tid_b = (int64_t)enc_b * blockDim.x + tid;
    bool have_a = tid_a < n_streams, have_b = tid_b < n_streams;
    if (!have_a && !have_b) return;

    RansCtx A, B;
    if (have_a) {
        int32_t base = block_offsets[enc_a], next = block_offsets[enc_a+1];
        int G = (next-base)/((int)blockDim.x*4);
        A.x = final_states[tid_a]; A.slab_idx = G;
        A.buf_hi=0; A.buf_lo=0; A.buf_avail=0;
        A.block_base_u32 = (int64_t)base/4;
    }
    if (have_b) {
        int32_t base = block_offsets[enc_b], next = block_offsets[enc_b+1];
        int G = (next-base)/((int)blockDim.x*4);
        B.x = final_states[tid_b]; B.slab_idx = G;
        B.buf_hi=0; B.buf_lo=0; B.buf_avail=0;
        B.block_base_u32 = (int64_t)base/4;
    }

    int64_t n_pairs = n_fp8_per_stream / 2;
    for (int64_t i = 0; i < n_pairs; i++) {
        uint8_t pa=0, pb=0;
        if (have_a) pa = decode_step_pair_ldg_branchless(A, sfc_global, compressed_u32, tid);
        if (have_b) pb = decode_step_pair_ldg_branchless(B, sfc_global, compressed_u32, tid);
        uint8_t sa=0, sb=0;
        if (have_a) sa = sign_mantissa[i * n_streams + tid_a];
        if (have_b) sb = sign_mantissa[i * n_streams + tid_b];
        if (have_a) {
            uint8_t e0=pa>>4, e1=pa&0xF, s0=sa&0xF, s1=sa>>4;
            output[(2*i)*n_streams+tid_a] = ((s0&8)<<4)|(e0<<3)|(s0&7);
            output[(2*i+1)*n_streams+tid_a] = ((s1&8)<<4)|(e1<<3)|(s1&7);
        }
        if (have_b) {
            uint8_t e0=pb>>4, e1=pb&0xF, s0=sb&0xF, s1=sb>>4;
            output[(2*i)*n_streams+tid_b] = ((s0&8)<<4)|(e0<<3)|(s0&7);
            output[(2*i+1)*n_streams+tid_b] = ((s1&8)<<4)|(e1<<3)|(s1&7);
        }
    }
}

// Dual-stream dump.
__global__ void rans_decode_fp8_pair_bl_dump_kernel(
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
    int enc_a = 2 * blockIdx.x, enc_b = 2 * blockIdx.x + 1;
    int64_t tid_a = (int64_t)enc_a * blockDim.x + tid;
    int64_t tid_b = (int64_t)enc_b * blockDim.x + tid;
    bool have_a = tid_a < n_streams, have_b = tid_b < n_streams;
    if (!have_a && !have_b) return;

    RansCtx A, B;
    if (have_a) {
        int32_t base = block_offsets[enc_a], next = block_offsets[enc_a+1];
        int G = (next-base)/((int)blockDim.x*4);
        A.x = final_states[tid_a]; A.slab_idx = G;
        A.buf_hi=0; A.buf_lo=0; A.buf_avail=0;
        A.block_base_u32 = (int64_t)base/4;
    }
    if (have_b) {
        int32_t base = block_offsets[enc_b], next = block_offsets[enc_b+1];
        int G = (next-base)/((int)blockDim.x*4);
        B.x = final_states[tid_b]; B.slab_idx = G;
        B.buf_hi=0; B.buf_lo=0; B.buf_avail=0;
        B.block_base_u32 = (int64_t)base/4;
    }

    uint8_t digest = 0;
    int64_t n_pairs = n_fp8_per_stream / 2;
    for (int64_t i = 0; i < n_pairs; i++) {
        uint8_t pa=0, pb=0;
        if (have_a) pa = decode_step_pair_ldg_branchless(A, sfc_global, compressed_u32, tid);
        if (have_b) pb = decode_step_pair_ldg_branchless(B, sfc_global, compressed_u32, tid);
        uint8_t sa=0, sb=0;
        if (have_a) sa = sign_mantissa[i * n_streams + tid_a];
        if (have_b) sb = sign_mantissa[i * n_streams + tid_b];
        if (have_a) {
            uint8_t e0=pa>>4, e1=pa&0xF, s0=sa&0xF, s1=sa>>4;
            digest ^= ((s0&8)<<4)|(e0<<3)|(s0&7);
            digest ^= ((s1&8)<<4)|(e1<<3)|(s1&7);
        }
        if (have_b) {
            uint8_t e0=pb>>4, e1=pb&0xF, s0=sb&0xF, s1=sb>>4;
            digest ^= ((s0&8)<<4)|(e0<<3)|(s0&7);
            digest ^= ((s1&8)<<4)|(e1<<3)|(s1&7);
        }
    }
    digest_out[(int64_t)blockIdx.x * blockDim.x + tid] = digest;
}

torch::Tensor gpu_rans_decode_fp8_pair_bl(
    torch::Tensor compressed, torch::Tensor block_offsets,
    torch::Tensor final_states, torch::Tensor sign_mantissa,
    int64_t n_fp8_per_stream, torch::Tensor freqs)
{
    int64_t n_streams, n_enc_blocks;
    auto sfc_gpu = build_sfc_table_pair(compressed, block_offsets, final_states,
                                        sign_mantissa, n_fp8_per_stream, freqs,
                                        n_streams, n_enc_blocks);
    auto output = torch::empty(
        {n_fp8_per_stream, n_streams},
        torch::TensorOptions().dtype(torch::kUInt8).device(compressed.device()));
    const int threads = rans_gpu_pair::BLOCK_STREAMS;
    int64_t blocks = (n_enc_blocks + 1) / 2;
    rans_decode_fp8_pair_bl_kernel<<<blocks, threads>>>(
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

torch::Tensor gpu_rans_decode_fp8_pair_bl_dump(
    torch::Tensor compressed, torch::Tensor block_offsets,
    torch::Tensor final_states, torch::Tensor sign_mantissa,
    int64_t n_fp8_per_stream, torch::Tensor freqs)
{
    int64_t n_streams, n_enc_blocks;
    auto sfc_gpu = build_sfc_table_pair(compressed, block_offsets, final_states,
                                        sign_mantissa, n_fp8_per_stream, freqs,
                                        n_streams, n_enc_blocks);
    const int threads = rans_gpu_pair::BLOCK_STREAMS;
    int64_t blocks = (n_enc_blocks + 1) / 2;
    auto digest = torch::empty(
        {blocks * threads},
        torch::TensorOptions().dtype(torch::kUInt8).device(compressed.device()));
    rans_decode_fp8_pair_bl_dump_kernel<<<blocks, threads>>>(
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

// ─── Joint triple decode (explicit M-entry table, uint64 entries) ────
//
// Extension of pair alphabet to triples: 16^3 = 4096 joint symbols.
// Each decode step produces 3 exponent nibbles. Table has M entries
// (M=8192 default), each a uint64 packed as:
//   bits  0..15 : symbol (0..4095, encodes 3 nibbles)
//   bits 16..31 : frequency f
//   bits 32..47 : cumulative c
//   bits 48..63 : unused
//
// One __ldg read of uint64 per step. L1 latency ~30 cycles for 3 symbols
// vs ~30 cycles for 2 symbols in pairs → 33% more output per L1 cycle.

namespace rans_gpu_triple {
constexpr uint32_t L_LOW         = 1u << 16;
constexpr int      BLOCK_STREAMS = 128;
}

// Templated on M_LOG so we can test different M values.
template <int M_LOG>
__global__ void rans_decode_fp8_triple_dump_kernel(
    const uint32_t* __restrict__ compressed_u32,
    const int32_t*  __restrict__ block_offsets,
    const uint32_t* __restrict__ final_states,
    const uint64_t* __restrict__ sfc_global,  // [M] uint64 entries
    const uint8_t*  __restrict__ sign_mantissa,
    uint8_t*        __restrict__ digest_out,
    int64_t n_fp8_per_stream,
    int64_t n_streams)
{
    constexpr uint32_t M_SIZE = 1u << M_LOG;

    int tid = threadIdx.x;
    int enc_a = 2 * blockIdx.x, enc_b = 2 * blockIdx.x + 1;
    int64_t tid_a = (int64_t)enc_a * blockDim.x + tid;
    int64_t tid_b = (int64_t)enc_b * blockDim.x + tid;
    bool have_a = tid_a < n_streams, have_b = tid_b < n_streams;
    if (!have_a && !have_b) return;

    RansCtx A, B;
    if (have_a) {
        int32_t base = block_offsets[enc_a], next = block_offsets[enc_a+1];
        int G = (next-base)/((int)blockDim.x*4);
        A.x = final_states[tid_a]; A.slab_idx = G;
        A.buf_hi=0; A.buf_lo=0; A.buf_avail=0;
        A.block_base_u32 = (int64_t)base/4;
    }
    if (have_b) {
        int32_t base = block_offsets[enc_b], next = block_offsets[enc_b+1];
        int G = (next-base)/((int)blockDim.x*4);
        B.x = final_states[tid_b]; B.slab_idx = G;
        B.buf_hi=0; B.buf_lo=0; B.buf_avail=0;
        B.block_base_u32 = (int64_t)base/4;
    }

    uint8_t digest = 0;
    int64_t n_triples = n_fp8_per_stream / 3;

    for (int64_t i = 0; i < n_triples; i++) {
        // ── Decode stream A ──
        uint16_t sym_a = 0;
        if (have_a) {
            uint32_t slot = A.x & (M_SIZE - 1);
            uint64_t entry = __ldg(&sfc_global[slot]);
            sym_a    = (uint16_t)(entry & 0xFFFF);
            uint32_t f = (uint32_t)((entry >> 16) & 0xFFFF);
            uint32_t c = (uint32_t)((entry >> 32) & 0xFFFF);
            A.x = (A.x >> M_LOG) * f + (slot - c);

            // Branchless renorm (may need 2 steps for large M_LOG)
            for (int r = 0; r < 2; r++) {
                bool need = (A.x < rans_gpu_triple::L_LOW);
                if (need && A.buf_avail == 0) {
                    A.slab_idx -= 2;
                    int64_t off = A.block_base_u32
                        + (int64_t)A.slab_idx * blockDim.x + tid;
                    A.buf_lo = compressed_u32[off];
                    A.buf_hi = compressed_u32[off + blockDim.x];
                    A.buf_avail = 4;
                }
                uint32_t x_r  = (A.x << 16) | (A.buf_hi >> 16);
                uint32_t bh_s = (A.buf_hi << 16) | (A.buf_lo >> 16);
                uint32_t bl_s = A.buf_lo << 16;
                int      av_d = A.buf_avail - 1;
                A.x         = need ? x_r  : A.x;
                A.buf_hi    = need ? bh_s : A.buf_hi;
                A.buf_lo    = need ? bl_s : A.buf_lo;
                A.buf_avail = need ? av_d : A.buf_avail;
            }
        }

        // ── Decode stream B ──
        uint16_t sym_b = 0;
        if (have_b) {
            uint32_t slot = B.x & (M_SIZE - 1);
            uint64_t entry = __ldg(&sfc_global[slot]);
            sym_b    = (uint16_t)(entry & 0xFFFF);
            uint32_t f = (uint32_t)((entry >> 16) & 0xFFFF);
            uint32_t c = (uint32_t)((entry >> 32) & 0xFFFF);
            B.x = (B.x >> M_LOG) * f + (slot - c);

            for (int r = 0; r < 2; r++) {
                bool need = (B.x < rans_gpu_triple::L_LOW);
                if (need && B.buf_avail == 0) {
                    B.slab_idx -= 2;
                    int64_t off = B.block_base_u32
                        + (int64_t)B.slab_idx * blockDim.x + tid;
                    B.buf_lo = compressed_u32[off];
                    B.buf_hi = compressed_u32[off + blockDim.x];
                    B.buf_avail = 4;
                }
                uint32_t x_r  = (B.x << 16) | (B.buf_hi >> 16);
                uint32_t bh_s = (B.buf_hi << 16) | (B.buf_lo >> 16);
                uint32_t bl_s = B.buf_lo << 16;
                int      av_d = B.buf_avail - 1;
                B.x         = need ? x_r  : B.x;
                B.buf_hi    = need ? bh_s : B.buf_hi;
                B.buf_lo    = need ? bl_s : B.buf_lo;
                B.buf_avail = need ? av_d : B.buf_avail;
            }
        }

        // ── Compose 3 fp8 bytes per stream, fold into digest ──
        // sym encodes (nib0 * 256 + nib1 * 16 + nib2).
        // sm is packed in pairs. For 3 fp8 at positions 3i, 3i+1, 3i+2:
        // sm bytes at indices 3i/2 and (3i+2)/2.
        if (have_a) {
            uint8_t n0 = (sym_a >> 8) & 0xF;
            uint8_t n1 = (sym_a >> 4) & 0xF;
            uint8_t n2 = sym_a & 0xF;
            for (int j = 0; j < 3; j++) {
                int fp8_idx = 3 * (int)i + j;
                int sm_pair = fp8_idx / 2;
                uint8_t sm_byte = sign_mantissa[sm_pair * n_streams + tid_a];
                uint8_t sm_nib = (fp8_idx & 1) ? (sm_byte >> 4) : (sm_byte & 0xF);
                uint8_t exp = (j == 0) ? n0 : (j == 1) ? n1 : n2;
                digest ^= ((sm_nib & 0x8) << 4) | (exp << 3) | (sm_nib & 0x7);
            }
        }
        if (have_b) {
            uint8_t n0 = (sym_b >> 8) & 0xF;
            uint8_t n1 = (sym_b >> 4) & 0xF;
            uint8_t n2 = sym_b & 0xF;
            for (int j = 0; j < 3; j++) {
                int fp8_idx = 3 * (int)i + j;
                int sm_pair = fp8_idx / 2;
                uint8_t sm_byte = sign_mantissa[sm_pair * n_streams + tid_b];
                uint8_t sm_nib = (fp8_idx & 1) ? (sm_byte >> 4) : (sm_byte & 0xF);
                uint8_t exp = (j == 0) ? n0 : (j == 1) ? n1 : n2;
                digest ^= ((sm_nib & 0x8) << 4) | (exp << 3) | (sm_nib & 0x7);
            }
        }
    }
    digest_out[(int64_t)blockIdx.x * blockDim.x + tid] = digest;
}

// Full-output variant for correctness.
template <int M_LOG>
__global__ void rans_decode_fp8_triple_kernel(
    const uint32_t* __restrict__ compressed_u32,
    const int32_t*  __restrict__ block_offsets,
    const uint32_t* __restrict__ final_states,
    const uint64_t* __restrict__ sfc_global,
    const uint8_t*  __restrict__ sign_mantissa,
    uint8_t*        __restrict__ output,
    int64_t n_fp8_per_stream,
    int64_t n_streams)
{
    constexpr uint32_t M_SIZE = 1u << M_LOG;

    int tid = threadIdx.x;
    int enc_a = 2 * blockIdx.x, enc_b = 2 * blockIdx.x + 1;
    int64_t tid_a = (int64_t)enc_a * blockDim.x + tid;
    int64_t tid_b = (int64_t)enc_b * blockDim.x + tid;
    bool have_a = tid_a < n_streams, have_b = tid_b < n_streams;
    if (!have_a && !have_b) return;

    RansCtx A, B;
    if (have_a) {
        int32_t base = block_offsets[enc_a], next = block_offsets[enc_a+1];
        int G = (next-base)/((int)blockDim.x*4);
        A.x = final_states[tid_a]; A.slab_idx = G;
        A.buf_hi=0; A.buf_lo=0; A.buf_avail=0;
        A.block_base_u32 = (int64_t)base/4;
    }
    if (have_b) {
        int32_t base = block_offsets[enc_b], next = block_offsets[enc_b+1];
        int G = (next-base)/((int)blockDim.x*4);
        B.x = final_states[tid_b]; B.slab_idx = G;
        B.buf_hi=0; B.buf_lo=0; B.buf_avail=0;
        B.block_base_u32 = (int64_t)base/4;
    }

    int64_t n_triples = n_fp8_per_stream / 3;
    for (int64_t i = 0; i < n_triples; i++) {
        uint16_t sym_a = 0, sym_b = 0;

        if (have_a) {
            uint32_t slot = A.x & (M_SIZE - 1);
            uint64_t entry = __ldg(&sfc_global[slot]);
            sym_a = (uint16_t)(entry & 0xFFFF);
            uint32_t f = (uint32_t)((entry >> 16) & 0xFFFF);
            uint32_t c = (uint32_t)((entry >> 32) & 0xFFFF);
            A.x = (A.x >> M_LOG) * f + (slot - c);
            for (int r = 0; r < 2; r++) {
                bool need = (A.x < rans_gpu_triple::L_LOW);
                if (need && A.buf_avail == 0) {
                    A.slab_idx -= 2;
                    int64_t off = A.block_base_u32 + (int64_t)A.slab_idx*blockDim.x+tid;
                    A.buf_lo=compressed_u32[off]; A.buf_hi=compressed_u32[off+blockDim.x];
                    A.buf_avail=4;
                }
                uint32_t xr=(A.x<<16)|(A.buf_hi>>16);
                uint32_t bh=(A.buf_hi<<16)|(A.buf_lo>>16), bl=A.buf_lo<<16;
                int av=A.buf_avail-1;
                A.x=need?xr:A.x; A.buf_hi=need?bh:A.buf_hi;
                A.buf_lo=need?bl:A.buf_lo; A.buf_avail=need?av:A.buf_avail;
            }
        }
        if (have_b) {
            uint32_t slot = B.x & (M_SIZE - 1);
            uint64_t entry = __ldg(&sfc_global[slot]);
            sym_b = (uint16_t)(entry & 0xFFFF);
            uint32_t f = (uint32_t)((entry >> 16) & 0xFFFF);
            uint32_t c = (uint32_t)((entry >> 32) & 0xFFFF);
            B.x = (B.x >> M_LOG) * f + (slot - c);
            for (int r = 0; r < 2; r++) {
                bool need = (B.x < rans_gpu_triple::L_LOW);
                if (need && B.buf_avail == 0) {
                    B.slab_idx -= 2;
                    int64_t off = B.block_base_u32 + (int64_t)B.slab_idx*blockDim.x+tid;
                    B.buf_lo=compressed_u32[off]; B.buf_hi=compressed_u32[off+blockDim.x];
                    B.buf_avail=4;
                }
                uint32_t xr=(B.x<<16)|(B.buf_hi>>16);
                uint32_t bh=(B.buf_hi<<16)|(B.buf_lo>>16), bl=B.buf_lo<<16;
                int av=B.buf_avail-1;
                B.x=need?xr:B.x; B.buf_hi=need?bh:B.buf_hi;
                B.buf_lo=need?bl:B.buf_lo; B.buf_avail=need?av:B.buf_avail;
            }
        }

        if (have_a) {
            uint8_t n0=(sym_a>>8)&0xF, n1=(sym_a>>4)&0xF, n2=sym_a&0xF;
            for (int j=0;j<3;j++) {
                int fi=3*(int)i+j, sp=fi/2;
                uint8_t sb=sign_mantissa[sp*n_streams+tid_a];
                uint8_t sn=(fi&1)?(sb>>4):(sb&0xF);
                uint8_t ex=(j==0)?n0:(j==1)?n1:n2;
                output[fi*n_streams+tid_a]=((sn&8)<<4)|(ex<<3)|(sn&7);
            }
        }
        if (have_b) {
            uint8_t n0=(sym_b>>8)&0xF, n1=(sym_b>>4)&0xF, n2=sym_b&0xF;
            for (int j=0;j<3;j++) {
                int fi=3*(int)i+j, sp=fi/2;
                uint8_t sb=sign_mantissa[sp*n_streams+tid_b];
                uint8_t sn=(fi&1)?(sb>>4):(sb&0xF);
                uint8_t ex=(j==0)?n0:(j==1)?n1:n2;
                output[fi*n_streams+tid_b]=((sn&8)<<4)|(ex<<3)|(sn&7);
            }
        }
    }
}

// Host: build uint64 sfc table for triple joint alphabet.
static torch::Tensor build_sfc_table_triple(
    torch::Tensor compressed, torch::Tensor block_offsets,
    torch::Tensor final_states, torch::Tensor sign_mantissa,
    int64_t n_fp8_per_stream, torch::Tensor freqs,
    int64_t M_total,
    int64_t& n_streams_out, int64_t& n_enc_blocks_out)
{
    TORCH_CHECK(compressed.is_cuda() && compressed.dtype() == torch::kUInt8);
    TORCH_CHECK(n_fp8_per_stream % 3 == 0);
    TORCH_CHECK(freqs.dtype() == torch::kInt32);

    int64_t n_streams = final_states.numel();
    int     n_alphabet = (int)freqs.numel();
    TORCH_CHECK(n_alphabet <= 4096, "triple alphabet max 4096 symbols");
    TORCH_CHECK(sign_mantissa.dim() == 2
                && sign_mantissa.size(0) == n_fp8_per_stream / 2
                && sign_mantissa.size(1) == n_streams);

    auto freqs_cpu = freqs.cpu().contiguous();
    const int32_t* f_ptr = freqs_cpu.data_ptr<int32_t>();

    uint32_t acc = 0;
    std::vector<uint32_t> cum_vec(n_alphabet);
    for (int i = 0; i < n_alphabet; i++) {
        cum_vec[i] = acc;
        acc += (uint32_t)f_ptr[i];
    }
    TORCH_CHECK((int64_t)acc == M_total,
                "freqs must sum to M_total=", M_total, "; got ", acc);
    TORCH_CHECK(M_total <= 65536, "M_total must fit in uint16 for packing");

    // Build uint64 sfc table: [M_total] entries.
    auto sfc_cpu = torch::zeros({M_total}, torch::kInt64);
    int64_t* sfc_ptr = sfc_cpu.data_ptr<int64_t>();
    uint32_t pos = 0;
    for (int s = 0; s < n_alphabet; s++) {
        uint32_t f = (uint32_t)f_ptr[s];
        uint32_t c = cum_vec[s];
        uint64_t entry = ((uint64_t)s & 0xFFFFu)
                       | (((uint64_t)f & 0xFFFFu) << 16)
                       | (((uint64_t)c & 0xFFFFu) << 32);
        for (uint32_t j = 0; j < f; j++)
            sfc_ptr[pos + j] = (int64_t)entry;
        pos += f;
    }

    int64_t n_enc_blocks = (n_streams + rans_gpu_triple::BLOCK_STREAMS - 1)
                           / rans_gpu_triple::BLOCK_STREAMS;
    if (n_enc_blocks < 1) n_enc_blocks = 1;
    TORCH_CHECK(block_offsets.numel() == n_enc_blocks + 1);

    n_streams_out = n_streams;
    n_enc_blocks_out = n_enc_blocks;
    return sfc_cpu.to(compressed.device());
}

torch::Tensor gpu_rans_decode_fp8_triple(
    torch::Tensor compressed, torch::Tensor block_offsets,
    torch::Tensor final_states, torch::Tensor sign_mantissa,
    int64_t n_fp8_per_stream, torch::Tensor freqs,
    int64_t M_total)
{
    int64_t n_streams, n_enc_blocks;
    auto sfc_gpu = build_sfc_table_triple(compressed, block_offsets, final_states,
                                          sign_mantissa, n_fp8_per_stream, freqs,
                                          M_total, n_streams, n_enc_blocks);
    auto output = torch::empty(
        {n_fp8_per_stream, n_streams},
        torch::TensorOptions().dtype(torch::kUInt8).device(compressed.device()));
    const int threads = rans_gpu_triple::BLOCK_STREAMS;
    int64_t blocks = (n_enc_blocks + 1) / 2;

    // Dispatch on M_LOG
    int m_log = 0;
    { int64_t tmp = M_total; while (tmp > 1) { m_log++; tmp >>= 1; } }

    #define LAUNCH_TRIPLE(ML) \
        rans_decode_fp8_triple_kernel<ML><<<blocks, threads>>>( \
            reinterpret_cast<const uint32_t*>(compressed.data_ptr<uint8_t>()), \
            block_offsets.data_ptr<int32_t>(), \
            reinterpret_cast<const uint32_t*>(final_states.data_ptr<int32_t>()), \
            reinterpret_cast<const uint64_t*>(sfc_gpu.data_ptr<int64_t>()), \
            sign_mantissa.data_ptr<uint8_t>(), \
            output.data_ptr<uint8_t>(), \
            n_fp8_per_stream, n_streams)

    switch (m_log) {
        case 12: LAUNCH_TRIPLE(12); break;
        case 13: LAUNCH_TRIPLE(13); break;
        case 14: LAUNCH_TRIPLE(14); break;
        default: TORCH_CHECK(false, "M_total must be 4096, 8192, or 16384");
    }
    #undef LAUNCH_TRIPLE
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor gpu_rans_decode_fp8_triple_dump(
    torch::Tensor compressed, torch::Tensor block_offsets,
    torch::Tensor final_states, torch::Tensor sign_mantissa,
    int64_t n_fp8_per_stream, torch::Tensor freqs,
    int64_t M_total)
{
    int64_t n_streams, n_enc_blocks;
    auto sfc_gpu = build_sfc_table_triple(compressed, block_offsets, final_states,
                                          sign_mantissa, n_fp8_per_stream, freqs,
                                          M_total, n_streams, n_enc_blocks);
    const int threads = rans_gpu_triple::BLOCK_STREAMS;
    int64_t blocks = (n_enc_blocks + 1) / 2;

    auto digest = torch::empty(
        {blocks * threads},
        torch::TensorOptions().dtype(torch::kUInt8).device(compressed.device()));

    int m_log = 0;
    { int64_t tmp = M_total; while (tmp > 1) { m_log++; tmp >>= 1; } }

    #define LAUNCH_TRIPLE_DUMP(ML) \
        rans_decode_fp8_triple_dump_kernel<ML><<<blocks, threads>>>( \
            reinterpret_cast<const uint32_t*>(compressed.data_ptr<uint8_t>()), \
            block_offsets.data_ptr<int32_t>(), \
            reinterpret_cast<const uint32_t*>(final_states.data_ptr<int32_t>()), \
            reinterpret_cast<const uint64_t*>(sfc_gpu.data_ptr<int64_t>()), \
            sign_mantissa.data_ptr<uint8_t>(), \
            digest.data_ptr<uint8_t>(), \
            n_fp8_per_stream, n_streams)

    switch (m_log) {
        case 12: LAUNCH_TRIPLE_DUMP(12); break;
        case 13: LAUNCH_TRIPLE_DUMP(13); break;
        case 14: LAUNCH_TRIPLE_DUMP(14); break;
        default: TORCH_CHECK(false, "M_total must be 4096, 8192, or 16384");
    }
    #undef LAUNCH_TRIPLE_DUMP
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return digest;
}

// ─── tANS (table ANS) decoder ───────────────────────────────────────
//
// tANS replaces rANS's multiply+divide+renorm with a single table
// lookup. Each decode step: read table[state-L] → (sym, nbBits, base),
// consume nbBits from a bit accumulator, state = base + L + bits.
// ~5 ALU + 1 L1 read per step (vs ~12 ALU + 1 read for rANS).
//
// Table: L entries × uint32. L = 2^tableLog (typically 2048 = 8 KB).
// Entry packing: sym(4) | nbBits(4) | nextBase(16) | unused(8).
// State range: [L, 2L). Table indexed by (state - L).
//
// Bit accumulator: 32-bit register, MSB-aligned. Consumed by shifting
// left. Refilled from slab uint32s when acc_bits drops below 16.

namespace rans_gpu_tans {
constexpr uint32_t L_LOW = 1u << 16;  // not used (tANS doesn't renorm like rANS)
constexpr int BLOCK_STREAMS = 128;
}

// Position-based bit reader. Matches the CPU decoder's backward reading
// exactly: bit_pos starts at (n_bits-1) and decrements. Bits are packed
// LSB-first into bytes, loaded as little-endian uint32s from slabs.
// Caches the current uint32 to avoid redundant global loads.
struct TansCtx {
    uint32_t state;
    int      bit_pos;        // current bit position (decrements)
    uint32_t cached_word;    // current uint32 from slab
    int      cached_u32_idx; // which uint32 is cached
    int      base_slab;      // first slab with data (G - G_s)
    int64_t  block_base_u32;
};

template <int TABLE_LOG>
__device__ __forceinline__ uint8_t tans_decode_step(
    TansCtx& ctx,
    const uint32_t* __restrict__ table,
    const uint32_t* __restrict__ compressed_u32,
    int tid)
{
    constexpr uint32_t L = 1u << TABLE_LOG;

    uint32_t entry = __ldg(&table[ctx.state - L]);
    uint8_t  sym   = entry & 0xFF;
    int      nb    = (entry >> 8) & 0xF;
    uint32_t base  = (entry >> 12) & 0xFFFF;

    // Read nb bits from bit_pos going DOWN, assembling LSB-first.
    // bit_pos points to the highest bit; we read (bit_pos, bit_pos-1, ...).
    // Within the bitstream, bit i is at byte i/8, position i%8 within byte.
    // In a uint32 (little-endian): bit i is at position i%32 within uint32 i/32.
    int end_pos   = ctx.bit_pos;           // highest bit to read
    int start_pos = ctx.bit_pos - nb + 1;  // lowest bit to read

    int end_u32   = end_pos / 32;
    int start_u32 = start_pos / 32;

    // Extract nb contiguous bits from the bitstream. The encoder stores
    // bits MSB-first, but the uint32 extraction pulls low→high, so the
    // raw value has reversed bit order. __brev fixes this in 1 instruction.
    uint32_t raw;
    if (start_u32 == end_u32) {
        // All bits in one uint32 (common case)
        if (end_u32 != ctx.cached_u32_idx) {
            ctx.cached_u32_idx = end_u32;
            ctx.cached_word = compressed_u32[ctx.block_base_u32
                + (int64_t)(ctx.base_slab + end_u32) * blockDim.x + tid];
        }
        int lo = start_pos & 31;
        raw = (ctx.cached_word >> lo) & ((1u << nb) - 1);
    } else {
        // Bits span two uint32s (rare: ~1 in 11 steps)
        if (end_u32 != ctx.cached_u32_idx) {
            ctx.cached_u32_idx = end_u32;
            ctx.cached_word = compressed_u32[ctx.block_base_u32
                + (int64_t)(ctx.base_slab + end_u32) * blockDim.x + tid];
        }
        int hi_bits = (end_pos & 31) + 1;
        uint32_t hi_part = ctx.cached_word & ((1u << hi_bits) - 1);

        ctx.cached_u32_idx = start_u32;
        ctx.cached_word = compressed_u32[ctx.block_base_u32
            + (int64_t)(ctx.base_slab + start_u32) * blockDim.x + tid];
        int lo_bits = nb - hi_bits;
        int lo_start = start_pos & 31;
        uint32_t lo_part = (ctx.cached_word >> lo_start) & ((1u << lo_bits) - 1);

        raw = lo_part | (hi_part << lo_bits);
    }

    // Bit-reverse: __brev reverses all 32 bits, then shift to keep nb.
    uint32_t bits = __brev(raw) >> (32 - nb);

    ctx.bit_pos -= nb;
    ctx.state = base + L + bits;
    return sym;
}

// Helper: initialize TansCtx for a stream.
__device__ __forceinline__ void tans_init_ctx(
    TansCtx& ctx,
    const int32_t* block_offsets, int enc_block,
    const uint32_t* final_states, int64_t stream_id,
    const int32_t* bit_counts,
    const uint32_t* compressed_u32, int tid)
{
    int32_t base = block_offsets[enc_block];
    int32_t next = block_offsets[enc_block + 1];
    int G = (next - base) / (128 * 4);  // BLOCK_STREAMS * 4 bytes per slab

    ctx.state = final_states[stream_id];
    ctx.block_base_u32 = (int64_t)base / 4;

    // Bitstream is right-aligned in slabs. Data starts at bit_pos = n_bits - 1.
    int n_bits = bit_counts[stream_id];
    int G_s = (n_bits + 31) / 32;  // uint32s needed for this stream
    ctx.base_slab = G - G_s;       // first slab with data
    ctx.bit_pos = n_bits - 1;      // start at the last data bit

    // Pre-cache the uint32 containing the starting bit_pos
    ctx.cached_u32_idx = ctx.bit_pos / 32;
    ctx.cached_word = compressed_u32[ctx.block_base_u32
        + (int64_t)(ctx.base_slab + ctx.cached_u32_idx) * 128 + tid];
}

// Dump kernel for benchmarking.
template <int TABLE_LOG>
__global__ void tans_decode_fp8_dump_kernel(
    const uint32_t* __restrict__ compressed_u32,
    const int32_t*  __restrict__ block_offsets,
    const uint32_t* __restrict__ final_states,
    const uint32_t* __restrict__ table,
    const uint8_t*  __restrict__ sign_mantissa,
    const int32_t*  __restrict__ bit_counts,
    uint8_t*        __restrict__ digest_out,
    int64_t n_fp8_per_stream,
    int64_t n_streams)
{
    int tid = threadIdx.x;
    int enc_a = 2 * blockIdx.x, enc_b = 2 * blockIdx.x + 1;
    int64_t tid_a = (int64_t)enc_a * blockDim.x + tid;
    int64_t tid_b = (int64_t)enc_b * blockDim.x + tid;
    bool have_a = tid_a < n_streams, have_b = tid_b < n_streams;
    if (!have_a && !have_b) return;

    TansCtx A, B;
    if (have_a) tans_init_ctx(A, block_offsets, enc_a, final_states, tid_a,
                              bit_counts, compressed_u32, tid);
    if (have_b) tans_init_ctx(B, block_offsets, enc_b, final_states, tid_b,
                              bit_counts, compressed_u32, tid);

    uint8_t digest = 0;
    for (int64_t i = 0; i < n_fp8_per_stream; i++) {
        uint8_t exp_a = 0, exp_b = 0;
        if (have_a) exp_a = tans_decode_step<TABLE_LOG>(A, table, compressed_u32, tid);
        if (have_b) exp_b = tans_decode_step<TABLE_LOG>(B, table, compressed_u32, tid);

        uint8_t sm_byte_a = 0, sm_byte_b = 0;
        if (have_a) sm_byte_a = sign_mantissa[(i >> 1) * n_streams + tid_a];
        if (have_b) sm_byte_b = sign_mantissa[(i >> 1) * n_streams + tid_b];
        uint8_t sm_nib_a = (i & 1) ? (sm_byte_a >> 4) : (sm_byte_a & 0xF);
        uint8_t sm_nib_b = (i & 1) ? (sm_byte_b >> 4) : (sm_byte_b & 0xF);

        if (have_a) digest ^= ((sm_nib_a & 8) << 4) | (exp_a << 3) | (sm_nib_a & 7);
        if (have_b) digest ^= ((sm_nib_b & 8) << 4) | (exp_b << 3) | (sm_nib_b & 7);
    }
    digest_out[(int64_t)blockIdx.x * blockDim.x + tid] = digest;
}

// Full-output kernel for correctness.
template <int TABLE_LOG>
__global__ void tans_decode_fp8_kernel(
    const uint32_t* __restrict__ compressed_u32,
    const int32_t*  __restrict__ block_offsets,
    const uint32_t* __restrict__ final_states,
    const uint32_t* __restrict__ table,
    const uint8_t*  __restrict__ sign_mantissa,
    const int32_t*  __restrict__ bit_counts,
    uint8_t*        __restrict__ output,
    int64_t n_fp8_per_stream,
    int64_t n_streams)
{
    int tid = threadIdx.x;
    int enc_a = 2 * blockIdx.x, enc_b = 2 * blockIdx.x + 1;
    int64_t tid_a = (int64_t)enc_a * blockDim.x + tid;
    int64_t tid_b = (int64_t)enc_b * blockDim.x + tid;
    bool have_a = tid_a < n_streams, have_b = tid_b < n_streams;
    if (!have_a && !have_b) return;

    TansCtx A, B;
    if (have_a) tans_init_ctx(A, block_offsets, enc_a, final_states, tid_a,
                              bit_counts, compressed_u32, tid);
    if (have_b) tans_init_ctx(B, block_offsets, enc_b, final_states, tid_b,
                              bit_counts, compressed_u32, tid);

    for (int64_t i = 0; i < n_fp8_per_stream; i++) {
        uint8_t exp_a = 0, exp_b = 0;
        if (have_a) exp_a = tans_decode_step<TABLE_LOG>(A, table, compressed_u32, tid);
        if (have_b) exp_b = tans_decode_step<TABLE_LOG>(B, table, compressed_u32, tid);

        uint8_t sm_byte_a = 0, sm_byte_b = 0;
        if (have_a) sm_byte_a = sign_mantissa[(i >> 1) * n_streams + tid_a];
        if (have_b) sm_byte_b = sign_mantissa[(i >> 1) * n_streams + tid_b];
        uint8_t sm_nib_a = (i & 1) ? (sm_byte_a >> 4) : (sm_byte_a & 0xF);
        uint8_t sm_nib_b = (i & 1) ? (sm_byte_b >> 4) : (sm_byte_b & 0xF);

        if (have_a)
            output[i * n_streams + tid_a] = ((sm_nib_a & 8) << 4) | (exp_a << 3) | (sm_nib_a & 7);
        if (have_b)
            output[i * n_streams + tid_b] = ((sm_nib_b & 8) << 4) | (exp_b << 3) | (sm_nib_b & 7);
    }
}

// Host launchers
torch::Tensor gpu_tans_decode_fp8(
    torch::Tensor compressed, torch::Tensor block_offsets,
    torch::Tensor final_states, torch::Tensor sign_mantissa,
    torch::Tensor decode_table, torch::Tensor bit_counts,
    int64_t n_fp8_per_stream, int64_t table_log)
{
    int64_t n_streams = final_states.numel();
    int64_t n_enc_blocks = (n_streams + rans_gpu_tans::BLOCK_STREAMS - 1)
                           / rans_gpu_tans::BLOCK_STREAMS;
    auto output = torch::empty(
        {n_fp8_per_stream, n_streams},
        torch::TensorOptions().dtype(torch::kUInt8).device(compressed.device()));
    const int threads = rans_gpu_tans::BLOCK_STREAMS;
    int64_t blocks = (n_enc_blocks + 1) / 2;

    #define LAUNCH_TANS(TL) \
        tans_decode_fp8_kernel<TL><<<blocks, threads>>>( \
            reinterpret_cast<const uint32_t*>(compressed.data_ptr<uint8_t>()), \
            block_offsets.data_ptr<int32_t>(), \
            reinterpret_cast<const uint32_t*>(final_states.data_ptr<int32_t>()), \
            reinterpret_cast<const uint32_t*>(decode_table.data_ptr<int32_t>()), \
            sign_mantissa.data_ptr<uint8_t>(), \
            bit_counts.data_ptr<int32_t>(), \
            output.data_ptr<uint8_t>(), \
            n_fp8_per_stream, n_streams)
    switch (table_log) {
        case 10: LAUNCH_TANS(10); break;
        case 11: LAUNCH_TANS(11); break;
        case 12: LAUNCH_TANS(12); break;
        default: TORCH_CHECK(false, "table_log must be 10, 11, or 12");
    }
    #undef LAUNCH_TANS
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor gpu_tans_decode_fp8_dump(
    torch::Tensor compressed, torch::Tensor block_offsets,
    torch::Tensor final_states, torch::Tensor sign_mantissa,
    torch::Tensor decode_table, torch::Tensor bit_counts,
    int64_t n_fp8_per_stream, int64_t table_log)
{
    int64_t n_streams = final_states.numel();
    int64_t n_enc_blocks = (n_streams + rans_gpu_tans::BLOCK_STREAMS - 1)
                           / rans_gpu_tans::BLOCK_STREAMS;
    const int threads = rans_gpu_tans::BLOCK_STREAMS;
    int64_t blocks = (n_enc_blocks + 1) / 2;
    auto digest = torch::empty(
        {blocks * threads},
        torch::TensorOptions().dtype(torch::kUInt8).device(compressed.device()));

    #define LAUNCH_TANS_DUMP(TL) \
        tans_decode_fp8_dump_kernel<TL><<<blocks, threads>>>( \
            reinterpret_cast<const uint32_t*>(compressed.data_ptr<uint8_t>()), \
            block_offsets.data_ptr<int32_t>(), \
            reinterpret_cast<const uint32_t*>(final_states.data_ptr<int32_t>()), \
            reinterpret_cast<const uint32_t*>(decode_table.data_ptr<int32_t>()), \
            sign_mantissa.data_ptr<uint8_t>(), \
            bit_counts.data_ptr<int32_t>(), \
            digest.data_ptr<uint8_t>(), \
            n_fp8_per_stream, n_streams)
    switch (table_log) {
        case 10: LAUNCH_TANS_DUMP(10); break;
        case 11: LAUNCH_TANS_DUMP(11); break;
        case 12: LAUNCH_TANS_DUMP(12); break;
        default: TORCH_CHECK(false, "table_log must be 10, 11, or 12");
    }
    #undef LAUNCH_TANS_DUMP
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return digest;
}

// ─── Pair-tANS dump kernel ──────────────────────────────────────────
// Same as tans_decode_fp8_dump_kernel but processes pair symbols:
// each decode step produces 2 exponent nibbles (1 sm byte consumed).
// Inner loop runs n_fp8_per_stream/2 times.

template <int TABLE_LOG>
__global__ void tans_decode_fp8_pair_dump_kernel(
    const uint32_t* __restrict__ compressed_u32,
    const int32_t*  __restrict__ block_offsets,
    const uint32_t* __restrict__ final_states,
    const uint32_t* __restrict__ table,
    const uint8_t*  __restrict__ sign_mantissa,
    const int32_t*  __restrict__ bit_counts,
    uint8_t*        __restrict__ digest_out,
    int64_t n_fp8_per_stream,
    int64_t n_streams)
{
    int tid = threadIdx.x;
    int enc_a = 2 * blockIdx.x, enc_b = 2 * blockIdx.x + 1;
    int64_t tid_a = (int64_t)enc_a * blockDim.x + tid;
    int64_t tid_b = (int64_t)enc_b * blockDim.x + tid;
    bool have_a = tid_a < n_streams, have_b = tid_b < n_streams;
    if (!have_a && !have_b) return;

    TansCtx A, B;
    if (have_a) tans_init_ctx(A, block_offsets, enc_a, final_states, tid_a,
                              bit_counts, compressed_u32, tid);
    if (have_b) tans_init_ctx(B, block_offsets, enc_b, final_states, tid_b,
                              bit_counts, compressed_u32, tid);

    uint8_t digest = 0;
    int64_t n_pairs = n_fp8_per_stream / 2;
    for (int64_t i = 0; i < n_pairs; i++) {
        uint8_t pair_a = 0, pair_b = 0;
        if (have_a) pair_a = tans_decode_step<TABLE_LOG>(A, table, compressed_u32, tid);
        if (have_b) pair_b = tans_decode_step<TABLE_LOG>(B, table, compressed_u32, tid);

        uint8_t sm_a = 0, sm_b = 0;
        if (have_a) sm_a = sign_mantissa[i * n_streams + tid_a];
        if (have_b) sm_b = sign_mantissa[i * n_streams + tid_b];

        if (have_a) {
            uint8_t e0 = pair_a >> 4, e1 = pair_a & 0xF;
            uint8_t s0 = sm_a & 0xF, s1 = sm_a >> 4;
            digest ^= ((s0 & 8) << 4) | (e0 << 3) | (s0 & 7);
            digest ^= ((s1 & 8) << 4) | (e1 << 3) | (s1 & 7);
        }
        if (have_b) {
            uint8_t e0 = pair_b >> 4, e1 = pair_b & 0xF;
            uint8_t s0 = sm_b & 0xF, s1 = sm_b >> 4;
            digest ^= ((s0 & 8) << 4) | (e0 << 3) | (s0 & 7);
            digest ^= ((s1 & 8) << 4) | (e1 << 3) | (s1 & 7);
        }
    }
    digest_out[(int64_t)blockIdx.x * blockDim.x + tid] = digest;
}

// Full output for correctness.
template <int TABLE_LOG>
__global__ void tans_decode_fp8_pair_kernel(
    const uint32_t* __restrict__ compressed_u32,
    const int32_t*  __restrict__ block_offsets,
    const uint32_t* __restrict__ final_states,
    const uint32_t* __restrict__ table,
    const uint8_t*  __restrict__ sign_mantissa,
    const int32_t*  __restrict__ bit_counts,
    uint8_t*        __restrict__ output,
    int64_t n_fp8_per_stream,
    int64_t n_streams)
{
    int tid = threadIdx.x;
    int enc_a = 2 * blockIdx.x, enc_b = 2 * blockIdx.x + 1;
    int64_t tid_a = (int64_t)enc_a * blockDim.x + tid;
    int64_t tid_b = (int64_t)enc_b * blockDim.x + tid;
    bool have_a = tid_a < n_streams, have_b = tid_b < n_streams;
    if (!have_a && !have_b) return;

    TansCtx A, B;
    if (have_a) tans_init_ctx(A, block_offsets, enc_a, final_states, tid_a,
                              bit_counts, compressed_u32, tid);
    if (have_b) tans_init_ctx(B, block_offsets, enc_b, final_states, tid_b,
                              bit_counts, compressed_u32, tid);

    int64_t n_pairs = n_fp8_per_stream / 2;
    for (int64_t i = 0; i < n_pairs; i++) {
        uint8_t pair_a = 0, pair_b = 0;
        if (have_a) pair_a = tans_decode_step<TABLE_LOG>(A, table, compressed_u32, tid);
        if (have_b) pair_b = tans_decode_step<TABLE_LOG>(B, table, compressed_u32, tid);

        uint8_t sm_a = 0, sm_b = 0;
        if (have_a) sm_a = sign_mantissa[i * n_streams + tid_a];
        if (have_b) sm_b = sign_mantissa[i * n_streams + tid_b];

        if (have_a) {
            uint8_t e0 = pair_a >> 4, e1 = pair_a & 0xF;
            uint8_t s0 = sm_a & 0xF, s1 = sm_a >> 4;
            output[(2*i)   * n_streams + tid_a] = ((s0&8)<<4)|(e0<<3)|(s0&7);
            output[(2*i+1) * n_streams + tid_a] = ((s1&8)<<4)|(e1<<3)|(s1&7);
        }
        if (have_b) {
            uint8_t e0 = pair_b >> 4, e1 = pair_b & 0xF;
            uint8_t s0 = sm_b & 0xF, s1 = sm_b >> 4;
            output[(2*i)   * n_streams + tid_b] = ((s0&8)<<4)|(e0<<3)|(s0&7);
            output[(2*i+1) * n_streams + tid_b] = ((s1&8)<<4)|(e1<<3)|(s1&7);
        }
    }
}

// Host launchers for pair-tANS
torch::Tensor gpu_tans_decode_fp8_pair(
    torch::Tensor compressed, torch::Tensor block_offsets,
    torch::Tensor final_states, torch::Tensor sign_mantissa,
    torch::Tensor decode_table, torch::Tensor bit_counts,
    int64_t n_fp8_per_stream, int64_t table_log)
{
    int64_t n_streams = final_states.numel();
    int64_t n_enc_blocks = (n_streams + 128 - 1) / 128;
    auto output = torch::empty(
        {n_fp8_per_stream, n_streams},
        torch::TensorOptions().dtype(torch::kUInt8).device(compressed.device()));
    int64_t blocks = (n_enc_blocks + 1) / 2;

    #define LAUNCH_TANS_PAIR(TL) \
        tans_decode_fp8_pair_kernel<TL><<<blocks, 128>>>( \
            reinterpret_cast<const uint32_t*>(compressed.data_ptr<uint8_t>()), \
            block_offsets.data_ptr<int32_t>(), \
            reinterpret_cast<const uint32_t*>(final_states.data_ptr<int32_t>()), \
            reinterpret_cast<const uint32_t*>(decode_table.data_ptr<int32_t>()), \
            sign_mantissa.data_ptr<uint8_t>(), \
            bit_counts.data_ptr<int32_t>(), \
            output.data_ptr<uint8_t>(), \
            n_fp8_per_stream, n_streams)
    switch (table_log) {
        case 11: LAUNCH_TANS_PAIR(11); break;
        case 12: LAUNCH_TANS_PAIR(12); break;
        default: TORCH_CHECK(false, "table_log must be 11 or 12 for pairs");
    }
    #undef LAUNCH_TANS_PAIR
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor gpu_tans_decode_fp8_pair_dump(
    torch::Tensor compressed, torch::Tensor block_offsets,
    torch::Tensor final_states, torch::Tensor sign_mantissa,
    torch::Tensor decode_table, torch::Tensor bit_counts,
    int64_t n_fp8_per_stream, int64_t table_log)
{
    int64_t n_streams = final_states.numel();
    int64_t n_enc_blocks = (n_streams + 128 - 1) / 128;
    int64_t blocks = (n_enc_blocks + 1) / 2;
    auto digest = torch::empty(
        {blocks * 128},
        torch::TensorOptions().dtype(torch::kUInt8).device(compressed.device()));

    #define LAUNCH_TANS_PAIR_DUMP(TL) \
        tans_decode_fp8_pair_dump_kernel<TL><<<blocks, 128>>>( \
            reinterpret_cast<const uint32_t*>(compressed.data_ptr<uint8_t>()), \
            block_offsets.data_ptr<int32_t>(), \
            reinterpret_cast<const uint32_t*>(final_states.data_ptr<int32_t>()), \
            reinterpret_cast<const uint32_t*>(decode_table.data_ptr<int32_t>()), \
            sign_mantissa.data_ptr<uint8_t>(), \
            bit_counts.data_ptr<int32_t>(), \
            digest.data_ptr<uint8_t>(), \
            n_fp8_per_stream, n_streams)
    switch (table_log) {
        case 11: LAUNCH_TANS_PAIR_DUMP(11); break;
        case 12: LAUNCH_TANS_PAIR_DUMP(12); break;
        default: TORCH_CHECK(false, "table_log must be 11 or 12 for pairs");
    }
    #undef LAUNCH_TANS_PAIR_DUMP
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return digest;
}

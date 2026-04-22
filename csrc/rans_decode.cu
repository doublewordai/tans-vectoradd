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

// ─── Factored triple decode (M=1024, n=3, M_total=2^30) ─────────────
//
// Exploits the factored joint distribution: p(a,b,c) = p(a)×p(b)×p(c).
// Instead of one lookup in a huge M^3-entry table, decomposes the slot
// into 3 base-M digits and does 3 PARALLEL lookups in the same 1024-entry
// marginal table. The L1 pipeline sees 3 independent reads per step.
//
// Each decode step produces 3 exponent nibbles = 3 fp8 bytes.
// f_total = f0 × f1 × f2 (exact, no requantization).
// c_total = c0 × M^2 + c1 × M + c2 (Horner's with shifts since M=2^10).

namespace rans_gpu_triple {
constexpr int      M_LOG     = 10;
constexpr uint32_t M_SIZE    = 1u << M_LOG;        // 1024
constexpr int      N_LOG     = 3 * M_LOG;           // 30 bits for slot
constexpr uint32_t L_LOW     = 1u << 16;
constexpr int      BLOCK_STREAMS = 128;
}

__global__ void rans_decode_fp8_triple_dump_kernel(
    const uint32_t* __restrict__ compressed_u32,
    const int32_t*  __restrict__ block_offsets,
    const uint32_t* __restrict__ final_states,
    const uint32_t* __restrict__ sfc_global,  // [1024] marginal table
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
    int64_t n_triples = n_fp8_per_stream / 3;

    for (int64_t i = 0; i < n_triples; i++) {
        // ── Decode stream A ──
        uint8_t sym_a[3] = {0,0,0};
        if (have_a) {
            uint32_t slot = A.x & ((1u << rans_gpu_triple::N_LOG) - 1);
            // Decompose into 3 base-1024 digits (shifts + masks)
            uint32_t s0 = (slot >> 20) & (rans_gpu_triple::M_SIZE - 1);
            uint32_t s1 = (slot >> 10) & (rans_gpu_triple::M_SIZE - 1);
            uint32_t s2 = slot & (rans_gpu_triple::M_SIZE - 1);

            // 3 parallel L1 reads from the same 4KB table
            uint32_t e0 = __ldg(&sfc_global[s0]);
            uint32_t e1 = __ldg(&sfc_global[s1]);
            uint32_t e2 = __ldg(&sfc_global[s2]);

            sym_a[0] = e0 & 0xFF;
            sym_a[1] = e1 & 0xFF;
            sym_a[2] = e2 & 0xFF;
            uint32_t f0 = (e0 >> 8) & 0x3FF;  // 10-bit freq for M=1024
            uint32_t f1 = (e1 >> 8) & 0x3FF;
            uint32_t f2 = (e2 >> 8) & 0x3FF;
            uint32_t c0 = (e0 >> 18) & 0x3FF;  // 10-bit cum
            uint32_t c1 = (e1 >> 18) & 0x3FF;
            uint32_t c2 = (e2 >> 18) & 0x3FF;

            uint32_t f_total = f0 * f1 * f2;
            uint32_t c_total = (c0 << 20) | (c1 << 10) | c2;

            A.x = (A.x >> rans_gpu_triple::N_LOG) * f_total + (slot - c_total);

            // Branchless renorm
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

            // Might need a second renorm (more likely with 30-bit M_total)
            need = (A.x < rans_gpu_triple::L_LOW);
            if (need && A.buf_avail == 0) {
                A.slab_idx -= 2;
                int64_t off = A.block_base_u32
                    + (int64_t)A.slab_idx * blockDim.x + tid;
                A.buf_lo = compressed_u32[off];
                A.buf_hi = compressed_u32[off + blockDim.x];
                A.buf_avail = 4;
            }
            x_r  = (A.x << 16) | (A.buf_hi >> 16);
            bh_s = (A.buf_hi << 16) | (A.buf_lo >> 16);
            bl_s = A.buf_lo << 16;
            av_d = A.buf_avail - 1;
            A.x         = need ? x_r  : A.x;
            A.buf_hi    = need ? bh_s : A.buf_hi;
            A.buf_lo    = need ? bl_s : A.buf_lo;
            A.buf_avail = need ? av_d : A.buf_avail;
        }

        // ── Decode stream B ──
        uint8_t sym_b[3] = {0,0,0};
        if (have_b) {
            uint32_t slot = B.x & ((1u << rans_gpu_triple::N_LOG) - 1);
            uint32_t s0 = (slot >> 20) & (rans_gpu_triple::M_SIZE - 1);
            uint32_t s1 = (slot >> 10) & (rans_gpu_triple::M_SIZE - 1);
            uint32_t s2 = slot & (rans_gpu_triple::M_SIZE - 1);

            uint32_t e0 = __ldg(&sfc_global[s0]);
            uint32_t e1 = __ldg(&sfc_global[s1]);
            uint32_t e2 = __ldg(&sfc_global[s2]);

            sym_b[0] = e0 & 0xFF;
            sym_b[1] = e1 & 0xFF;
            sym_b[2] = e2 & 0xFF;
            uint32_t f0 = (e0 >> 8) & 0x3FF;
            uint32_t f1 = (e1 >> 8) & 0x3FF;
            uint32_t f2 = (e2 >> 8) & 0x3FF;
            uint32_t c0 = (e0 >> 18) & 0x3FF;
            uint32_t c1 = (e1 >> 18) & 0x3FF;
            uint32_t c2 = (e2 >> 18) & 0x3FF;

            uint32_t f_total = f0 * f1 * f2;
            uint32_t c_total = (c0 << 20) | (c1 << 10) | c2;

            B.x = (B.x >> rans_gpu_triple::N_LOG) * f_total + (slot - c_total);

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

            need = (B.x < rans_gpu_triple::L_LOW);
            if (need && B.buf_avail == 0) {
                B.slab_idx -= 2;
                int64_t off = B.block_base_u32
                    + (int64_t)B.slab_idx * blockDim.x + tid;
                B.buf_lo = compressed_u32[off];
                B.buf_hi = compressed_u32[off + blockDim.x];
                B.buf_avail = 4;
            }
            x_r  = (B.x << 16) | (B.buf_hi >> 16);
            bh_s = (B.buf_hi << 16) | (B.buf_lo >> 16);
            bl_s = B.buf_lo << 16;
            av_d = B.buf_avail - 1;
            B.x         = need ? x_r  : B.x;
            B.buf_hi    = need ? bh_s : B.buf_hi;
            B.buf_lo    = need ? bl_s : B.buf_lo;
            B.buf_avail = need ? av_d : B.buf_avail;
        }

        // ── Compose 3 fp8 bytes per stream and fold into digest ──
        // sm is packed in pairs (2 nibbles/byte). For 3 fp8 bytes at
        // positions 3i, 3i+1, 3i+2: sm byte at 3i/2 and (3i+2)/2.
        // Since N%6==0, the alignment works out every 2 triple steps.
        if (have_a) {
            for (int j = 0; j < 3; j++) {
                int fp8_idx = 3 * (int)i + j;
                int sm_pair = fp8_idx / 2;
                uint8_t sm_byte = sign_mantissa[sm_pair * n_streams + tid_a];
                uint8_t sm_nib = (fp8_idx & 1) ? (sm_byte >> 4) : (sm_byte & 0xF);
                digest ^= ((sm_nib & 0x8) << 4) | (sym_a[j] << 3) | (sm_nib & 0x7);
            }
        }
        if (have_b) {
            for (int j = 0; j < 3; j++) {
                int fp8_idx = 3 * (int)i + j;
                int sm_pair = fp8_idx / 2;
                uint8_t sm_byte = sign_mantissa[sm_pair * n_streams + tid_b];
                uint8_t sm_nib = (fp8_idx & 1) ? (sm_byte >> 4) : (sm_byte & 0xF);
                digest ^= ((sm_nib & 0x8) << 4) | (sym_b[j] << 3) | (sm_nib & 0x7);
            }
        }
    }
    digest_out[(int64_t)blockIdx.x * blockDim.x + tid] = digest;
}

// Full-output variant for correctness testing.
__global__ void rans_decode_fp8_triple_kernel(
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

    int64_t n_triples = n_fp8_per_stream / 3;
    for (int64_t i = 0; i < n_triples; i++) {
        // Decode A
        uint8_t sym_a[3] = {0,0,0};
        if (have_a) {
            uint32_t slot = A.x & ((1u << rans_gpu_triple::N_LOG) - 1);
            uint32_t s0 = (slot >> 20) & 0x3FF, s1 = (slot >> 10) & 0x3FF, s2 = slot & 0x3FF;
            uint32_t e0 = __ldg(&sfc_global[s0]), e1 = __ldg(&sfc_global[s1]), e2 = __ldg(&sfc_global[s2]);
            sym_a[0] = e0 & 0xFF; sym_a[1] = e1 & 0xFF; sym_a[2] = e2 & 0xFF;
            uint32_t f0=(e0>>8)&0x3FF, f1=(e1>>8)&0x3FF, f2=(e2>>8)&0x3FF;
            uint32_t c0=(e0>>18)&0x3FF, c1=(e1>>18)&0x3FF, c2=(e2>>18)&0x3FF;
            uint32_t f_total = f0*f1*f2, c_total = (c0<<20)|(c1<<10)|c2;
            A.x = (A.x >> 30) * f_total + (slot - c_total);

            // Two branchless renorm steps
            for (int r = 0; r < 2; r++) {
                bool need = (A.x < rans_gpu_triple::L_LOW);
                if (need && A.buf_avail == 0) {
                    A.slab_idx -= 2;
                    int64_t off = A.block_base_u32 + (int64_t)A.slab_idx * blockDim.x + tid;
                    A.buf_lo = compressed_u32[off]; A.buf_hi = compressed_u32[off + blockDim.x];
                    A.buf_avail = 4;
                }
                uint32_t x_r = (A.x<<16)|(A.buf_hi>>16);
                uint32_t bh = (A.buf_hi<<16)|(A.buf_lo>>16);
                uint32_t bl = A.buf_lo<<16;
                int av = A.buf_avail-1;
                A.x = need?x_r:A.x; A.buf_hi = need?bh:A.buf_hi;
                A.buf_lo = need?bl:A.buf_lo; A.buf_avail = need?av:A.buf_avail;
            }
        }
        // Decode B
        uint8_t sym_b[3] = {0,0,0};
        if (have_b) {
            uint32_t slot = B.x & ((1u << rans_gpu_triple::N_LOG) - 1);
            uint32_t s0 = (slot >> 20) & 0x3FF, s1 = (slot >> 10) & 0x3FF, s2 = slot & 0x3FF;
            uint32_t e0 = __ldg(&sfc_global[s0]), e1 = __ldg(&sfc_global[s1]), e2 = __ldg(&sfc_global[s2]);
            sym_b[0] = e0 & 0xFF; sym_b[1] = e1 & 0xFF; sym_b[2] = e2 & 0xFF;
            uint32_t f0=(e0>>8)&0x3FF, f1=(e1>>8)&0x3FF, f2=(e2>>8)&0x3FF;
            uint32_t c0=(e0>>18)&0x3FF, c1=(e1>>18)&0x3FF, c2=(e2>>18)&0x3FF;
            uint32_t f_total = f0*f1*f2, c_total = (c0<<20)|(c1<<10)|c2;
            B.x = (B.x >> 30) * f_total + (slot - c_total);
            for (int r = 0; r < 2; r++) {
                bool need = (B.x < rans_gpu_triple::L_LOW);
                if (need && B.buf_avail == 0) {
                    B.slab_idx -= 2;
                    int64_t off = B.block_base_u32 + (int64_t)B.slab_idx * blockDim.x + tid;
                    B.buf_lo = compressed_u32[off]; B.buf_hi = compressed_u32[off + blockDim.x];
                    B.buf_avail = 4;
                }
                uint32_t x_r = (B.x<<16)|(B.buf_hi>>16);
                uint32_t bh = (B.buf_hi<<16)|(B.buf_lo>>16);
                uint32_t bl = B.buf_lo<<16;
                int av = B.buf_avail-1;
                B.x = need?x_r:B.x; B.buf_hi = need?bh:B.buf_hi;
                B.buf_lo = need?bl:B.buf_lo; B.buf_avail = need?av:B.buf_avail;
            }
        }
        // Write 3 fp8 bytes per stream
        if (have_a) {
            for (int j = 0; j < 3; j++) {
                int fp8_idx = 3*(int)i + j;
                int sm_pair = fp8_idx / 2;
                uint8_t sm_byte = sign_mantissa[sm_pair * n_streams + tid_a];
                uint8_t sm_nib = (fp8_idx & 1) ? (sm_byte >> 4) : (sm_byte & 0xF);
                output[fp8_idx * n_streams + tid_a] = ((sm_nib&8)<<4)|(sym_a[j]<<3)|(sm_nib&7);
            }
        }
        if (have_b) {
            for (int j = 0; j < 3; j++) {
                int fp8_idx = 3*(int)i + j;
                int sm_pair = fp8_idx / 2;
                uint8_t sm_byte = sign_mantissa[sm_pair * n_streams + tid_b];
                uint8_t sm_nib = (fp8_idx & 1) ? (sm_byte >> 4) : (sm_byte & 0xF);
                output[fp8_idx * n_streams + tid_b] = ((sm_nib&8)<<4)|(sym_b[j]<<3)|(sm_nib&7);
            }
        }
    }
}

// Host: build 1024-entry sfc table for M=1024 marginal.
// Packing: sym(8) | f(10) | c(10) = 28 bits in uint32.
// (Shifted to: sym bits [0:7], f bits [8:17], c bits [18:27])
static torch::Tensor build_sfc_table_triple(
    torch::Tensor freqs,
    int64_t& n_streams_out,
    int64_t& n_enc_blocks_out,
    torch::Tensor compressed,
    torch::Tensor block_offsets,
    torch::Tensor final_states,
    torch::Tensor sign_mantissa,
    int64_t n_fp8_per_stream)
{
    TORCH_CHECK(compressed.is_cuda() && compressed.dtype() == torch::kUInt8);
    TORCH_CHECK(n_fp8_per_stream % 3 == 0);
    TORCH_CHECK(compressed.numel() % 4 == 0);
    TORCH_CHECK(freqs.dtype() == torch::kInt32);

    int64_t n_streams = final_states.numel();
    int n_alphabet = (int)freqs.numel();

    // freqs are the TRIPLE joint freqs (4096 entries summing to M^3).
    // But our sfc table is the MARGINAL (16 entries summing to M=1024).
    // We need the marginal freqs. Extract from the first 16 entries of
    // the joint by summing: marginal_f[s] = sum_{a,b} joint_f[s*256 + a*16 + b]
    // = f[s] * M^2 (since joint = product). So marginal_f[s] = joint_f[s*256] / (f[0]*f[1]*...) ...
    // Actually easier: the caller should pass the marginal freqs separately.
    // For now, reconstruct: marginal[s] = round(cbrt(sum of joint_f where s is first digit))
    // This is fragile. Let's just accept marginal freqs directly.
    TORCH_CHECK(n_alphabet == 16, "triple decoder needs 16-entry marginal freqs");

    auto freqs_cpu = freqs.cpu().contiguous();
    const int32_t* f_ptr = freqs_cpu.data_ptr<int32_t>();
    uint32_t acc = 0;
    for (int i = 0; i < 16; i++) acc += (uint32_t)f_ptr[i];
    TORCH_CHECK(acc == rans_gpu_triple::M_SIZE,
                "marginal freqs must sum to ", rans_gpu_triple::M_SIZE);

    auto sfc_cpu = torch::zeros({(int64_t)rans_gpu_triple::M_SIZE}, torch::kInt32);
    auto* sfc_ptr = reinterpret_cast<uint32_t*>(sfc_cpu.data_ptr<int32_t>());
    uint32_t pos = 0;
    std::vector<uint32_t> cum_vec(16);
    uint32_t cum_acc = 0;
    for (int s = 0; s < 16; s++) {
        cum_vec[s] = cum_acc;
        uint32_t f = (uint32_t)f_ptr[s];
        uint32_t c = cum_acc;
        uint32_t entry = ((uint32_t)s & 0xFFu)
                       | ((f & 0x3FFu) << 8)
                       | ((c & 0x3FFu) << 18);
        for (uint32_t j = 0; j < f; j++) sfc_ptr[pos + j] = entry;
        pos += f;
        cum_acc += f;
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
    int64_t n_fp8_per_stream, torch::Tensor marginal_freqs)
{
    int64_t n_streams, n_enc_blocks;
    auto sfc_gpu = build_sfc_table_triple(marginal_freqs, n_streams, n_enc_blocks,
                                          compressed, block_offsets, final_states,
                                          sign_mantissa, n_fp8_per_stream);
    auto output = torch::empty(
        {n_fp8_per_stream, n_streams},
        torch::TensorOptions().dtype(torch::kUInt8).device(compressed.device()));
    const int threads = rans_gpu_triple::BLOCK_STREAMS;
    int64_t blocks = (n_enc_blocks + 1) / 2;
    rans_decode_fp8_triple_kernel<<<blocks, threads>>>(
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

torch::Tensor gpu_rans_decode_fp8_triple_dump(
    torch::Tensor compressed, torch::Tensor block_offsets,
    torch::Tensor final_states, torch::Tensor sign_mantissa,
    int64_t n_fp8_per_stream, torch::Tensor marginal_freqs)
{
    int64_t n_streams, n_enc_blocks;
    auto sfc_gpu = build_sfc_table_triple(marginal_freqs, n_streams, n_enc_blocks,
                                          compressed, block_offsets, final_states,
                                          sign_mantissa, n_fp8_per_stream);
    const int threads = rans_gpu_triple::BLOCK_STREAMS;
    int64_t blocks = (n_enc_blocks + 1) / 2;
    auto digest = torch::empty(
        {blocks * threads},
        torch::TensorOptions().dtype(torch::kUInt8).device(compressed.device()));
    rans_decode_fp8_triple_dump_kernel<<<blocks, threads>>>(
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

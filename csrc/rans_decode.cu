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

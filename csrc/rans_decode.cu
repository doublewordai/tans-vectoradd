// GPU rANS decoder for FP8 E4M3 — pair alphabet (256 symbols, M=4096).
//
// Decodes pairs of exponent nibbles per rANS step via __ldg through L1.
// Dual-stream interleaved: issues both streams' __ldg reads before
// consuming either, overlapping L1 latency across streams.
// Branchless renorm: predicated state/buffer update, zero warp divergence.
//
// Layout: each encoder block owns 128 streams. Compressed data is stored
// in slabs (128 * 4 = 512 bytes each), right-aligned per stream. The
// decoder reads slabs from the end, consuming uint16 chunks from a
// 4-chunk (8-byte) buffer.

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

// ─── Per-stream decode context ───────────────────────────────────────

struct RansCtx {
    uint32_t x;
    int      slab_idx;
    uint32_t buf_hi;
    uint32_t buf_lo;
    int      buf_avail;
    int64_t  block_base_u32;
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

// ─── Interleaved dual-stream kernels ─────────────────────────────────

// Helper: inline decode + branchless renorm for one stream.
// NOT a separate function — the interleaving requires issuing __ldg
// across streams before consuming, so the caller manages the phases.

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
        // Phase A: issue both __ldg reads
        uint32_t slot_a=0, slot_b=0, entry_a=0, entry_b=0;
        if (have_a) { slot_a = A.x & (M_SIZE-1); entry_a = __ldg(&sfc_global[slot_a]); }
        if (have_b) { slot_b = B.x & (M_SIZE-1); entry_b = __ldg(&sfc_global[slot_b]); }

        // Phase B: consume A
        uint8_t sym_a = 0;
        if (have_a) {
            sym_a = (uint8_t)(entry_a & 0xFF);
            uint32_t f=(entry_a>>8)&0xFFF, c=(entry_a>>20)&0xFFF;
            A.x = (A.x >> M_LOG) * f + (slot_a - c);
            bool need = (A.x < L_LOW);
            if (need && A.buf_avail == 0) {
                A.slab_idx -= 2;
                int64_t off = A.block_base_u32 + (int64_t)A.slab_idx*blockDim.x + tid;
                A.buf_lo=compressed_u32[off]; A.buf_hi=compressed_u32[off+blockDim.x];
                A.buf_avail=4;
            }
            uint32_t xr=(A.x<<16)|(A.buf_hi>>16), bh=(A.buf_hi<<16)|(A.buf_lo>>16);
            uint32_t bl=A.buf_lo<<16; int av=A.buf_avail-1;
            A.x=need?xr:A.x; A.buf_hi=need?bh:A.buf_hi;
            A.buf_lo=need?bl:A.buf_lo; A.buf_avail=need?av:A.buf_avail;
        }

        // Phase C: consume B
        uint8_t sym_b = 0;
        if (have_b) {
            sym_b = (uint8_t)(entry_b & 0xFF);
            uint32_t f=(entry_b>>8)&0xFFF, c=(entry_b>>20)&0xFFF;
            B.x = (B.x >> M_LOG) * f + (slot_b - c);
            bool need = (B.x < L_LOW);
            if (need && B.buf_avail == 0) {
                B.slab_idx -= 2;
                int64_t off = B.block_base_u32 + (int64_t)B.slab_idx*blockDim.x + tid;
                B.buf_lo=compressed_u32[off]; B.buf_hi=compressed_u32[off+blockDim.x];
                B.buf_avail=4;
            }
            uint32_t xr=(B.x<<16)|(B.buf_hi>>16), bh=(B.buf_hi<<16)|(B.buf_lo>>16);
            uint32_t bl=B.buf_lo<<16; int av=B.buf_avail-1;
            B.x=need?xr:B.x; B.buf_hi=need?bh:B.buf_hi;
            B.buf_lo=need?bl:B.buf_lo; B.buf_avail=need?av:B.buf_avail;
        }

        // Phase D: compose fp8 + write
        if (have_a) {
            uint8_t sm = sign_mantissa[i*n_streams+tid_a];
            uint8_t e0=sym_a>>4, e1=sym_a&0xF, s0=sm&0xF, s1=sm>>4;
            output[(2*i)*n_streams+tid_a] = ((s0&8)<<4)|(e0<<3)|(s0&7);
            output[(2*i+1)*n_streams+tid_a] = ((s1&8)<<4)|(e1<<3)|(s1&7);
        }
        if (have_b) {
            uint8_t sm = sign_mantissa[i*n_streams+tid_b];
            uint8_t e0=sym_b>>4, e1=sym_b&0xF, s0=sm&0xF, s1=sm>>4;
            output[(2*i)*n_streams+tid_b] = ((s0&8)<<4)|(e0<<3)|(s0&7);
            output[(2*i+1)*n_streams+tid_b] = ((s1&8)<<4)|(e1<<3)|(s1&7);
        }
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
        uint32_t slot_a=0, slot_b=0, entry_a=0, entry_b=0;
        if (have_a) { slot_a = A.x & (M_SIZE-1); entry_a = __ldg(&sfc_global[slot_a]); }
        if (have_b) { slot_b = B.x & (M_SIZE-1); entry_b = __ldg(&sfc_global[slot_b]); }

        uint8_t sym_a=0;
        if (have_a) {
            sym_a = (uint8_t)(entry_a & 0xFF);
            uint32_t f=(entry_a>>8)&0xFFF, c=(entry_a>>20)&0xFFF;
            A.x = (A.x >> M_LOG) * f + (slot_a - c);
            bool need = (A.x < L_LOW);
            if (need && A.buf_avail == 0) {
                A.slab_idx -= 2;
                int64_t off = A.block_base_u32 + (int64_t)A.slab_idx*blockDim.x + tid;
                A.buf_lo=compressed_u32[off]; A.buf_hi=compressed_u32[off+blockDim.x];
                A.buf_avail=4;
            }
            uint32_t xr=(A.x<<16)|(A.buf_hi>>16), bh=(A.buf_hi<<16)|(A.buf_lo>>16);
            uint32_t bl=A.buf_lo<<16; int av=A.buf_avail-1;
            A.x=need?xr:A.x; A.buf_hi=need?bh:A.buf_hi;
            A.buf_lo=need?bl:A.buf_lo; A.buf_avail=need?av:A.buf_avail;
        }

        uint8_t sym_b=0;
        if (have_b) {
            sym_b = (uint8_t)(entry_b & 0xFF);
            uint32_t f=(entry_b>>8)&0xFFF, c=(entry_b>>20)&0xFFF;
            B.x = (B.x >> M_LOG) * f + (slot_b - c);
            bool need = (B.x < L_LOW);
            if (need && B.buf_avail == 0) {
                B.slab_idx -= 2;
                int64_t off = B.block_base_u32 + (int64_t)B.slab_idx*blockDim.x + tid;
                B.buf_lo=compressed_u32[off]; B.buf_hi=compressed_u32[off+blockDim.x];
                B.buf_avail=4;
            }
            uint32_t xr=(B.x<<16)|(B.buf_hi>>16), bh=(B.buf_hi<<16)|(B.buf_lo>>16);
            uint32_t bl=B.buf_lo<<16; int av=B.buf_avail-1;
            B.x=need?xr:B.x; B.buf_hi=need?bh:B.buf_hi;
            B.buf_lo=need?bl:B.buf_lo; B.buf_avail=need?av:B.buf_avail;
        }

        if (have_a) {
            uint8_t sm = sign_mantissa[i*n_streams+tid_a];
            uint8_t e0=sym_a>>4, e1=sym_a&0xF, s0=sm&0xF, s1=sm>>4;
            digest ^= ((s0&8)<<4)|(e0<<3)|(s0&7);
            digest ^= ((s1&8)<<4)|(e1<<3)|(s1&7);
        }
        if (have_b) {
            uint8_t sm = sign_mantissa[i*n_streams+tid_b];
            uint8_t e0=sym_b>>4, e1=sym_b&0xF, s0=sm&0xF, s1=sm>>4;
            digest ^= ((s0&8)<<4)|(e0<<3)|(s0&7);
            digest ^= ((s1&8)<<4)|(e1<<3)|(s1&7);
        }
    }
    digest_out[(int64_t)blockIdx.x * blockDim.x + tid] = digest;
}

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
    int64_t blocks = (n_enc_blocks + 1) / 2;
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
    int64_t blocks = (n_enc_blocks + 1) / 2;
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

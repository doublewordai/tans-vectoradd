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
    int      slab_avail;   // (slab_idx << 3) | buf_avail — saves 1 register
    uint32_t buf_hi;
    uint32_t buf_lo;
    int32_t  block_base_u32;
};

#define CTX_SLAB_IDX(ctx)  ((ctx).slab_avail >> 3)
#define CTX_BUF_AVAIL(ctx) ((ctx).slab_avail & 7)
#define CTX_SET_SLAB_AVAIL(ctx, slab, avail) ((ctx).slab_avail = ((slab) << 3) | (avail))

// Multi-stream fused kernel with shared sfc table.
//
// Template parameters:
//   NS  — encoder blocks per thread (stream pairs per thread)
//   EPB — encoder blocks per CUDA block (≥1). Multiple encoder blocks
//         share the 16 KB SMEM sfc table, increasing occupancy.
//
// CUDA block has BLOCK_STREAMS * EPB threads. Thread i belongs to
// encoder-group (i / BLOCK_STREAMS) and has local tid (i % BLOCK_STREAMS).
template <int NS, int EPB>
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
    // Load 16 KB sfc table into SMEM, shared by all EPB encoder groups.
    __shared__ uint32_t sfc_smem[4096];
    for (int i = threadIdx.x; i < 4096; i += blockDim.x)
        sfc_smem[i] = sfc[i];
    __syncthreads();

    constexpr int EW = BLOCK_STREAMS;      // encoder block width (128)
    int enc_group = threadIdx.x / EW;      // which encoder block in this CUDA block
    int tid       = threadIdx.x % EW;      // thread within encoder block

    RansCtx ctxA[NS], ctxB[NS];
    int32_t sid[NS];
    bool have[NS];

    #pragma unroll
    for (int q = 0; q < NS; q++) {
        int enc = NS * (EPB * blockIdx.x + enc_group) + q;
        sid[q] = enc * EW + tid;
        have[q] = sid[q] < n_streams;
        if (have[q]) {
            int32_t ba = a_offsets[enc], na = a_offsets[enc+1];
            int Ga = (na-ba)/(EW*4);
            ctxA[q].x = a_states[sid[q]];
            CTX_SET_SLAB_AVAIL(ctxA[q], Ga, 0);
            ctxA[q].buf_hi=0; ctxA[q].buf_lo=0;
            ctxA[q].block_base_u32 = ba/4;

            int32_t bb = b_offsets[enc], nb = b_offsets[enc+1];
            int Gb = (nb-bb)/(EW*4);
            ctxB[q].x = b_states[sid[q]];
            CTX_SET_SLAB_AVAIL(ctxB[q], Gb, 0);
            ctxB[q].buf_hi=0; ctxB[q].buf_lo=0;
            ctxB[q].block_base_u32 = bb/4;
        }
    }

    bool any = false;
    #pragma unroll
    for (int q = 0; q < NS; q++) any |= have[q];
    if (!any) return;

    int64_t n_pairs = n_fp8_per_stream / 2;

    // Prologue: issue first iteration's reads
    uint32_t entA[NS], entB[NS];
    uint8_t smA_val[NS], smB_val[NS];
    #pragma unroll
    for (int q = 0; q < NS; q++) {
        if (have[q]) {
            entA[q] = sfc_smem[ctxA[q].x & (M_SIZE-1)];
            entB[q] = sfc_smem[ctxB[q].x & (M_SIZE-1)];
            smA_val[q] = a_sm[0 * n_streams + sid[q]];
            smB_val[q] = b_sm[0 * n_streams + sid[q]];
        }
    }

    uint8_t out0[NS], out1[NS];

    for (int64_t i = 0; i < n_pairs; i++) {
        // ── Step 1: Write PREVIOUS iteration's output (DRAM store) ──
        if (i > 0) {
            #pragma unroll
            for (int q = 0; q < NS; q++) {
                if (have[q]) {
                    output[(2*(i-1))*n_streams + sid[q]] = out0[q];
                    output[(2*(i-1)+1)*n_streams + sid[q]] = out1[q];
                }
            }
        }

        // ── Step 2: Consume current entries ──
        #pragma unroll
        for (int q = 0; q < NS; q++) {
            if (have[q]) {
                // Consume A
                uint32_t slA = ctxA[q].x & (M_SIZE-1);
                uint8_t symA = entA[q] & 0xFF;
                uint32_t fA=(entA[q]>>8)&0xFFF, cA=(entA[q]>>20)&0xFFF;
                ctxA[q].x = (ctxA[q].x >> M_LOG) * fA + (slA - cA);
                bool needA = (ctxA[q].x < L_LOW);
                if (needA && CTX_BUF_AVAIL(ctxA[q]) == 0) {
                    int sA = CTX_SLAB_IDX(ctxA[q]) - 2;
                    int32_t off = ctxA[q].block_base_u32 + sA*EW + tid;
                    ctxA[q].buf_lo=a_compressed[off]; ctxA[q].buf_hi=a_compressed[off+EW];
                    CTX_SET_SLAB_AVAIL(ctxA[q], sA, 4);
                }
                uint32_t xrA=(ctxA[q].x<<16)|(ctxA[q].buf_hi>>16);
                uint32_t bhA=(ctxA[q].buf_hi<<16)|(ctxA[q].buf_lo>>16);
                uint32_t blA=ctxA[q].buf_lo<<16;
                int avA=CTX_BUF_AVAIL(ctxA[q])-1;
                ctxA[q].x=needA?xrA:ctxA[q].x; ctxA[q].buf_hi=needA?bhA:ctxA[q].buf_hi;
                ctxA[q].buf_lo=needA?blA:ctxA[q].buf_lo;
                CTX_SET_SLAB_AVAIL(ctxA[q], CTX_SLAB_IDX(ctxA[q]), needA?avA:CTX_BUF_AVAIL(ctxA[q]));

                // Consume B
                uint32_t slB = ctxB[q].x & (M_SIZE-1);
                uint8_t symB = entB[q] & 0xFF;
                uint32_t fB=(entB[q]>>8)&0xFFF, cB=(entB[q]>>20)&0xFFF;
                ctxB[q].x = (ctxB[q].x >> M_LOG) * fB + (slB - cB);
                bool needB = (ctxB[q].x < L_LOW);
                if (needB && CTX_BUF_AVAIL(ctxB[q]) == 0) {
                    int sB = CTX_SLAB_IDX(ctxB[q]) - 2;
                    int32_t off = ctxB[q].block_base_u32 + sB*EW + tid;
                    ctxB[q].buf_lo=b_compressed[off]; ctxB[q].buf_hi=b_compressed[off+EW];
                    CTX_SET_SLAB_AVAIL(ctxB[q], sB, 4);
                }
                uint32_t xrB=(ctxB[q].x<<16)|(ctxB[q].buf_hi>>16);
                uint32_t bhB=(ctxB[q].buf_hi<<16)|(ctxB[q].buf_lo>>16);
                uint32_t blB=ctxB[q].buf_lo<<16;
                int avB=CTX_BUF_AVAIL(ctxB[q])-1;
                ctxB[q].x=needB?xrB:ctxB[q].x; ctxB[q].buf_hi=needB?bhB:ctxB[q].buf_hi;
                ctxB[q].buf_lo=needB?blB:ctxB[q].buf_lo;
                CTX_SET_SLAB_AVAIL(ctxB[q], CTX_SLAB_IDX(ctxB[q]), needB?avB:CTX_BUF_AVAIL(ctxB[q]));

                // Packed compose: build both FP8 bytes in parallel using
                // 16-bit halves of a uint32, then use hardware F2FP.
                {
                    // A: compose 2 FP8 bytes from symA + smA_val
                    uint32_t eA = ((uint32_t)(symA >> 4)) | (((uint32_t)(symA & 0xF)) << 16);
                    uint32_t sA = ((uint32_t)(smA_val[q] & 0xF)) | (((uint32_t)(smA_val[q] >> 4)) << 16);
                    uint32_t pA = ((sA & 0x80008u) << 4) | ((eA << 3) & 0x780078u) | (sA & 0x70007u);
                    __nv_fp8_e4m3 a0, a1;
                    memcpy(&a0, &pA, 1);
                    uint8_t pA1 = (uint8_t)(pA >> 16);
                    memcpy(&a1, &pA1, 1);

                    // B: same
                    uint32_t eB = ((uint32_t)(symB >> 4)) | (((uint32_t)(symB & 0xF)) << 16);
                    uint32_t sB = ((uint32_t)(smB_val[q] & 0xF)) | (((uint32_t)(smB_val[q] >> 4)) << 16);
                    uint32_t pB = ((sB & 0x80008u) << 4) | ((eB << 3) & 0x780078u) | (sB & 0x70007u);
                    __nv_fp8_e4m3 b0, b1;
                    memcpy(&b0, &pB, 1);
                    uint8_t pB1 = (uint8_t)(pB >> 16);
                    memcpy(&b1, &pB1, 1);

                    // Add + convert back
                    __nv_fp8_e4m3 c0(float(a0) + float(b0));
                    __nv_fp8_e4m3 c1(float(a1) + float(b1));
                    memcpy(&out0[q], &c0, 1);
                    memcpy(&out1[q], &c1, 1);
                }
            }
        }

        // ── Step 3: Issue NEXT iteration's sfc reads + sm reads ──
        if (i + 1 < n_pairs) {
            #pragma unroll
            for (int q = 0; q < NS; q++) {
                if (have[q]) {
                    entA[q] = sfc_smem[ctxA[q].x & (M_SIZE-1)];
                    entB[q] = sfc_smem[ctxB[q].x & (M_SIZE-1)];
                    smA_val[q] = a_sm[(i+1) * n_streams + sid[q]];
                    smB_val[q] = b_sm[(i+1) * n_streams + sid[q]];
                }
            }
        }
    }

    // Epilogue: write final iteration
    #pragma unroll
    for (int q = 0; q < NS; q++) {
        if (have[q]) {
            output[(2*(n_pairs-1))*n_streams + sid[q]] = out0[q];
            output[(2*(n_pairs-1)+1)*n_streams + sid[q]] = out1[q];
        }
    }
}

// ─── Two-pass tiled fused kernel ─────────────────────────────────────
//
// Splits the decode loop into two passes within each tile:
//   Pass 1 (decode-only): Decode pair symbols, write decoded bytes to SMEM.
//   Pass 2 (compose+add): Read decoded syms from SMEM + sm from global,
//                         compose FP8 bytes, add, write output.
//
// Pass 1 sheds the compose+add ALU, so warps cycle faster and issue memory
// requests more evenly. At EPB=4 this combines with 3 blocks/SM and becomes
// the best measured fused kernel on Ada.
template <int EPB, int TILE>
__global__ void fp8_vecadd_twopass_kernel(
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
    constexpr int EW = BLOCK_STREAMS;      // 128

    extern __shared__ uint8_t smem_raw[];
    uint32_t* sfc_smem     = reinterpret_cast<uint32_t*>(smem_raw);
    uint8_t*  decoded_symA = smem_raw + 4096 * sizeof(uint32_t);
    uint8_t*  decoded_symB = decoded_symA + TILE * blockDim.x;

    for (int i = threadIdx.x; i < 4096; i += blockDim.x)
        sfc_smem[i] = sfc[i];
    __syncthreads();

    int enc_group = threadIdx.x / EW;
    int tid       = threadIdx.x % EW;

    int enc = EPB * blockIdx.x + enc_group;
    int32_t sid = enc * EW + tid;
    bool active = sid < n_streams;

    RansCtx ctxA, ctxB;
    if (active) {
        int32_t ba = a_offsets[enc], na = a_offsets[enc+1];
        int Ga = (na-ba)/(EW*4);
        ctxA.x = a_states[sid];
        CTX_SET_SLAB_AVAIL(ctxA, Ga, 0);
        ctxA.buf_hi=0; ctxA.buf_lo=0;
        ctxA.block_base_u32 = ba/4;

        int32_t bb = b_offsets[enc], nb = b_offsets[enc+1];
        int Gb = (nb-bb)/(EW*4);
        ctxB.x = b_states[sid];
        CTX_SET_SLAB_AVAIL(ctxB, Gb, 0);
        ctxB.buf_hi=0; ctxB.buf_lo=0;
        ctxB.block_base_u32 = bb/4;
    }

    if (!active) return;

    int64_t n_pairs = n_fp8_per_stream / 2;

    for (int64_t tile_start = 0; tile_start < n_pairs; tile_start += TILE) {
        int64_t tile_end = tile_start + TILE;
        if (tile_end > n_pairs) tile_end = n_pairs;
        int tile_len = (int)(tile_end - tile_start);

        // Pass 1: decode only.
        for (int t = 0; t < tile_len; t++) {
            uint32_t entA = sfc_smem[ctxA.x & (M_SIZE-1)];
            uint8_t symA = entA & 0xFF;
            uint32_t fA=(entA>>8)&0xFFF, cA=(entA>>20)&0xFFF;
            uint32_t slA = ctxA.x & (M_SIZE-1);
            ctxA.x = (ctxA.x >> M_LOG) * fA + (slA - cA);
            bool needA = (ctxA.x < L_LOW);
            if (needA && CTX_BUF_AVAIL(ctxA) == 0) {
                int sA = CTX_SLAB_IDX(ctxA) - 2;
                int32_t off = ctxA.block_base_u32 + sA*EW + tid;
                ctxA.buf_lo=a_compressed[off]; ctxA.buf_hi=a_compressed[off+EW];
                CTX_SET_SLAB_AVAIL(ctxA, sA, 4);
            }
            uint32_t xrA=(ctxA.x<<16)|(ctxA.buf_hi>>16);
            uint32_t bhA=(ctxA.buf_hi<<16)|(ctxA.buf_lo>>16);
            uint32_t blA=ctxA.buf_lo<<16;
            int avA=CTX_BUF_AVAIL(ctxA)-1;
            ctxA.x=needA?xrA:ctxA.x; ctxA.buf_hi=needA?bhA:ctxA.buf_hi;
            ctxA.buf_lo=needA?blA:ctxA.buf_lo;
            CTX_SET_SLAB_AVAIL(ctxA, CTX_SLAB_IDX(ctxA), needA?avA:CTX_BUF_AVAIL(ctxA));
            decoded_symA[t * blockDim.x + threadIdx.x] = symA;

            uint32_t entB = sfc_smem[ctxB.x & (M_SIZE-1)];
            uint8_t symB = entB & 0xFF;
            uint32_t fB=(entB>>8)&0xFFF, cB=(entB>>20)&0xFFF;
            uint32_t slB = ctxB.x & (M_SIZE-1);
            ctxB.x = (ctxB.x >> M_LOG) * fB + (slB - cB);
            bool needB = (ctxB.x < L_LOW);
            if (needB && CTX_BUF_AVAIL(ctxB) == 0) {
                int sB = CTX_SLAB_IDX(ctxB) - 2;
                int32_t off = ctxB.block_base_u32 + sB*EW + tid;
                ctxB.buf_lo=b_compressed[off]; ctxB.buf_hi=b_compressed[off+EW];
                CTX_SET_SLAB_AVAIL(ctxB, sB, 4);
            }
            uint32_t xrB=(ctxB.x<<16)|(ctxB.buf_hi>>16);
            uint32_t bhB=(ctxB.buf_hi<<16)|(ctxB.buf_lo>>16);
            uint32_t blB=ctxB.buf_lo<<16;
            int avB=CTX_BUF_AVAIL(ctxB)-1;
            ctxB.x=needB?xrB:ctxB.x; ctxB.buf_hi=needB?bhB:ctxB.buf_hi;
            ctxB.buf_lo=needB?blB:ctxB.buf_lo;
            CTX_SET_SLAB_AVAIL(ctxB, CTX_SLAB_IDX(ctxB), needB?avB:CTX_BUF_AVAIL(ctxB));
            decoded_symB[t * blockDim.x + threadIdx.x] = symB;
        }
        __syncthreads();

        // Pass 2: compose + add + write.
        for (int t = 0; t < tile_len; t++) {
            int64_t i = tile_start + t;
            uint8_t symA = decoded_symA[t * blockDim.x + threadIdx.x];
            uint8_t symB = decoded_symB[t * blockDim.x + threadIdx.x];
            uint8_t smA_val = a_sm[i * n_streams + sid];
            uint8_t smB_val = b_sm[i * n_streams + sid];

            uint32_t eA = ((uint32_t)(symA >> 4)) | (((uint32_t)(symA & 0xF)) << 16);
            uint32_t sA_ = ((uint32_t)(smA_val & 0xF)) | (((uint32_t)(smA_val >> 4)) << 16);
            uint32_t pA = ((sA_ & 0x80008u) << 4) | ((eA << 3) & 0x780078u) | (sA_ & 0x70007u);
            __nv_fp8_e4m3 a0, a1;
            memcpy(&a0, &pA, 1);
            uint8_t pA1 = (uint8_t)(pA >> 16);
            memcpy(&a1, &pA1, 1);

            uint32_t eB = ((uint32_t)(symB >> 4)) | (((uint32_t)(symB & 0xF)) << 16);
            uint32_t sB_ = ((uint32_t)(smB_val & 0xF)) | (((uint32_t)(smB_val >> 4)) << 16);
            uint32_t pB = ((sB_ & 0x80008u) << 4) | ((eB << 3) & 0x780078u) | (sB_ & 0x70007u);
            __nv_fp8_e4m3 b0, b1;
            memcpy(&b0, &pB, 1);
            uint8_t pB1 = (uint8_t)(pB >> 16);
            memcpy(&b1, &pB1, 1);

            __nv_fp8_e4m3 c0(float(a0) + float(b0));
            __nv_fp8_e4m3 c1(float(a1) + float(b1));
            uint8_t r0, r1;
            memcpy(&r0, &c0, 1);
            memcpy(&r1, &c1, 1);
            output[(2*i)*n_streams + sid] = r0;
            output[(2*i+1)*n_streams + sid] = r1;
        }
        __syncthreads();
    }
}

// ─── Tiled-layout two-pass kernel ────────────────────────────────────
//
// sm layout: [n_tiles, n_streams, TILE] — TILE consecutive bytes per
// thread per tile, fitting in one DRAM page for ~100% row buffer hits.
// output layout: [n_tiles, n_streams, TILE*2].
//
// Pass 2 reads sm via uint64 loads and writes output via uint64 stores,
// replacing byte-granularity accesses that caused DRAM row misses at
// large N.

template <int EPB, int TILE>
__global__ void fp8_vecadd_tiled_kernel(
    const uint32_t* __restrict__ a_compressed,
    const int32_t*  __restrict__ a_offsets,
    const uint32_t* __restrict__ a_states,
    const uint8_t*  __restrict__ a_sm_tiled,
    const uint32_t* __restrict__ b_compressed,
    const int32_t*  __restrict__ b_offsets,
    const uint32_t* __restrict__ b_states,
    const uint8_t*  __restrict__ b_sm_tiled,
    const uint32_t* __restrict__ sfc,
    uint8_t*        __restrict__ output_tiled,
    int64_t n_fp8_per_stream,
    int64_t n_streams)
{
    constexpr int EW = BLOCK_STREAMS;
    static_assert(TILE % 8 == 0, "TILE must be a multiple of 8 for uint64 vectorization");

    extern __shared__ uint8_t smem_raw[];
    uint32_t* sfc_smem     = reinterpret_cast<uint32_t*>(smem_raw);
    uint8_t*  decoded_symA = smem_raw + 4096 * sizeof(uint32_t);
    uint8_t*  decoded_symB = decoded_symA + TILE * blockDim.x;

    for (int i = threadIdx.x; i < 4096; i += blockDim.x)
        sfc_smem[i] = sfc[i];
    __syncthreads();

    int enc_group = threadIdx.x / EW;
    int tid       = threadIdx.x % EW;
    int enc = EPB * blockIdx.x + enc_group;
    int32_t sid = enc * EW + tid;
    bool active = sid < n_streams;

    RansCtx ctxA, ctxB;
    if (active) {
        int32_t ba = a_offsets[enc], na = a_offsets[enc+1];
        int Ga = (na-ba)/(EW*4);
        ctxA.x = a_states[sid];
        CTX_SET_SLAB_AVAIL(ctxA, Ga, 0);
        ctxA.buf_hi=0; ctxA.buf_lo=0;
        ctxA.block_base_u32 = ba/4;

        int32_t bb = b_offsets[enc], nb = b_offsets[enc+1];
        int Gb = (nb-bb)/(EW*4);
        ctxB.x = b_states[sid];
        CTX_SET_SLAB_AVAIL(ctxB, Gb, 0);
        ctxB.buf_hi=0; ctxB.buf_lo=0;
        ctxB.block_base_u32 = bb/4;
    }

    if (!active) return;

    int64_t n_pairs = n_fp8_per_stream / 2;
    int64_t n_tiles_total = n_pairs / TILE;

    for (int64_t tile_idx = 0; tile_idx < n_tiles_total; tile_idx++) {
        // Prefetch sm for this tile — uint64 loads issued before decode,
        // in-flight during pass 1, ready by pass 2.
        int64_t sm_base = (tile_idx * n_streams + sid) * (int64_t)TILE;
        constexpr int SM_CHUNKS = TILE / 8;
        uint64_t smA_chunks[SM_CHUNKS], smB_chunks[SM_CHUNKS];
        #pragma unroll
        for (int c = 0; c < SM_CHUNKS; c++) {
            smA_chunks[c] = *reinterpret_cast<const uint64_t*>(a_sm_tiled + sm_base + c * 8);
            smB_chunks[c] = *reinterpret_cast<const uint64_t*>(b_sm_tiled + sm_base + c * 8);
        }

        // Pass 1: decode only
        #pragma unroll
        for (int t = 0; t < TILE; t++) {
            uint32_t entA = sfc_smem[ctxA.x & (M_SIZE-1)];
            uint8_t symA = entA & 0xFF;
            uint32_t fA=(entA>>8)&0xFFF, cA=(entA>>20)&0xFFF;
            uint32_t slA = ctxA.x & (M_SIZE-1);
            ctxA.x = (ctxA.x >> M_LOG) * fA + (slA - cA);
            bool needA = (ctxA.x < L_LOW);
            if (needA && CTX_BUF_AVAIL(ctxA) == 0) {
                int sA = CTX_SLAB_IDX(ctxA) - 2;
                int32_t off = ctxA.block_base_u32 + sA*EW + tid;
                ctxA.buf_lo=a_compressed[off]; ctxA.buf_hi=a_compressed[off+EW];
                CTX_SET_SLAB_AVAIL(ctxA, sA, 4);
            }
            uint32_t xrA=(ctxA.x<<16)|(ctxA.buf_hi>>16);
            uint32_t bhA=(ctxA.buf_hi<<16)|(ctxA.buf_lo>>16);
            uint32_t blA=ctxA.buf_lo<<16;
            int avA=CTX_BUF_AVAIL(ctxA)-1;
            ctxA.x=needA?xrA:ctxA.x; ctxA.buf_hi=needA?bhA:ctxA.buf_hi;
            ctxA.buf_lo=needA?blA:ctxA.buf_lo;
            CTX_SET_SLAB_AVAIL(ctxA, CTX_SLAB_IDX(ctxA), needA?avA:CTX_BUF_AVAIL(ctxA));
            decoded_symA[t * blockDim.x + threadIdx.x] = symA;

            uint32_t entB = sfc_smem[ctxB.x & (M_SIZE-1)];
            uint8_t symB = entB & 0xFF;
            uint32_t fB=(entB>>8)&0xFFF, cB=(entB>>20)&0xFFF;
            uint32_t slB = ctxB.x & (M_SIZE-1);
            ctxB.x = (ctxB.x >> M_LOG) * fB + (slB - cB);
            bool needB = (ctxB.x < L_LOW);
            if (needB && CTX_BUF_AVAIL(ctxB) == 0) {
                int sB = CTX_SLAB_IDX(ctxB) - 2;
                int32_t off = ctxB.block_base_u32 + sB*EW + tid;
                ctxB.buf_lo=b_compressed[off]; ctxB.buf_hi=b_compressed[off+EW];
                CTX_SET_SLAB_AVAIL(ctxB, sB, 4);
            }
            uint32_t xrB=(ctxB.x<<16)|(ctxB.buf_hi>>16);
            uint32_t bhB=(ctxB.buf_hi<<16)|(ctxB.buf_lo>>16);
            uint32_t blB=ctxB.buf_lo<<16;
            int avB=CTX_BUF_AVAIL(ctxB)-1;
            ctxB.x=needB?xrB:ctxB.x; ctxB.buf_hi=needB?bhB:ctxB.buf_hi;
            ctxB.buf_lo=needB?blB:ctxB.buf_lo;
            CTX_SET_SLAB_AVAIL(ctxB, CTX_SLAB_IDX(ctxB), needB?avB:CTX_BUF_AVAIL(ctxB));
            decoded_symB[t * blockDim.x + threadIdx.x] = symB;
        }
        __syncthreads();

        // Pass 2: compose + add + write from pre-loaded sm chunks
        uint8_t out_buf[TILE * 2];

        #pragma unroll
        for (int t = 0; t < TILE; t++) {
            uint8_t symA = decoded_symA[t * blockDim.x + threadIdx.x];
            uint8_t symB = decoded_symB[t * blockDim.x + threadIdx.x];
            uint8_t smA_val = (uint8_t)(smA_chunks[t / 8] >> ((t % 8) * 8));
            uint8_t smB_val = (uint8_t)(smB_chunks[t / 8] >> ((t % 8) * 8));

            uint32_t eA = ((uint32_t)(symA >> 4)) | (((uint32_t)(symA & 0xF)) << 16);
            uint32_t sA_ = ((uint32_t)(smA_val & 0xF)) | (((uint32_t)(smA_val >> 4)) << 16);
            uint32_t pA = ((sA_ & 0x80008u) << 4) | ((eA << 3) & 0x780078u) | (sA_ & 0x70007u);
            __nv_fp8_e4m3 a0, a1;
            memcpy(&a0, &pA, 1);
            uint8_t pA1 = (uint8_t)(pA >> 16);
            memcpy(&a1, &pA1, 1);

            uint32_t eB = ((uint32_t)(symB >> 4)) | (((uint32_t)(symB & 0xF)) << 16);
            uint32_t sB_ = ((uint32_t)(smB_val & 0xF)) | (((uint32_t)(smB_val >> 4)) << 16);
            uint32_t pB = ((sB_ & 0x80008u) << 4) | ((eB << 3) & 0x780078u) | (sB_ & 0x70007u);
            __nv_fp8_e4m3 b0, b1;
            memcpy(&b0, &pB, 1);
            uint8_t pB1 = (uint8_t)(pB >> 16);
            memcpy(&b1, &pB1, 1);

            __nv_fp8_e4m3 c0(float(a0) + float(b0));
            __nv_fp8_e4m3 c1(float(a1) + float(b1));
            memcpy(&out_buf[t * 2], &c0, 1);
            memcpy(&out_buf[t * 2 + 1], &c1, 1);
        }

        int64_t out_base = (tile_idx * n_streams + sid) * (int64_t)(TILE * 2);
        *reinterpret_cast<uint4*>(output_tiled + out_base) =
            *reinterpret_cast<const uint4*>(out_buf);

        __syncthreads();
    }
}

// Host launcher — dispatch on (NS, EPB) combinations
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

    #define LAUNCH_FUSED(N, E) \
        fp8_vecadd_fused_kernel<N, E><<<(n_enc_blocks+(N*E)-1)/(N*E), BLOCK_STREAMS*E>>>( \
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
        // Single-pass pipelined, NS=1 EPB=8 (1024 threads/block)
        case 8: LAUNCH_FUSED(1, 8); break;
        // Two-pass tiled, EPB=4 TILE=8 (pair-major sm layout)
        case 504: {
            constexpr int E = 4, T = 8;
            int smem_bytes = 4096 * 4 + 2 * T * BLOCK_STREAMS * E;
            fp8_vecadd_twopass_kernel<E, T><<<n_enc_blocks/E, BLOCK_STREAMS*E, smem_bytes>>>(
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
            break;
        }
        // Two-pass tiled layout, EPB=4 TILE=8 (tiled sm + output for DRAM row hits)
        case 704: {
            constexpr int E = 4, T = 8;
            int64_t n_pairs = n_fp8_per_stream / 2;
            int64_t n_tiles = n_pairs / T;
            auto output_tiled = torch::empty(
                {n_tiles * n_streams * T * 2},
                torch::TensorOptions().dtype(torch::kUInt8).device(a_comp.device()));
            int smem_bytes = 4096 * 4 + 2 * T * BLOCK_STREAMS * E;
            fp8_vecadd_tiled_kernel<E, T><<<n_enc_blocks/E, BLOCK_STREAMS*E, smem_bytes>>>(
                reinterpret_cast<const uint32_t*>(a_comp.data_ptr<uint8_t>()),
                a_offsets.data_ptr<int32_t>(),
                reinterpret_cast<const uint32_t*>(a_states.data_ptr<int32_t>()),
                a_sm.data_ptr<uint8_t>(),
                reinterpret_cast<const uint32_t*>(b_comp.data_ptr<uint8_t>()),
                b_offsets.data_ptr<int32_t>(),
                reinterpret_cast<const uint32_t*>(b_states.data_ptr<int32_t>()),
                b_sm.data_ptr<uint8_t>(),
                reinterpret_cast<const uint32_t*>(sfc_table.data_ptr<int32_t>()),
                output_tiled.data_ptr<uint8_t>(),
                n_fp8_per_stream, n_streams);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return output_tiled;
        }
        default: TORCH_CHECK(false, "ns must be 8, 504, or 704");
    }
    #undef LAUNCH_FUSED
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

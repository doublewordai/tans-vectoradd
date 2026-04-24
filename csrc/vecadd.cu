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

// ─── Fused decompress + tiled vecadd ────────────────────────────────
//
// Two-pass tiled architecture with ballot-gated tile-boundary refills.
//
// Data layouts:
//   sm:     [n_tiles, n_streams, TILE]    — tiled for DRAM row hits
//   output: [n_tiles, n_streams, TILE*2]  — tiled for DRAM row hits
//
// Pass 1: decode rANS pair symbols, write to SMEM
// Pass 2: read decoded syms from SMEM + sm from global (uint64),
//          compose FP8 bytes, add, write output (uint4)

constexpr int      M_LOG         = 12;
constexpr uint32_t M_SIZE        = 1u << M_LOG;
constexpr uint32_t L_LOW         = 1u << 16;
constexpr int      BLOCK_STREAMS = 128;

struct RansCtx {
    uint32_t x;
    int      slab_avail;   // (slab_idx << 3) | buf_avail
    uint32_t buf_hi;
    uint32_t buf_lo;
    int32_t  block_base_u32;
};

#define CTX_SLAB_IDX(ctx)  ((ctx).slab_avail >> 3)
#define CTX_BUF_AVAIL(ctx) ((ctx).slab_avail & 7)
#define CTX_SET_SLAB_AVAIL(ctx, slab, avail) ((ctx).slab_avail = ((slab) << 3) | (avail))

#define REFILL_BUF(CTX, COMPRESSED, EW, TID) do {                      \
    int sl_ = CTX_SLAB_IDX(CTX) - 2;                                   \
    int32_t off_ = CTX.block_base_u32 + sl_*(EW) + (TID);              \
    CTX.buf_lo = COMPRESSED[off_];                                      \
    CTX.buf_hi = COMPRESSED[off_ + (EW)];                              \
    CTX_SET_SLAB_AVAIL(CTX, sl_, 4);                                   \
} while(0)

template <int EPB, int TILE>
__global__ void fp8_vecadd_fused_kernel(
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
        // Tile-boundary refill: ballot-gated, uniform across warp
        {
            bool need_A = (CTX_BUF_AVAIL(ctxA) == 0);
            if (__ballot_sync(0xFFFFFFFF, need_A)) {
                if (need_A) { REFILL_BUF(ctxA, a_compressed, EW, tid); }
            }
            bool need_B = (CTX_BUF_AVAIL(ctxB) == 0);
            if (__ballot_sync(0xFFFFFFFF, need_B)) {
                if (need_B) { REFILL_BUF(ctxB, b_compressed, EW, tid); }
            }
        }

        // Prefetch sm for this tile
        int64_t sm_base = (tile_idx * n_streams + sid) * (int64_t)TILE;
        constexpr int SM_CHUNKS = TILE / 8;
        uint64_t smA_chunks[SM_CHUNKS], smB_chunks[SM_CHUNKS];
        #pragma unroll
        for (int c = 0; c < SM_CHUNKS; c++) {
            smA_chunks[c] = *reinterpret_cast<const uint64_t*>(a_sm_tiled + sm_base + c * 8);
            smB_chunks[c] = *reinterpret_cast<const uint64_t*>(b_sm_tiled + sm_base + c * 8);
        }

        // Pass 1: decode with safety-net refill
        #pragma unroll
        for (int t = 0; t < TILE; t++) {
            uint32_t entA = sfc_smem[ctxA.x & (M_SIZE-1)];
            uint8_t symA = entA & 0xFF;
            uint32_t fA=(entA>>8)&0xFFF, cA=(entA>>20)&0xFFF;
            uint32_t slA = ctxA.x & (M_SIZE-1);
            ctxA.x = (ctxA.x >> M_LOG) * fA + (slA - cA);
            bool needA = (ctxA.x < L_LOW);
            if (needA && CTX_BUF_AVAIL(ctxA) == 0) {
                REFILL_BUF(ctxA, a_compressed, EW, tid);
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
                REFILL_BUF(ctxB, b_compressed, EW, tid);
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

        // Pass 2: compose + add + write
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

// ─── Host launcher ──────────────────────────────────────────────────

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
    constexpr int E = 4, T = 8;
    int64_t n_pairs = n_fp8_per_stream / 2;
    int64_t n_tiles = n_pairs / T;

    auto output = torch::empty(
        {n_tiles * n_streams * T * 2},
        torch::TensorOptions().dtype(torch::kUInt8).device(a_comp.device()));

    int smem_bytes = 4096 * 4 + 2 * T * BLOCK_STREAMS * E;
    fp8_vecadd_fused_kernel<E, T><<<n_enc_blocks/E, BLOCK_STREAMS*E, smem_bytes>>>(
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

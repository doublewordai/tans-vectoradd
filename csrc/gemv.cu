// Fused decompress + FP8 GEMV.
//
// y = W @ x where W is [M, K] compressed FP8 weights and x is [K] FP16.
// Each thread decodes one row of W and accumulates its dot product with x.
//
// Data layout for W mirrors vecadd's:
//   M rows of W become M rANS streams of length K.
//   BLOCK_STREAMS=128 rows per encoder block.
//   sm is tiled as [n_tiles, M, TILE] for DRAM row hits.
//
// x is loaded once into SMEM and shared across all threads in a block.

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <stdint.h>

// rANS constants — must match the encoder and vecadd.cu.
namespace {
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
}  // namespace

// ─── Raw FP8 GEMV baseline ───────────────────────────────────────────
//
// One warp per output row. 32 threads cooperatively stream the row
// using uint4 (16-byte) loads — each warp-cycle reads 32×16 = 512
// bytes of W (= 16 sectors, perfectly coalesced). Warp-level reduction
// at the end.
//
// Constraints for first pass: K must be divisible by 32*16 = 512.

__global__ void fp8_gemv_raw_kernel(
    const uint4*   __restrict__ W,    // [M, K/16] uint4 view of [M, K] FP8
    const __half*  __restrict__ x,    // [K]
    __half*        __restrict__ y,    // [M]
    int M, int K)
{
    extern __shared__ uint8_t smem_raw[];
    __half* x_smem = reinterpret_cast<__half*>(smem_raw);

    for (int i = threadIdx.x; i < K; i += blockDim.x)
        x_smem[i] = x[i];
    __syncthreads();

    int warps_per_block = blockDim.x / 32;
    int warp_in_block   = threadIdx.x / 32;
    int lane            = threadIdx.x & 31;
    int m = (int)blockIdx.x * warps_per_block + warp_in_block;
    if (m >= M) return;

    int n_u4     = K / 16;            // uint4s per row
    int n_u4_lane = n_u4 / 32;        // uint4s per lane

    float acc = 0.0f;
    const uint4* W_row = W + (int64_t)m * n_u4;
    #pragma unroll 1
    for (int i = 0; i < n_u4_lane; i++) {
        int u4_idx = i * 32 + lane;
        uint4 w_vec = W_row[u4_idx];
        int k_base = u4_idx * 16;
        #pragma unroll
        for (int w_off = 0; w_off < 4; w_off++) {
            uint32_t word = (&w_vec.x)[w_off];
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                uint8_t wb = (word >> (b * 8)) & 0xFF;
                __nv_fp8_e4m3 w;
                memcpy(&w, &wb, 1);
                acc += float(w) * __half2float(x_smem[k_base + w_off * 4 + b]);
            }
        }
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);

    if (lane == 0)
        y[m] = __float2half(acc);
}

torch::Tensor fp8_gemv_raw(torch::Tensor W, torch::Tensor x) {
    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kUInt8);
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf);
    TORCH_CHECK(W.dim() == 2 && x.dim() == 1);
    int M = (int)W.size(0), K = (int)W.size(1);
    TORCH_CHECK((int)x.size(0) == K);
    TORCH_CHECK(K % 512 == 0, "K must be a multiple of 512 (first pass)");

    auto y = torch::empty({M},
        torch::TensorOptions().dtype(torch::kHalf).device(W.device()));

    // 256 threads/block = 8 warps/block = 8 output rows/block
    const int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = (M + warps_per_block - 1) / warps_per_block;
    size_t smem = K * sizeof(__half);
    TORCH_CHECK(smem <= 99 * 1024, "K too large for SMEM (first pass)");
    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(fp8_gemv_raw_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    }

    fp8_gemv_raw_kernel<<<blocks, threads, smem>>>(
        reinterpret_cast<const uint4*>(W.data_ptr<uint8_t>()),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        M, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// ─── Fused decompress + GEMV ────────────────────────────────────────
//
// Segmentation: each row of W (length K) is split into SEG_PER_ROW=32
// independent rANS streams of length N_seg = K/32. This gives enough
// parallelism to saturate the GPU at realistic inference M sizes.
//
// Thread assignment: each warp (32 lanes) owns ONE output row. Lane s
// decodes segment s of that row (covering x[s*N_seg : (s+1)*N_seg]).
// Warp-level __shfl_xor reduction sums the 32 partial dot products,
// lane 0 writes y[m].
//
// Encoder-block alignment: BLOCK_STREAMS=128 = 4 warps = 4 rows per
// encoder block. Consecutive 32 streams in an encoder block form one
// row's 32 segments, naturally aligned to a warp.

constexpr int SEG_PER_ROW = 32;

template <int EPB, int TILE>
__global__ void fp8_gemv_fused_kernel(
    const uint32_t* __restrict__ W_compressed,
    const int32_t*  __restrict__ W_offsets,
    const uint32_t* __restrict__ W_states,
    const uint8_t*  __restrict__ W_sm_tiled,   // [n_tiles, M*SEG_PER_ROW, TILE]
    const uint32_t* __restrict__ sfc,
    const __half*   __restrict__ x,            // [K]
    __half*         __restrict__ y,            // [M]
    int M, int K)
{
    constexpr int EW = BLOCK_STREAMS;
    constexpr int ROWS_PER_EB = EW / SEG_PER_ROW;  // 4
    static_assert(TILE % 8 == 0, "TILE must be a multiple of 8");
    static_assert(EW % SEG_PER_ROW == 0, "BLOCK_STREAMS must be divisible by SEG_PER_ROW");

    extern __shared__ uint8_t smem_raw[];
    uint32_t* sfc_smem = reinterpret_cast<uint32_t*>(smem_raw);
    uint8_t*  decoded  = smem_raw + 4096 * sizeof(uint32_t);
    __half*   x_smem   = reinterpret_cast<__half*>(decoded + TILE * blockDim.x);

    for (int i = threadIdx.x; i < 4096; i += blockDim.x)
        sfc_smem[i] = sfc[i];
    for (int i = threadIdx.x; i < K; i += blockDim.x)
        x_smem[i] = x[i];
    __syncthreads();

    int enc_group = threadIdx.x / EW;
    int tid       = threadIdx.x % EW;            // 0..127 within enc block
    int enc       = EPB * (int)blockIdx.x + enc_group;

    int row_in_eb = tid / SEG_PER_ROW;            // 0..ROWS_PER_EB-1
    int seg_idx   = tid % SEG_PER_ROW;            // 0..SEG_PER_ROW-1 (= lane)
    int m         = enc * ROWS_PER_EB + row_in_eb;
    int32_t stream_id = enc * EW + tid;
    bool active   = m < M;

    RansCtx ctx;
    if (active) {
        int32_t base = W_offsets[enc], next = W_offsets[enc+1];
        int G = (next - base) / (EW * 4);
        ctx.x = W_states[stream_id];
        CTX_SET_SLAB_AVAIL(ctx, G, 0);
        ctx.buf_hi = 0; ctx.buf_lo = 0;
        ctx.block_base_u32 = base / 4;
    }
    unsigned active_mask = __ballot_sync(0xFFFFFFFF, active);
    if (!active) return;

    int N_seg = K / SEG_PER_ROW;
    int n_pairs = N_seg / 2;
    int n_tiles_total = n_pairs / TILE;
    int x_base = seg_idx * N_seg;
    int64_t n_streams = (int64_t)M * SEG_PER_ROW;

    float acc = 0.0f;

    for (int tile_idx = 0; tile_idx < n_tiles_total; tile_idx++) {
        {
            bool need = (CTX_BUF_AVAIL(ctx) == 0);
            if (__ballot_sync(active_mask, need)) {
                if (need) { REFILL_BUF(ctx, W_compressed, EW, tid); }
            }
        }

        int64_t sm_base = ((int64_t)tile_idx * n_streams + stream_id) * (int64_t)TILE;
        constexpr int SM_CHUNKS = TILE / 8;
        uint64_t sm_chunks[SM_CHUNKS];
        #pragma unroll
        for (int c = 0; c < SM_CHUNKS; c++) {
            sm_chunks[c] = *reinterpret_cast<const uint64_t*>(
                W_sm_tiled + sm_base + c * 8);
        }

        #pragma unroll
        for (int t = 0; t < TILE; t++) {
            uint32_t ent = sfc_smem[ctx.x & (M_SIZE-1)];
            uint8_t sym = ent & 0xFF;
            uint32_t f = (ent>>8)&0xFFF, c = (ent>>20)&0xFFF;
            uint32_t sl = ctx.x & (M_SIZE-1);
            ctx.x = (ctx.x >> M_LOG) * f + (sl - c);
            bool need = (ctx.x < L_LOW);
            if (need && CTX_BUF_AVAIL(ctx) == 0) {
                REFILL_BUF(ctx, W_compressed, EW, tid);
            }
            uint32_t xr = (ctx.x<<16) | (ctx.buf_hi>>16);
            uint32_t bh = (ctx.buf_hi<<16) | (ctx.buf_lo>>16);
            uint32_t bl = ctx.buf_lo<<16;
            int av = CTX_BUF_AVAIL(ctx) - 1;
            ctx.x      = need ? xr : ctx.x;
            ctx.buf_hi = need ? bh : ctx.buf_hi;
            ctx.buf_lo = need ? bl : ctx.buf_lo;
            CTX_SET_SLAB_AVAIL(ctx, CTX_SLAB_IDX(ctx),
                               need ? av : CTX_BUF_AVAIL(ctx));
            decoded[t * blockDim.x + threadIdx.x] = sym;
        }
        __syncthreads();

        #pragma unroll
        for (int t = 0; t < TILE; t++) {
            uint8_t sym    = decoded[t * blockDim.x + threadIdx.x];
            uint8_t sm_val = (uint8_t)(sm_chunks[t / 8] >> ((t % 8) * 8));
            uint8_t w0_b = ((sm_val & 8) << 4) | ((sym >> 4) << 3) | (sm_val & 7);
            uint8_t w1_b = (((sm_val >> 4) & 8) << 4) | ((sym & 0xF) << 3) | ((sm_val >> 4) & 7);
            __nv_fp8_e4m3 w0, w1;
            memcpy(&w0, &w0_b, 1);
            memcpy(&w1, &w1_b, 1);

            int k_in_seg = (tile_idx * TILE + t) * 2;
            acc += float(w0) * __half2float(x_smem[x_base + k_in_seg]);
            acc += float(w1) * __half2float(x_smem[x_base + k_in_seg + 1]);
        }
        __syncthreads();
    }

    // Warp-level reduction: 32 threads per row → single sum.
    #pragma unroll
    for (int offset = SEG_PER_ROW / 2; offset > 0; offset >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);

    if (seg_idx == 0)
        y[m] = __float2half(acc);
}

torch::Tensor fp8_gemv_fused(
    torch::Tensor W_compressed, torch::Tensor W_offsets, torch::Tensor W_states,
    torch::Tensor W_sm_tiled,
    torch::Tensor sfc_table,
    torch::Tensor x,
    int64_t K)
{
    TORCH_CHECK(W_compressed.is_cuda() && W_compressed.dtype() == torch::kUInt8);
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf);
    TORCH_CHECK((int64_t)x.size(0) == K);

    int64_t n_streams = W_states.numel();
    TORCH_CHECK(n_streams % SEG_PER_ROW == 0, "n_streams must be divisible by SEG_PER_ROW");
    int64_t M = n_streams / SEG_PER_ROW;
    int64_t n_enc_blocks = n_streams / BLOCK_STREAMS;
    constexpr int E = 4, T = 8;
    TORCH_CHECK(n_enc_blocks % E == 0, "n_enc_blocks must be a multiple of EPB=4");
    int N_seg = (int)K / SEG_PER_ROW;
    TORCH_CHECK(N_seg % (2 * T) == 0, "N_seg must be a multiple of 2*TILE=16");

    auto y = torch::empty({M},
        torch::TensorOptions().dtype(torch::kHalf).device(W_compressed.device()));

    size_t smem_bytes = 4096 * 4
                      + (size_t)T * BLOCK_STREAMS * E
                      + K * sizeof(__half);
    TORCH_CHECK(smem_bytes <= 99 * 1024, "K too large for SMEM (first pass)");
    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(fp8_gemv_fused_kernel<E, T>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    }

    fp8_gemv_fused_kernel<E, T><<<n_enc_blocks / E, BLOCK_STREAMS * E, smem_bytes>>>(
        reinterpret_cast<const uint32_t*>(W_compressed.data_ptr<uint8_t>()),
        W_offsets.data_ptr<int32_t>(),
        reinterpret_cast<const uint32_t*>(W_states.data_ptr<int32_t>()),
        W_sm_tiled.data_ptr<uint8_t>(),
        reinterpret_cast<const uint32_t*>(sfc_table.data_ptr<int32_t>()),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        (int)M, (int)K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

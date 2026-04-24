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
// x is read directly from global memory through the read-only cache path.

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <cstdlib>

// rANS constants — must match the encoder and vecadd.cu.
namespace {
constexpr int      M_LOG         = 12;
constexpr uint32_t M_SIZE        = 1u << M_LOG;
constexpr uint32_t L_LOW         = 1u << 16;
constexpr int      BLOCK_STREAMS = 128;

int getenv_int(const char* name, int default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return default_value;
    return std::atoi(value);
}

bool getenv_is_set(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0';
}

int choose_auto_epb(int64_t M, int64_t n_enc_blocks) {
    int epb = (M >= 8192) ? 8 : 4;
    while (epb > 1 && (n_enc_blocks % epb) != 0) {
        epb >>= 1;
    }
    return epb;
}

int choose_auto_rpt(int64_t M) {
    return (M >= 8192) ? 2 : 1;
}

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
            for (int p = 0; p < 2; p++) {
                __nv_fp8x2_e4m3 w_pair;
                w_pair.__x = (__nv_fp8x2_storage_t)((word >> (p * 16)) & 0xFFFF);
                float2 wf = static_cast<float2>(w_pair);
                acc += wf.x * __half2float(x[k_base + w_off * 4 + p * 2]);
                acc += wf.y * __half2float(x[k_base + w_off * 4 + p * 2 + 1]);
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

    // 128 threads/block = 4 warps/block = 4 output rows/block. Smaller
    // blocks improve scheduling granularity for this one-warp-per-row kernel.
    const int threads = 128;
    int warps_per_block = threads / 32;
    int blocks = (M + warps_per_block - 1) / warps_per_block;
    fp8_gemv_raw_kernel<<<blocks, threads>>>(
        reinterpret_cast<const uint4*>(W.data_ptr<uint8_t>()),
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        M, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

template <int B>
__global__ void fp8_gemv_raw_batch_kernel(
    const uint4*   __restrict__ W,    // [M, K/16] uint4 view of [M, K] FP8
    const __half*  __restrict__ x,    // [B, K]
    __half*        __restrict__ y,    // [B, M]
    int M, int K)
{
    int warps_per_block = blockDim.x / 32;
    int warp_in_block   = threadIdx.x / 32;
    int lane            = threadIdx.x & 31;
    int m = (int)blockIdx.x * warps_per_block + warp_in_block;
    if (m >= M) return;

    int n_u4      = K / 16;
    int n_u4_lane = n_u4 / 32;

    float acc[B];
    #pragma unroll
    for (int b = 0; b < B; b++) {
        acc[b] = 0.0f;
    }

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
            for (int p = 0; p < 2; p++) {
                __nv_fp8x2_e4m3 w_pair;
                w_pair.__x = (__nv_fp8x2_storage_t)((word >> (p * 16)) & 0xFFFF);
                float2 wf = static_cast<float2>(w_pair);
                int k = k_base + w_off * 4 + p * 2;
                #pragma unroll
                for (int b = 0; b < B; b++) {
                    const __half* xb = x + (int64_t)b * K;
                    acc[b] += wf.x * __half2float(__ldg(xb + k));
                    acc[b] += wf.y * __half2float(__ldg(xb + k + 1));
                }
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int b = 0; b < B; b++) {
            acc[b] += __shfl_xor_sync(0xFFFFFFFF, acc[b], offset);
        }
    }

    if (lane == 0) {
        #pragma unroll
        for (int b = 0; b < B; b++) {
            y[(int64_t)b * M + m] = __float2half(acc[b]);
        }
    }
}

torch::Tensor fp8_gemv_raw_batch(torch::Tensor W, torch::Tensor x) {
    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kUInt8);
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf);
    TORCH_CHECK(W.dim() == 2 && x.dim() == 2);
    TORCH_CHECK(W.is_contiguous() && x.is_contiguous(), "contiguous inputs required");
    int M = (int)W.size(0), K = (int)W.size(1);
    int B = (int)x.size(0);
    TORCH_CHECK((int)x.size(1) == K);
    TORCH_CHECK(B == 2 || B == 4 || B == 8, "B must be one of 2, 4, 8");
    TORCH_CHECK(K % 512 == 0, "K must be a multiple of 512 (first pass)");

    auto y = torch::empty({B, M},
        torch::TensorOptions().dtype(torch::kHalf).device(W.device()));

    const int threads = 128;
    int warps_per_block = threads / 32;
    int blocks = (M + warps_per_block - 1) / warps_per_block;

#define LAUNCH_RAW_BATCH(B_VALUE) do {                                                \
    fp8_gemv_raw_batch_kernel<B_VALUE><<<blocks, threads>>>(                          \
        reinterpret_cast<const uint4*>(W.data_ptr<uint8_t>()),                        \
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),                      \
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),                            \
        M, K);                                                                        \
} while (0)

    if      (B == 2) { LAUNCH_RAW_BATCH(2); }
    else if (B == 4) { LAUNCH_RAW_BATCH(4); }
    else             { LAUNCH_RAW_BATCH(8); }

#undef LAUNCH_RAW_BATCH

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

__device__ __forceinline__ void refill_rans_buf(
    const uint32_t* __restrict__ W_compressed,
    int tid,
    int& slab_idx,
    int& buf_avail,
    uint32_t& buf_hi,
    uint32_t& buf_lo,
    int32_t block_base_u32)
{
    constexpr int EW = BLOCK_STREAMS;
    int refill_sl = slab_idx - 2;
    int32_t off = block_base_u32 + refill_sl * EW + tid;
    buf_lo = W_compressed[off];
    buf_hi = W_compressed[off + EW];
    slab_idx = refill_sl;
    buf_avail = 4;
}

template <int TILE>
__device__ __forceinline__ void decode_accumulate_symbol(
    const uint32_t* __restrict__ W_compressed,
    const uint32_t* __restrict__ sfc,
    const __half* __restrict__ x_smem,
    int tid,
    int seg_idx,
    int t,
    uint8_t sm_val,
    uint32_t& rans_x,
    uint32_t& buf_hi,
    uint32_t& buf_lo,
    int& slab_idx,
    int& buf_avail,
    int32_t block_base_u32,
    float& acc)
{
    uint32_t x0 = rans_x;
    uint32_t sl = x0 & (M_SIZE - 1);
    uint32_t ent = __ldg(sfc + sl);
    uint8_t sym = ent & 0xFF;
    uint32_t f = (ent >> 8) & 0xFFF;
    uint32_t c = (ent >> 20) & 0xFFF;
    uint32_t x_new = (x0 >> M_LOG) * f + (sl - c);
    bool need = x_new < L_LOW;
    if (need && buf_avail == 0) {
        refill_rans_buf(W_compressed, tid, slab_idx, buf_avail,
                        buf_hi, buf_lo, block_base_u32);
    }
    uint32_t xr = (x_new << 16) | (buf_hi >> 16);
    uint32_t bh = (buf_hi << 16) | (buf_lo >> 16);
    uint32_t bl = buf_lo << 16;
    int av = buf_avail - 1;
    rans_x = need ? xr : x_new;
    buf_hi = need ? bh : buf_hi;
    buf_lo = need ? bl : buf_lo;
    buf_avail = need ? av : buf_avail;

    uint8_t w0_b = ((sm_val & 8) << 4) | ((sym >> 4) << 3) | (sm_val & 7);
    uint8_t w1_b = (((sm_val >> 4) & 8) << 4) | ((sym & 0xF) << 3) | ((sm_val >> 4) & 7);
    __nv_fp8_e4m3 w0;
    __nv_fp8_e4m3 w1;
    w0.__x = (__nv_fp8_storage_t)w0_b;
    w1.__x = (__nv_fp8_storage_t)w1_b;
    float2 wf = make_float2((float)w0, (float)w1);

    acc += wf.x * __half2float(x_smem[seg_idx * TILE * 2 + t * 2]);
    acc += wf.y * __half2float(x_smem[seg_idx * TILE * 2 + t * 2 + 1]);
}

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
    __half* x_smem = reinterpret_cast<__half*>(smem_raw);

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
    int N_seg = K / SEG_PER_ROW;
    int n_pairs = N_seg / 2;
    int n_tiles_total = n_pairs / TILE;
    int64_t n_streams = (int64_t)M * SEG_PER_ROW;

    float acc = 0.0f;

    for (int tile_idx = 0; tile_idx < n_tiles_total; tile_idx++) {
        for (int i = threadIdx.x; i < SEG_PER_ROW * TILE * 2; i += blockDim.x) {
            int s = i / (TILE * 2);
            int k = i - s * (TILE * 2);
            x_smem[i] = x[s * N_seg + tile_idx * TILE * 2 + k];
        }
        __syncthreads();

        if (active) {
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
            uint32_t ent = __ldg(sfc + (ctx.x & (M_SIZE-1)));
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

            uint8_t sm_val = (uint8_t)(sm_chunks[t / 8] >> ((t % 8) * 8));
            uint8_t w0_b = ((sm_val & 8) << 4) | ((sym >> 4) << 3) | (sm_val & 7);
            uint8_t w1_b = (((sm_val >> 4) & 8) << 4) | ((sym & 0xF) << 3) | ((sm_val >> 4) & 7);
            __nv_fp8_e4m3 w0;
            __nv_fp8_e4m3 w1;
            w0.__x = (__nv_fp8_storage_t)w0_b;
            w1.__x = (__nv_fp8_storage_t)w1_b;
            float2 wf = make_float2((float)w0, (float)w1);

            acc += wf.x * __half2float(x_smem[seg_idx * TILE * 2 + t * 2]);
            acc += wf.y * __half2float(x_smem[seg_idx * TILE * 2 + t * 2 + 1]);
        }
        }
        __syncthreads();
    }

    // Warp-level reduction: 32 threads per row → single sum.
    #pragma unroll
    for (int offset = SEG_PER_ROW / 2; offset > 0; offset >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);

    if (active && seg_idx == 0)
        y[m] = __float2half(acc);
}

template <int EPB, int TILE>
__global__ void fp8_gemv_fused_rpt2_kernel(
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
    constexpr int RPT = 2;
    constexpr int THREADS_PER_EB = EW / RPT;
    static_assert(TILE % 8 == 0, "TILE must be a multiple of 8");
    static_assert(ROWS_PER_EB % RPT == 0, "RPT must divide rows per encoder block");

    extern __shared__ uint8_t smem_raw[];
    __half* x_smem = reinterpret_cast<__half*>(smem_raw);

    int enc_group = threadIdx.x / THREADS_PER_EB;
    int local_tid = threadIdx.x % THREADS_PER_EB;
    int enc       = EPB * (int)blockIdx.x + enc_group;

    int row_pair = local_tid / SEG_PER_ROW;      // 0..1
    int seg_idx  = local_tid % SEG_PER_ROW;
    int row0     = row_pair * RPT;
    int row1     = row0 + 1;
    int tid0     = row0 * SEG_PER_ROW + seg_idx;
    int tid1     = row1 * SEG_PER_ROW + seg_idx;
    int m0       = enc * ROWS_PER_EB + row0;
    int m1       = m0 + 1;
    int32_t stream0 = enc * EW + tid0;
    int32_t stream1 = enc * EW + tid1;
    bool active0 = m0 < M;
    bool active1 = m1 < M;
    bool active = active0 || active1;

    uint32_t rans_x0 = 0, rans_x1 = 0;
    uint32_t buf_hi0 = 0, buf_hi1 = 0;
    uint32_t buf_lo0 = 0, buf_lo1 = 0;
    int slab_idx0 = 0, slab_idx1 = 0;
    int buf_avail0 = 0, buf_avail1 = 0;
    int32_t block_base_u32 = 0;
    if (active) {
        int32_t base = W_offsets[enc], next = W_offsets[enc+1];
        int G = (next - base) / (EW * 4);
        block_base_u32 = base / 4;
        if (active0) {
            rans_x0 = W_states[stream0];
            slab_idx0 = G;
        }
        if (active1) {
            rans_x1 = W_states[stream1];
            slab_idx1 = G;
        }
    }

    int N_seg = K / SEG_PER_ROW;
    int n_pairs = N_seg / 2;
    int n_tiles_total = n_pairs / TILE;
    int64_t n_streams = (int64_t)M * SEG_PER_ROW;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int tile_idx = 0; tile_idx < n_tiles_total; tile_idx++) {
        for (int i = threadIdx.x; i < SEG_PER_ROW * TILE * 2; i += blockDim.x) {
            int s = i / (TILE * 2);
            int k = i - s * (TILE * 2);
            x_smem[i] = x[s * N_seg + tile_idx * TILE * 2 + k];
        }
        __syncthreads();

        if (active0 && buf_avail0 == 0) {
            refill_rans_buf(W_compressed, tid0, slab_idx0, buf_avail0,
                            buf_hi0, buf_lo0, block_base_u32);
        }
        if (active1 && buf_avail1 == 0) {
            refill_rans_buf(W_compressed, tid1, slab_idx1, buf_avail1,
                            buf_hi1, buf_lo1, block_base_u32);
        }

        int64_t tile_base = (int64_t)tile_idx * n_streams;
        int64_t sm_base0 = (tile_base + stream0) * (int64_t)TILE;
        int64_t sm_base1 = (tile_base + stream1) * (int64_t)TILE;
        constexpr int SM_CHUNKS = TILE / 8;
        uint64_t sm_chunks0[SM_CHUNKS];
        uint64_t sm_chunks1[SM_CHUNKS];
        if (active0) {
            #pragma unroll
            for (int c = 0; c < SM_CHUNKS; c++) {
                sm_chunks0[c] = *reinterpret_cast<const uint64_t*>(
                    W_sm_tiled + sm_base0 + c * 8);
            }
        }
        if (active1) {
            #pragma unroll
            for (int c = 0; c < SM_CHUNKS; c++) {
                sm_chunks1[c] = *reinterpret_cast<const uint64_t*>(
                    W_sm_tiled + sm_base1 + c * 8);
            }
        }

        #pragma unroll
        for (int t = 0; t < TILE; t++) {
            if (active0) {
                uint8_t sm_val0 = (uint8_t)(sm_chunks0[t / 8] >> ((t % 8) * 8));
                decode_accumulate_symbol<TILE>(
                    W_compressed, sfc, x_smem, tid0, seg_idx, t, sm_val0,
                    rans_x0, buf_hi0, buf_lo0, slab_idx0, buf_avail0,
                    block_base_u32, acc0);
            }
            if (active1) {
                uint8_t sm_val1 = (uint8_t)(sm_chunks1[t / 8] >> ((t % 8) * 8));
                decode_accumulate_symbol<TILE>(
                    W_compressed, sfc, x_smem, tid1, seg_idx, t, sm_val1,
                    rans_x1, buf_hi1, buf_lo1, slab_idx1, buf_avail1,
                    block_base_u32, acc1);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int offset = SEG_PER_ROW / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_xor_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_xor_sync(0xFFFFFFFF, acc1, offset);
    }

    if (seg_idx == 0) {
        if (active0) y[m0] = __float2half(acc0);
        if (active1) y[m1] = __float2half(acc1);
    }
}

template <int EPB, int TILE, int B>
__global__ void fp8_gemv_fused_batch_kernel(
    const uint32_t* __restrict__ W_compressed,
    const int32_t*  __restrict__ W_offsets,
    const uint32_t* __restrict__ W_states,
    const uint8_t*  __restrict__ W_sm_tiled,   // [n_tiles, M*SEG_PER_ROW, TILE]
    const uint32_t* __restrict__ sfc,
    const __half*   __restrict__ x,            // [B, K]
    __half*         __restrict__ y,            // [B, M]
    int M, int K)
{
    constexpr int EW = BLOCK_STREAMS;
    constexpr int ROWS_PER_EB = EW / SEG_PER_ROW;
    constexpr int X_TILE_ELEMS = SEG_PER_ROW * TILE * 2;
    static_assert(TILE % 8 == 0, "TILE must be a multiple of 8");
    static_assert(EW % SEG_PER_ROW == 0, "BLOCK_STREAMS must be divisible by SEG_PER_ROW");

    extern __shared__ uint8_t smem_raw[];
    __half* x_smem = reinterpret_cast<__half*>(smem_raw);

    int enc_group = threadIdx.x / EW;
    int tid       = threadIdx.x % EW;
    int enc       = EPB * (int)blockIdx.x + enc_group;

    int row_in_eb = tid / SEG_PER_ROW;
    int seg_idx   = tid % SEG_PER_ROW;
    int m         = enc * ROWS_PER_EB + row_in_eb;
    int32_t stream_id = enc * EW + tid;
    bool active = m < M;

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
    int N_seg = K / SEG_PER_ROW;
    int n_pairs = N_seg / 2;
    int n_tiles_total = n_pairs / TILE;
    int64_t n_streams = (int64_t)M * SEG_PER_ROW;

    float acc[B];
    #pragma unroll
    for (int b = 0; b < B; b++) {
        acc[b] = 0.0f;
    }

    for (int tile_idx = 0; tile_idx < n_tiles_total; tile_idx++) {
        for (int i = threadIdx.x; i < B * X_TILE_ELEMS; i += blockDim.x) {
            int b = i / X_TILE_ELEMS;
            int local = i - b * X_TILE_ELEMS;
            int s = local / (TILE * 2);
            int k = local - s * (TILE * 2);
            x_smem[i] = x[(int64_t)b * K + s * N_seg + tile_idx * TILE * 2 + k];
        }
        __syncthreads();

        if (active) {
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
            uint32_t ent = __ldg(sfc + (ctx.x & (M_SIZE-1)));
            uint8_t sym = ent & 0xFF;
            uint32_t f = (ent >> 8) & 0xFFF, c = (ent >> 20) & 0xFFF;
            uint32_t sl = ctx.x & (M_SIZE-1);
            ctx.x = (ctx.x >> M_LOG) * f + (sl - c);
            bool need = (ctx.x < L_LOW);
            if (need && CTX_BUF_AVAIL(ctx) == 0) {
                REFILL_BUF(ctx, W_compressed, EW, tid);
            }
            uint32_t xr = (ctx.x << 16) | (ctx.buf_hi >> 16);
            uint32_t bh = (ctx.buf_hi << 16) | (ctx.buf_lo >> 16);
            uint32_t bl = ctx.buf_lo << 16;
            int av = CTX_BUF_AVAIL(ctx) - 1;
            ctx.x      = need ? xr : ctx.x;
            ctx.buf_hi = need ? bh : ctx.buf_hi;
            ctx.buf_lo = need ? bl : ctx.buf_lo;
            CTX_SET_SLAB_AVAIL(ctx, CTX_SLAB_IDX(ctx),
                               need ? av : CTX_BUF_AVAIL(ctx));

            uint8_t sm_val = (uint8_t)(sm_chunks[t / 8] >> ((t % 8) * 8));
            uint8_t w0_b = ((sm_val & 8) << 4) | ((sym >> 4) << 3) | (sm_val & 7);
            uint8_t w1_b = (((sm_val >> 4) & 8) << 4) | ((sym & 0xF) << 3) | ((sm_val >> 4) & 7);
            __nv_fp8x2_e4m3 w_pair;
            w_pair.__x = (__nv_fp8x2_storage_t)((uint16_t)w0_b | ((uint16_t)w1_b << 8));
            float2 wf = static_cast<float2>(w_pair);
            int x_base = seg_idx * TILE * 2 + t * 2;

            #pragma unroll
            for (int b = 0; b < B; b++) {
                const __half* x_tile = x_smem + b * X_TILE_ELEMS;
                acc[b] += wf.x * __half2float(x_tile[x_base]);
                acc[b] += wf.y * __half2float(x_tile[x_base + 1]);
            }
        }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int offset = SEG_PER_ROW / 2; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int b = 0; b < B; b++) {
            acc[b] += __shfl_xor_sync(0xFFFFFFFF, acc[b], offset);
        }
    }

    if (active && seg_idx == 0) {
        #pragma unroll
        for (int b = 0; b < B; b++) {
            y[(int64_t)b * M + m] = __float2half(acc[b]);
        }
    }
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
    TORCH_CHECK(x.dim() == 1 || x.dim() == 2, "x must have shape [K] or [B, K]");
    int64_t B = (x.dim() == 1) ? 1 : x.size(0);
    int64_t x_k = (x.dim() == 1) ? x.size(0) : x.size(1);
    TORCH_CHECK(x_k == K);
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int64_t n_streams = W_states.numel();
    TORCH_CHECK(n_streams % SEG_PER_ROW == 0, "n_streams must be divisible by SEG_PER_ROW");
    int64_t M = n_streams / SEG_PER_ROW;
    int64_t n_enc_blocks = n_streams / BLOCK_STREAMS;
    bool epb_env_set = getenv_is_set("RANS_GEMV_EPB");
    bool tile_env_set = getenv_is_set("RANS_GEMV_TILE");
    bool rpt_env_set = getenv_is_set("RANS_GEMV_RPT");
    int epb = epb_env_set ? getenv_int("RANS_GEMV_EPB", 2) : choose_auto_epb(M, n_enc_blocks);
    int tile = tile_env_set ? getenv_int("RANS_GEMV_TILE", 8) : 8;
    int rpt = rpt_env_set ? getenv_int("RANS_GEMV_RPT", 1) : choose_auto_rpt(M);
    TORCH_CHECK(epb == 1 || epb == 2 || epb == 4 || epb == 8,
                "RANS_GEMV_EPB must be one of 1, 2, 4, 8");
    TORCH_CHECK(tile == 8 || tile == 16 || tile == 32,
                "RANS_GEMV_TILE must be one of 8, 16, 32");
    TORCH_CHECK(rpt == 1 || rpt == 2, "RANS_GEMV_RPT must be one of 1, 2");
    TORCH_CHECK(n_enc_blocks % epb == 0, "n_enc_blocks must be a multiple of RANS_GEMV_EPB");
    int N_seg = (int)K / SEG_PER_ROW;
    TORCH_CHECK(N_seg % (2 * tile) == 0, "N_seg must be a multiple of 2*RANS_GEMV_TILE");

    auto y = (x.dim() == 1)
        ? torch::empty({M},
            torch::TensorOptions().dtype(torch::kHalf).device(W_compressed.device()))
        : torch::empty({B, M},
            torch::TensorOptions().dtype(torch::kHalf).device(W_compressed.device()));

    size_t smem_bytes = SEG_PER_ROW * tile * 2 * sizeof(__half);
    TORCH_CHECK(smem_bytes <= 99 * 1024, "dynamic SMEM request is too large");
#define LAUNCH_FUSED(EPB_VALUE, TILE_VALUE) do {                                      \
    if (smem_bytes > 48 * 1024) {                                                     \
        cudaFuncSetAttribute(fp8_gemv_fused_kernel<EPB_VALUE, TILE_VALUE>,            \
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);            \
    }                                                                                 \
    fp8_gemv_fused_kernel<EPB_VALUE, TILE_VALUE>                                      \
        <<<n_enc_blocks / (EPB_VALUE), BLOCK_STREAMS * (EPB_VALUE), smem_bytes>>>(    \
            reinterpret_cast<const uint32_t*>(W_compressed.data_ptr<uint8_t>()),      \
            W_offsets.data_ptr<int32_t>(),                                            \
            reinterpret_cast<const uint32_t*>(W_states.data_ptr<int32_t>()),          \
            W_sm_tiled.data_ptr<uint8_t>(),                                           \
            reinterpret_cast<const uint32_t*>(sfc_table.data_ptr<int32_t>()),         \
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),                  \
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),                        \
            (int)M, (int)K);                                                          \
} while (0)

#define LAUNCH_FUSED_RPT2(EPB_VALUE, TILE_VALUE) do {                                 \
    if (smem_bytes > 48 * 1024) {                                                     \
        cudaFuncSetAttribute(fp8_gemv_fused_rpt2_kernel<EPB_VALUE, TILE_VALUE>,       \
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);            \
    }                                                                                 \
    fp8_gemv_fused_rpt2_kernel<EPB_VALUE, TILE_VALUE>                                 \
        <<<n_enc_blocks / (EPB_VALUE), (BLOCK_STREAMS / 2) * (EPB_VALUE), smem_bytes>>>( \
            reinterpret_cast<const uint32_t*>(W_compressed.data_ptr<uint8_t>()),      \
            W_offsets.data_ptr<int32_t>(),                                            \
            reinterpret_cast<const uint32_t*>(W_states.data_ptr<int32_t>()),          \
            W_sm_tiled.data_ptr<uint8_t>(),                                           \
            reinterpret_cast<const uint32_t*>(sfc_table.data_ptr<int32_t>()),         \
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),                  \
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),                        \
            (int)M, (int)K);                                                          \
} while (0)

#define LAUNCH_FUSED_BATCH(EPB_VALUE, TILE_VALUE, B_VALUE) do {                       \
    size_t batch_smem_bytes = (size_t)(B_VALUE) * SEG_PER_ROW * (TILE_VALUE) * 2 * sizeof(__half); \
    if (batch_smem_bytes > 48 * 1024) {                                               \
        cudaFuncSetAttribute(fp8_gemv_fused_batch_kernel<EPB_VALUE, TILE_VALUE, B_VALUE>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)batch_smem_bytes);      \
    }                                                                                 \
    fp8_gemv_fused_batch_kernel<EPB_VALUE, TILE_VALUE, B_VALUE>                       \
        <<<n_enc_blocks / (EPB_VALUE), BLOCK_STREAMS * (EPB_VALUE), batch_smem_bytes>>>( \
            reinterpret_cast<const uint32_t*>(W_compressed.data_ptr<uint8_t>()),      \
            W_offsets.data_ptr<int32_t>(),                                            \
            reinterpret_cast<const uint32_t*>(W_states.data_ptr<int32_t>()),          \
            W_sm_tiled.data_ptr<uint8_t>(),                                           \
            reinterpret_cast<const uint32_t*>(sfc_table.data_ptr<int32_t>()),         \
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),                  \
            reinterpret_cast<__half*>(y.data_ptr<at::Half>()),                        \
            (int)M, (int)K);                                                          \
} while (0)

    if (x.dim() == 2) {
        TORCH_CHECK(B == 2 || B == 4 || B == 8, "batched x currently supports B in {2, 4, 8}");
        size_t batch_smem_bytes = (size_t)B * SEG_PER_ROW * tile * 2 * sizeof(__half);
        TORCH_CHECK(batch_smem_bytes <= 99 * 1024, "dynamic SMEM request is too large");
        if (B == 2) {
            if (tile == 8) {
                if      (epb == 1) { LAUNCH_FUSED_BATCH(1, 8, 2); }
                else if (epb == 2) { LAUNCH_FUSED_BATCH(2, 8, 2); }
                else if (epb == 4) { LAUNCH_FUSED_BATCH(4, 8, 2); }
                else               { LAUNCH_FUSED_BATCH(8, 8, 2); }
            } else if (tile == 16) {
                if      (epb == 1) { LAUNCH_FUSED_BATCH(1, 16, 2); }
                else if (epb == 2) { LAUNCH_FUSED_BATCH(2, 16, 2); }
                else if (epb == 4) { LAUNCH_FUSED_BATCH(4, 16, 2); }
                else               { LAUNCH_FUSED_BATCH(8, 16, 2); }
            } else {
                if      (epb == 1) { LAUNCH_FUSED_BATCH(1, 32, 2); }
                else if (epb == 2) { LAUNCH_FUSED_BATCH(2, 32, 2); }
                else if (epb == 4) { LAUNCH_FUSED_BATCH(4, 32, 2); }
                else               { LAUNCH_FUSED_BATCH(8, 32, 2); }
            }
        } else if (B == 4) {
            if (tile == 8) {
                if      (epb == 1) { LAUNCH_FUSED_BATCH(1, 8, 4); }
                else if (epb == 2) { LAUNCH_FUSED_BATCH(2, 8, 4); }
                else if (epb == 4) { LAUNCH_FUSED_BATCH(4, 8, 4); }
                else               { LAUNCH_FUSED_BATCH(8, 8, 4); }
            } else if (tile == 16) {
                if      (epb == 1) { LAUNCH_FUSED_BATCH(1, 16, 4); }
                else if (epb == 2) { LAUNCH_FUSED_BATCH(2, 16, 4); }
                else if (epb == 4) { LAUNCH_FUSED_BATCH(4, 16, 4); }
                else               { LAUNCH_FUSED_BATCH(8, 16, 4); }
            } else {
                if      (epb == 1) { LAUNCH_FUSED_BATCH(1, 32, 4); }
                else if (epb == 2) { LAUNCH_FUSED_BATCH(2, 32, 4); }
                else if (epb == 4) { LAUNCH_FUSED_BATCH(4, 32, 4); }
                else               { LAUNCH_FUSED_BATCH(8, 32, 4); }
            }
        } else {
            if (tile == 8) {
                if      (epb == 1) { LAUNCH_FUSED_BATCH(1, 8, 8); }
                else if (epb == 2) { LAUNCH_FUSED_BATCH(2, 8, 8); }
                else if (epb == 4) { LAUNCH_FUSED_BATCH(4, 8, 8); }
                else               { LAUNCH_FUSED_BATCH(8, 8, 8); }
            } else if (tile == 16) {
                if      (epb == 1) { LAUNCH_FUSED_BATCH(1, 16, 8); }
                else if (epb == 2) { LAUNCH_FUSED_BATCH(2, 16, 8); }
                else if (epb == 4) { LAUNCH_FUSED_BATCH(4, 16, 8); }
                else               { LAUNCH_FUSED_BATCH(8, 16, 8); }
            } else {
                if      (epb == 1) { LAUNCH_FUSED_BATCH(1, 32, 8); }
                else if (epb == 2) { LAUNCH_FUSED_BATCH(2, 32, 8); }
                else if (epb == 4) { LAUNCH_FUSED_BATCH(4, 32, 8); }
                else               { LAUNCH_FUSED_BATCH(8, 32, 8); }
            }
        }
    } else if (rpt == 2) {
        if (tile == 8) {
            if      (epb == 1) { LAUNCH_FUSED_RPT2(1, 8); }
            else if (epb == 2) { LAUNCH_FUSED_RPT2(2, 8); }
            else if (epb == 4) { LAUNCH_FUSED_RPT2(4, 8); }
            else               { LAUNCH_FUSED_RPT2(8, 8); }
        } else if (tile == 16) {
            if      (epb == 1) { LAUNCH_FUSED_RPT2(1, 16); }
            else if (epb == 2) { LAUNCH_FUSED_RPT2(2, 16); }
            else if (epb == 4) { LAUNCH_FUSED_RPT2(4, 16); }
            else               { LAUNCH_FUSED_RPT2(8, 16); }
        } else {
            if      (epb == 1) { LAUNCH_FUSED_RPT2(1, 32); }
            else if (epb == 2) { LAUNCH_FUSED_RPT2(2, 32); }
            else if (epb == 4) { LAUNCH_FUSED_RPT2(4, 32); }
            else               { LAUNCH_FUSED_RPT2(8, 32); }
        }
    } else {
        if (tile == 8) {
            if      (epb == 1) { LAUNCH_FUSED(1, 8); }
            else if (epb == 2) { LAUNCH_FUSED(2, 8); }
            else if (epb == 4) { LAUNCH_FUSED(4, 8); }
            else               { LAUNCH_FUSED(8, 8); }
        } else if (tile == 16) {
            if      (epb == 1) { LAUNCH_FUSED(1, 16); }
            else if (epb == 2) { LAUNCH_FUSED(2, 16); }
            else if (epb == 4) { LAUNCH_FUSED(4, 16); }
            else               { LAUNCH_FUSED(8, 16); }
        } else {
            if      (epb == 1) { LAUNCH_FUSED(1, 32); }
            else if (epb == 2) { LAUNCH_FUSED(2, 32); }
            else if (epb == 4) { LAUNCH_FUSED(4, 32); }
            else               { LAUNCH_FUSED(8, 32); }
        }
    }

#undef LAUNCH_FUSED
#undef LAUNCH_FUSED_RPT2
#undef LAUNCH_FUSED_BATCH

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

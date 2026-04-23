// CPU rANS encoder. Produces the interleaved, block-oriented compressed
// layout that the GPU decoder in rans_decode.cu expects.
//
// Parameters (fixed for now, tuned to FP8 exponent compression):
//   M     = 2048        total frequency, 11-bit precision
//   L     = 2^16        state lower bound
//   b     = 2^16        uint16-level renormalization radix
//   state range: [L, bL) = [2^16, 2^32), fills uint32.
//
// Encoder flow: for each stream, process symbols in reverse order, spill
// renorm uint16 chunks (2 bytes each) to the per-stream output as state
// grows, emit final state. Then pack per-stream byte arrays into the
// GPU's (G_b, block_streams, 4) slab layout.
//
// Byte ordering: each spilled uint16 is written as [low, high]. In
// the slab's little-endian uint32 load, two consecutive uint16s occupy
// bits [0..15] (first-spilled) and [16..31] (second-spilled). The
// decoder consumes the MSB uint16 first, which is the last-spilled —
// the correct LIFO order for rANS.

#include <torch/extension.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <tuple>

namespace rans {

constexpr uint32_t L_LOW  = 1u << 16;      // 65536

// Encode one stream. Templated on symbol type to handle alphabets > 256.
template <typename SymT>
static std::tuple<uint32_t, std::vector<uint8_t>>
encode_with_cum(const SymT* symbols, int64_t n,
                const uint32_t* freqs, const uint32_t* cum,
                uint32_t M, int renorm_shift)
{
    uint32_t x = L_LOW;
    std::vector<uint8_t> stream;
    stream.reserve(static_cast<size_t>(n) + 16);

    for (int64_t i = n - 1; i >= 0; i--) {
        int s = (int)symbols[i];
        uint32_t f = freqs[s];
        uint32_t c = cum[s];

        uint32_t thresh = f << renorm_shift;
        while (x >= thresh) {
            stream.push_back(static_cast<uint8_t>(x & 0xFF));
            stream.push_back(static_cast<uint8_t>((x >> 8) & 0xFF));
            x >>= 16;
        }

        x = (x / f) * M + c + (x % f);
    }

    return {x, std::move(stream)};
}

}  // namespace rans

// Full encoder → interleaved-layout pipeline. Produces the exact layout
// the GPU decoder expects: concatenated blocks of (G_b, block_streams, 4)
// bytes, one block per `block_streams` streams. Each stream's compressed
// data is right-aligned in its block's slab array, with leading zero
// bytes padding the stream to a multiple of 4.
//
// Returns (compressed, final_states[K], block_offsets[n_blocks+1]).
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rans_encode_interleaved(torch::Tensor symbols, torch::Tensor freqs,
                        int64_t block_streams)
{
    TORCH_CHECK(symbols.dim() == 2, "symbols must be [K, N]");
    TORCH_CHECK(symbols.dtype() == torch::kUInt8 || symbols.dtype() == torch::kInt16,
                "symbols must be uint8 or int16");
    TORCH_CHECK(freqs.dtype() == torch::kInt32, "freqs must be int32");
    TORCH_CHECK(symbols.is_contiguous() && symbols.is_cpu(),
                "symbols must be contiguous CPU");
    TORCH_CHECK(freqs.is_contiguous() && freqs.is_cpu(),
                "freqs must be contiguous CPU");
    TORCH_CHECK(block_streams > 0, "block_streams must be positive");

    int64_t K = symbols.size(0);
    int64_t N = symbols.size(1);
    int64_t n_syms = freqs.numel();
    int64_t n_blocks = (K + block_streams - 1) / block_streams;

    const int32_t* f_ptr = freqs.data_ptr<int32_t>();
    const uint32_t* f_u32 = reinterpret_cast<const uint32_t*>(f_ptr);
    bool use_int16 = (symbols.dtype() == torch::kInt16);

    std::vector<uint32_t> cum_vec(n_syms);
    uint32_t acc = 0;
    for (int64_t i = 0; i < n_syms; i++) {
        TORCH_CHECK(f_ptr[i] >= 0, "freqs must be non-negative");
        cum_vec[i] = acc;
        acc += (uint32_t)f_ptr[i];
    }
    uint32_t M = acc;
    TORCH_CHECK(M > 0 && (M & (M - 1)) == 0,
                "frequencies must sum to a power of 2; got ", M);
    int m_log = 0;
    { uint32_t tmp = M; while (tmp > 1) { m_log++; tmp >>= 1; } }
    int renorm_shift = 32 - m_log;

    // Phase 1: per-stream encode. Streams are fully independent, so we
    // parallelize across CPU cores via OpenMP.
    std::vector<std::vector<uint8_t>> streams(K);
    std::vector<int32_t> padded_lens(K);
    std::vector<int32_t> states_vec(K);

    #pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < K; k++) {
        auto encode_stream = [&]() {
            if (use_int16)
                return rans::encode_with_cum(
                    symbols.data_ptr<int16_t>() + k * N, N,
                    f_u32, cum_vec.data(), M, renorm_shift);
            else
                return rans::encode_with_cum(
                    symbols.data_ptr<uint8_t>() + k * N, N,
                    f_u32, cum_vec.data(), M, renorm_shift);
        };
        auto result = encode_stream();
        states_vec[k]    = (int32_t)std::get<0>(result);
        streams[k]       = std::move(std::get<1>(result));
        int64_t raw_len  = (int64_t)streams[k].size();
        int64_t pad      = (4 - raw_len % 4) % 4;
        padded_lens[k]   = (int32_t)(raw_len + pad);
    }

    // Phase 2: per-block G_b (= max padded stream length in block / 4)
    // and cumulative byte offsets for each block.
    std::vector<int32_t> block_offsets_vec(n_blocks + 1);
    std::vector<int32_t> block_Gs(n_blocks);
    int64_t cum_offset = 0;
    for (int64_t b = 0; b < n_blocks; b++) {
        int64_t start = b * block_streams;
        int64_t end   = std::min(start + block_streams, K);
        int32_t max_len = 0;
        for (int64_t k = start; k < end; k++) {
            if (padded_lens[k] > max_len) max_len = padded_lens[k];
        }
        if (max_len == 0) max_len = 4;   // ensure at least 1 slab
        int32_t G_b = max_len / 4;
        // Pad G_b to even: GPU decoder refills 2 slabs at a time to reduce
        // per-lane refill divergence for global-memory coalescing. Overhead
        // <1 slab per block = 512 B per block, negligible.
        if (G_b & 1) G_b++;
        block_Gs[b] = G_b;
        block_offsets_vec[b] = (int32_t)cum_offset;
        cum_offset += (int64_t)G_b * block_streams * 4;
    }
    block_offsets_vec[n_blocks] = (int32_t)cum_offset;

    // Phase 3: allocate output and interleave. Each stream's groups go
    // into the TOP slabs of its block (slab g → G_b - G_s + g), bytes at
    // offset 4*tid within the slab.
    auto compressed = torch::zeros({cum_offset}, torch::kUInt8);
    uint8_t* out = compressed.data_ptr<uint8_t>();

    #pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < K; k++) {
        int64_t b   = k / block_streams;
        int64_t tid = k % block_streams;
        int32_t G_b = block_Gs[b];
        int32_t L_pad = padded_lens[k];
        int32_t G_s = L_pad / 4;
        if (G_s == 0) continue;

        int32_t pad = L_pad - (int32_t)streams[k].size();
        int32_t block_base = block_offsets_vec[b];

        const uint8_t* src = streams[k].data();
        int64_t src_pos = 0;

        for (int32_t g = 0; g < G_s; g++) {
            int32_t slab_idx = G_b - G_s + g;
            int64_t dst = (int64_t)block_base
                        + (int64_t)slab_idx * block_streams * 4
                        + tid * 4;
            for (int j = 0; j < 4; j++) {
                int32_t byte_pos = g * 4 + j;
                // First `pad` bytes of the stream-within-block are zero
                // (leading zero padding). The rest come from `src`.
                if (byte_pos >= pad) {
                    out[dst + j] = src[src_pos++];
                }
            }
        }
    }

    auto states_t = torch::from_blob(
        states_vec.data(), {K}, torch::TensorOptions().dtype(torch::kInt32)
    ).clone();
    auto offsets_t = torch::from_blob(
        block_offsets_vec.data(), {n_blocks + 1},
        torch::TensorOptions().dtype(torch::kInt32)
    ).clone();

    return {compressed, states_t, offsets_t};
}

// ─── tANS encoder ───────────────────────────────────────────────────
//
// tANS encoding: for each symbol, normalize state to [f, 2f) by
// outputting bits (MSB-first), then map to table position.
// Output is a packed bitstream (LSB-first in bytes) per stream.

namespace tans {

// Encode one stream with tANS. Returns (final_state, byte_stream, n_bits).
template <typename SymT>
static std::tuple<uint32_t, std::vector<uint8_t>, int32_t>
encode_stream(const SymT* symbols, int64_t n,
              const int64_t* freqs, int n_sym,
              const std::vector<std::vector<int>>& sym_positions,
              int table_log)
{
    int L = 1 << table_log;
    uint32_t state = (uint32_t)L;
    std::vector<uint8_t> bits_out;
    bits_out.reserve(static_cast<size_t>(n * 4));

    for (int64_t i = n - 1; i >= 0; i--) {
        int s = (int)symbols[i];
        int64_t f = freqs[s];

        // Normalize: output bits until state ∈ [f, 2f)
        int nb_out = 0;
        uint32_t temp = state;
        while (temp >= (uint32_t)(2 * f)) { nb_out++; temp >>= 1; }

        if (nb_out > 0) {
            uint32_t bits_value = state & ((1u << nb_out) - 1);
            // MSB-first: the CPU decoder reads backward assembling LSB-first.
            for (int b = nb_out - 1; b >= 0; b--) {
                bits_out.push_back((bits_value >> b) & 1);
            }
            state >>= nb_out;
        }

        // Map to table position
        int substate = (int)state - (int)f;
        int table_pos = sym_positions[s][substate];
        state = (uint32_t)(table_pos + L);
    }

    // Pack bits into bytes (LSB-first per byte)
    int n_bits = (int)bits_out.size();
    int n_bytes = (n_bits + 7) / 8;
    std::vector<uint8_t> stream(n_bytes, 0);
    for (int i = 0; i < n_bits; i++) {
        if (bits_out[i]) {
            stream[i / 8] |= (1u << (i % 8));
        }
    }

    return {state, std::move(stream), (int32_t)n_bits};
}

}  // namespace tans

// Returns (compressed, final_states, block_offsets, bit_counts).
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
tans_encode_interleaved(torch::Tensor symbols, torch::Tensor freqs,
                        int64_t block_streams, int64_t table_log)
{
    TORCH_CHECK(symbols.dim() == 2, "symbols must be [K, N]");
    TORCH_CHECK(symbols.dtype() == torch::kUInt8 || symbols.dtype() == torch::kInt16,
                "symbols must be uint8 or int16");
    TORCH_CHECK(freqs.dtype() == torch::kInt32, "freqs must be int32");
    TORCH_CHECK(symbols.is_contiguous() && symbols.is_cpu());
    TORCH_CHECK(freqs.is_contiguous() && freqs.is_cpu());

    int64_t K = symbols.size(0);
    int64_t N = symbols.size(1);
    int64_t n_syms = freqs.numel();
    int64_t n_blocks = (K + block_streams - 1) / block_streams;

    int L = 1 << table_log;
    const int32_t* f_ptr = freqs.data_ptr<int32_t>();

    // Verify freqs sum to L
    int64_t acc = 0;
    for (int64_t i = 0; i < n_syms; i++) {
        TORCH_CHECK(f_ptr[i] > 0, "all freqs must be > 0");
        acc += f_ptr[i];
    }
    TORCH_CHECK(acc == L, "freqs must sum to ", L, "; got ", acc);

    // Build freq array as int64 for the encoder
    std::vector<int64_t> freqs_i64(n_syms);
    for (int64_t i = 0; i < n_syms; i++) freqs_i64[i] = f_ptr[i];

    // Build symbol spread + positions (same as Python build_tans_decode_table)
    int step = (L >> 1) + (L >> 3) + 3;
    if (step % 2 == 0) step++;
    std::vector<int> symbol_table(L);
    int pos = 0;
    for (int s = 0; s < (int)n_syms; s++) {
        for (int64_t j = 0; j < freqs_i64[s]; j++) {
            symbol_table[pos] = s;
            pos = (pos + step) % L;
        }
    }
    std::vector<std::vector<int>> sym_positions(n_syms);
    for (int x = 0; x < L; x++) {
        sym_positions[symbol_table[x]].push_back(x);
    }

    // Phase 1: per-stream encode (OpenMP parallel)
    std::vector<std::vector<uint8_t>> streams(K);
    std::vector<int32_t> padded_lens(K);
    std::vector<int32_t> states_vec(K);
    std::vector<int32_t> nbits_vec(K);
    bool use_int16 = (symbols.dtype() == torch::kInt16);

    #pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < K; k++) {
        std::tuple<uint32_t, std::vector<uint8_t>, int32_t> result;
        if (use_int16)
            result = tans::encode_stream(
                symbols.data_ptr<int16_t>() + k * N, N,
                freqs_i64.data(), (int)n_syms, sym_positions, (int)table_log);
        else
            result = tans::encode_stream(
                symbols.data_ptr<uint8_t>() + k * N, N,
                freqs_i64.data(), (int)n_syms, sym_positions, (int)table_log);

        states_vec[k] = (int32_t)std::get<0>(result);
        streams[k]    = std::move(std::get<1>(result));
        nbits_vec[k]  = std::get<2>(result);
        int64_t raw_len = (int64_t)streams[k].size();
        int64_t pad = (4 - raw_len % 4) % 4;
        padded_lens[k] = (int32_t)(raw_len + pad);
    }

    // Phase 2: block layout (identical to rANS)
    std::vector<int32_t> block_offsets_vec(n_blocks + 1);
    std::vector<int32_t> block_Gs(n_blocks);
    int64_t cum_off = 0;
    for (int64_t b = 0; b < n_blocks; b++) {
        int64_t start = b * block_streams;
        int64_t end = std::min(start + block_streams, K);
        int32_t max_len = 0;
        for (int64_t k = start; k < end; k++)
            if (padded_lens[k] > max_len) max_len = padded_lens[k];
        if (max_len == 0) max_len = 4;
        int32_t G_b = max_len / 4;
        if (G_b & 1) G_b++;
        block_Gs[b] = G_b;
        block_offsets_vec[b] = (int32_t)cum_off;
        cum_off += (int64_t)G_b * block_streams * 4;
    }
    block_offsets_vec[n_blocks] = (int32_t)cum_off;

    // Phase 3: interleave into slabs
    auto compressed = torch::zeros({cum_off}, torch::kUInt8);
    uint8_t* out = compressed.data_ptr<uint8_t>();

    #pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < K; k++) {
        int64_t b = k / block_streams;
        int64_t tid = k % block_streams;
        int32_t G_b = block_Gs[b];
        int32_t L_pad = padded_lens[k];
        int32_t G_s = L_pad / 4;
        if (G_s == 0) continue;

        int32_t pad = L_pad - (int32_t)streams[k].size();
        int32_t block_base = block_offsets_vec[b];
        const uint8_t* src = streams[k].data();
        int64_t src_pos = 0;

        for (int32_t g = 0; g < G_s; g++) {
            int32_t slab_idx = G_b - G_s + g;
            int64_t dst = (int64_t)block_base
                + (int64_t)slab_idx * block_streams * 4
                + tid * 4;
            for (int j = 0; j < 4; j++) {
                int32_t byte_pos = g * 4 + j;
                if (byte_pos >= pad) out[dst + j] = src[src_pos++];
            }
        }
    }

    auto states_t = torch::from_blob(
        states_vec.data(), {K}, torch::TensorOptions().dtype(torch::kInt32)
    ).clone();
    auto offsets_t = torch::from_blob(
        block_offsets_vec.data(), {n_blocks + 1},
        torch::TensorOptions().dtype(torch::kInt32)
    ).clone();
    auto nbits_t = torch::from_blob(
        nbits_vec.data(), {K}, torch::TensorOptions().dtype(torch::kInt32)
    ).clone();

    return {compressed, states_t, offsets_t, nbits_t};
}

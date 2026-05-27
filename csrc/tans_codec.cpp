// CPU tANS encoder. Produces the slab-interleaved layout consumed by the
// GPU decoder and fused vecadd kernel.
//
// State pinned to [L, 2L) with L = M = 4096. Bit stream packed LSB-first
// within each uint32 slab. The final partial slab (if any) is emitted
// LSB-aligned (no shift); per-stream `partial_cnt` reports how many low
// bits are valid so the decoder can skip the MSB padding at the start.

#include <torch/extension.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>
#include <tuple>

namespace tans {

constexpr uint32_t L = 4096;
constexpr uint32_t M = L;
constexpr int64_t BLOCK_STREAMS = 128;

template <typename SymT>
static std::tuple<uint32_t, std::vector<uint8_t>, uint8_t>
encode_one(const SymT* symbols, int64_t n,
           const uint32_t* const* encode_table,
           const uint32_t* freqs)
{
    uint32_t x = L;
    std::vector<uint32_t> slabs;
    slabs.reserve(static_cast<size_t>(n / 4) + 4);
    uint32_t cur = 0;
    uint32_t cnt = 0;

    for (int64_t i = n - 1; i >= 0; i--) {
        int s = (int)symbols[i];
        uint32_t f = freqs[s];
        uint32_t two_f = 2u * f;
        while (x >= two_f) {
            cur |= (x & 1u) << cnt;
            cnt++;
            if (cnt == 32u) {
                slabs.push_back(cur);
                cur = 0u;
                cnt = 0u;
            }
            x >>= 1;
        }
        x = encode_table[s][x - f];
    }

    uint8_t partial = (uint8_t)cnt;
    if (cnt > 0u) {
        slabs.push_back(cur);
    }

    // Convert uint32 slabs -> little-endian bytes.
    std::vector<uint8_t> bytes(slabs.size() * 4);
    std::memcpy(bytes.data(), slabs.data(), bytes.size());
    return {x, std::move(bytes), partial};
}

}  // namespace tans

// Returns (compressed, final_states, block_offsets, partial_cnts).
// Inputs:
//   symbols [K, N] uint8
//   freqs   [n_alphabet] int32, sums to L=4096
//   spread  [L] int32, output of Yann's spread (slot -> symbol)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
tans_encode_interleaved(torch::Tensor symbols, torch::Tensor freqs,
                        torch::Tensor spread)
{
    TORCH_CHECK(symbols.dim() == 2, "symbols must be [K, N]");
    TORCH_CHECK(symbols.dtype() == torch::kUInt8);
    TORCH_CHECK(freqs.dtype() == torch::kInt32);
    TORCH_CHECK(spread.dtype() == torch::kInt32);
    TORCH_CHECK(symbols.is_contiguous() && symbols.is_cpu());
    TORCH_CHECK(freqs.is_contiguous() && freqs.is_cpu());
    TORCH_CHECK(spread.is_contiguous() && spread.is_cpu());
    TORCH_CHECK(spread.numel() == (int64_t)tans::L);

    int64_t K = symbols.size(0);
    int64_t N = symbols.size(1);
    int64_t n_syms = freqs.numel();
    int64_t n_blocks = (K + tans::BLOCK_STREAMS - 1) / tans::BLOCK_STREAMS;

    const int32_t* f_ptr = freqs.data_ptr<int32_t>();
    const uint32_t* f_u32 = reinterpret_cast<const uint32_t*>(f_ptr);
    const int32_t* sp_ptr = spread.data_ptr<int32_t>();

    // Validate freqs sum to L.
    uint32_t freq_sum = 0;
    for (int64_t i = 0; i < n_syms; i++) {
        TORCH_CHECK(f_ptr[i] >= 0);
        freq_sum += (uint32_t)f_ptr[i];
    }
    TORCH_CHECK(freq_sum == tans::L,
                "freqs must sum to L=", tans::L, "; got ", freq_sum);

    // Build encode tables from spread.
    // For each symbol s, an array of length f_s with the next-state for each rank.
    std::vector<std::vector<uint32_t>> encode_tables_storage(n_syms);
    std::vector<const uint32_t*> encode_tables(n_syms);
    for (int64_t s = 0; s < n_syms; s++) {
        encode_tables_storage[s].assign(f_ptr[s], 0u);
    }
    {
        std::vector<uint32_t> counts(n_syms, 0);
        for (uint32_t slot = 0; slot < tans::L; slot++) {
            int s = sp_ptr[slot];
            TORCH_CHECK(s >= 0 && s < (int)n_syms);
            uint32_t rank = counts[s]++;
            encode_tables_storage[s][rank] = tans::L + slot;
        }
    }
    for (int64_t s = 0; s < n_syms; s++) {
        encode_tables[s] = encode_tables_storage[s].data();
    }

    // Phase 1: per-stream encode in parallel.
    std::vector<std::vector<uint8_t>> streams(K);
    // State is in [L=4096, 2L=8192) = 13 bits, fits in uint16.
    std::vector<uint16_t> states_vec(K);
    std::vector<uint8_t> partial_vec(K);
    std::vector<int32_t> padded_lens(K);

    #pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < K; k++) {
        auto result = tans::encode_one(
            symbols.data_ptr<uint8_t>() + k * N, N,
            encode_tables.data(), f_u32);
        states_vec[k]   = (uint16_t)std::get<0>(result);
        streams[k]      = std::move(std::get<1>(result));
        partial_vec[k]  = std::get<2>(result);
        padded_lens[k]  = (int32_t)streams[k].size();   // already multiple of 4
    }

    // Phase 2: compute G_b per block and cumulative offsets.
    std::vector<int32_t> block_offsets_vec(n_blocks + 1);
    std::vector<int32_t> block_Gs(n_blocks);
    int64_t cum_offset = 0;
    for (int64_t b = 0; b < n_blocks; b++) {
        int64_t start = b * tans::BLOCK_STREAMS;
        int64_t end   = std::min(start + tans::BLOCK_STREAMS, K);
        int32_t max_len = 0;
        for (int64_t k = start; k < end; k++) {
            if (padded_lens[k] > max_len) max_len = padded_lens[k];
        }
        if (max_len == 0) max_len = 4;
        int32_t G_b = max_len / 4;
        block_Gs[b] = G_b;
        block_offsets_vec[b] = (int32_t)cum_offset;
        cum_offset += (int64_t)G_b * tans::BLOCK_STREAMS * 4;
    }
    block_offsets_vec[n_blocks] = (int32_t)cum_offset;

    // Phase 3: pack bytes into slabs. Stream byte_pos g*4+j → slab g (local),
    // absolute slab idx = G_b - G_s + g, byte position tid*4 + j.
    auto compressed = torch::zeros({cum_offset}, torch::kUInt8);
    uint8_t* out = compressed.data_ptr<uint8_t>();

    #pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < K; k++) {
        int64_t b   = k / tans::BLOCK_STREAMS;
        int64_t tid = k % tans::BLOCK_STREAMS;
        int32_t G_b = block_Gs[b];
        int32_t L_pad = padded_lens[k];
        int32_t G_s = L_pad / 4;
        if (G_s == 0) continue;

        int32_t block_base = block_offsets_vec[b];
        const uint8_t* src = streams[k].data();

        for (int32_t g = 0; g < G_s; g++) {
            int32_t slab_idx = G_b - G_s + g;
            int64_t dst = (int64_t)block_base
                        + (int64_t)slab_idx * tans::BLOCK_STREAMS * 4
                        + tid * 4;
            std::memcpy(out + dst, src + g * 4, 4);
        }
    }

    auto states_t = torch::from_blob(
        states_vec.data(), {K}, torch::TensorOptions().dtype(torch::kUInt16)
    ).clone();
    auto offsets_t = torch::from_blob(
        block_offsets_vec.data(), {n_blocks + 1},
        torch::TensorOptions().dtype(torch::kInt32)
    ).clone();
    auto partial_t = torch::from_blob(
        partial_vec.data(), {K},
        torch::TensorOptions().dtype(torch::kUInt8)
    ).clone();

    return {compressed, states_t, offsets_t, partial_t};
}

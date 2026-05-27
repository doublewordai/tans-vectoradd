"""tANS encoder for FP8 pair-alphabet streams.

The compressor splits each FP8 byte into:
- a 4-bit exponent, entropy-coded in pairs with tANS;
- a 4-bit sign+mantissa nibble, stored uncompressed.

Encoding runs on CPU once per benchmark and is not performance-critical.

Wire format (per stream):
- Encoder produces a sequence of uint32 slabs. Bits accumulate LSB-first
  within each slab. The final partial slab (if any) is left LSB-aligned;
  the encoder reports `partial_cnt` so the decoder can skip the MSB-end
  zero padding on its first read.
- Per-block layout: G_b slabs per stream (max in block, padded). Each
  stream's slabs occupy the TOP G_s indices of its block; lower indices
  are zero-padded.
"""
from __future__ import annotations

import numpy as np
import torch


L = 4096
M = L
SIGMA = (L >> 1) + (L >> 3) + 3   # Yann's hash-walk stride
BLOCK_STREAMS = 128


def quantize_freqs(probs: np.ndarray, target: int = M) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    f = np.maximum(np.round(probs * target).astype(np.int64), 1)
    f[probs == 0] = 0
    diff = target - f.sum()
    if diff > 0:
        for _ in range(int(diff)):
            err = probs * target - f
            err[probs == 0] = -np.inf
            f[int(np.argmax(err))] += 1
    elif diff < 0:
        for _ in range(int(-diff)):
            err = f - probs * target
            err[(probs > 0) & (f <= 1)] = -np.inf
            f[int(np.argmax(err))] -= 1
    assert f.sum() == target
    return f.astype(np.int32)


def build_spread(freqs: np.ndarray) -> np.ndarray:
    """Yann's hash-walk: place each symbol f_s times at strides of SIGMA."""
    assert freqs.sum() == L
    spread = np.full(L, -1, dtype=np.int32)
    cursor = 0
    for s in range(len(freqs)):
        for _ in range(int(freqs[s])):
            assert spread[cursor] == -1
            spread[cursor] = s
            cursor = (cursor + SIGMA) % L
    assert (spread >= 0).all()
    return spread


def build_decode_table(spread: np.ndarray, freqs: np.ndarray) -> torch.Tensor:
    """Return [L] int32 decode table.

    Entry layout: sym[0..7] | nbBits[8..11] | base_state[16..31].
    Decoder reads `nb` bits and transitions to `x = base_state | bits`.
    """
    spread = np.asarray(spread, dtype=np.int32)
    freqs  = np.asarray(freqs, dtype=np.int32)
    table = np.zeros(L, dtype=np.uint32)
    counts = np.zeros(len(freqs), dtype=np.int64)
    for slot in range(L):
        s = int(spread[slot])
        rank = int(counts[s])
        f = int(freqs[s])
        x_prev = f + rank
        nb = 0
        v = x_prev
        while v < L:
            v <<= 1
            nb += 1
        base = x_prev << nb
        assert L <= base < 2 * L
        table[slot] = (s & 0xFF) | ((nb & 0xF) << 8) | ((base & 0xFFFF) << 16)
        counts[s] += 1
    return torch.from_numpy(table.view(np.int32))


def encode(
    fp8_bytes: torch.Tensor,
    exp_freqs: torch.Tensor,
    tile: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor]:
    """tANS encode FP8 bytes (pair alphabet).

    If tile > 0, sm_packed is returned in tiled layout [n_tiles, K, tile]
    for fused vecadd. Otherwise it is [N/2, K] for standalone decode.

    Returns:
      compressed [bytes], final_states [K], block_offsets [n_blocks+1],
      partial_cnts [K] (uint8), sm_packed, pair_freqs, spread.
    """
    assert fp8_bytes.dim() == 2 and fp8_bytes.dtype == torch.uint8
    K, N = fp8_bytes.shape
    assert N % 2 == 0

    exp_nibbles = (fp8_bytes >> 3) & 0xF
    sm_nibbles  = ((fp8_bytes >> 4) & 0x8) | (fp8_bytes & 0x7)

    sm_pairs     = sm_nibbles.view(K, N // 2, 2)
    sm_packed_kn = sm_pairs[..., 0] | (sm_pairs[..., 1] << 4)

    if tile > 0:
        n_pairs = N // 2
        assert n_pairs % tile == 0, f"n_pairs={n_pairs} not divisible by tile={tile}"
        n_tiles = n_pairs // tile
        sm_packed = sm_packed_kn.reshape(K, n_tiles, tile).permute(1, 0, 2).contiguous()
    else:
        sm_packed = sm_packed_kn.t().contiguous()

    exp_pairs = exp_nibbles.view(K, N // 2, 2)
    pair_syms = (exp_pairs[..., 0] * 16 + exp_pairs[..., 1]).to(torch.uint8).numpy()

    probs = exp_freqs.float().numpy()
    probs = probs / probs.sum()
    pair_probs = (probs[:, None] * probs[None, :]).flatten()
    pair_freqs = quantize_freqs(pair_probs, target=M)
    spread = build_spread(pair_freqs)

    pair_syms_t = torch.from_numpy(pair_syms).contiguous()
    spread_t = torch.from_numpy(spread)
    pair_freqs_t = torch.from_numpy(pair_freqs)

    from rans_vectoradd._C import tans_encode_interleaved
    compressed, states, offsets, partial_cnts = tans_encode_interleaved(
        pair_syms_t, pair_freqs_t, spread_t)

    return (
        compressed, states, offsets, partial_cnts,
        sm_packed, pair_freqs_t, spread_t,
    )

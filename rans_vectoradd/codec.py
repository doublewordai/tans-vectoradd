"""FP8 rANS encoding — pair alphabet (256 symbols, M=4096).

Encoding plan for FP8 E4M3:
  Each byte:  S EEEE MMM   (sign 1 bit | exponent 4 bits | mantissa 3 bits)

  - Pairs of exponent nibbles are entropy-coded with rANS against the
    measured Qwen-style exponent distribution (outer product of marginals).
  - Sign+mantissa nibbles (4 bits each) are stored uncompressed, packed
    two per byte.

Each pair of FP8 bytes produces one 256-symbol rANS symbol.  The GPU
decoder processes one symbol per step, yielding 2 decoded FP8 bytes.
"""
from __future__ import annotations

import numpy as np
import torch

from rans_vectoradd._C import rans_encode_interleaved as _cpu_encode_interleaved


M = 4096                # pair-alphabet frequency precision
BLOCK_STREAMS = 128     # must match rans_decode.cu


def quantize_freqs(probs: np.ndarray, M: int = M) -> torch.Tensor:
    """Quantize probabilities to integer frequencies summing to M.

    Every symbol with p > 0 gets freq >= 1. Rounding residuals are
    redistributed so the total equals M exactly. Returns int32.
    """
    probs = np.asarray(probs, dtype=np.float64)
    assert abs(probs.sum() - 1.0) < 1e-6, f"probs must sum to 1, got {probs.sum()}"

    f = np.maximum(np.round(probs * M).astype(np.int64), 1)
    f[probs == 0] = 0
    diff = M - f.sum()

    if diff > 0:
        for _ in range(int(diff)):
            err = probs * M - f
            err[probs == 0] = -np.inf
            f[int(np.argmax(err))] += 1
    elif diff < 0:
        for _ in range(int(-diff)):
            err = f - probs * M
            err[(probs > 0) & (f <= 1)] = -np.inf
            f[int(np.argmax(err))] -= 1

    assert f.sum() == M
    assert np.all(f[probs > 0] >= 1)
    return torch.from_numpy(f.astype(np.int32))


def encode(
    fp8_bytes: torch.Tensor,   # [K, N] uint8, FP8 E4M3
    exp_freqs: torch.Tensor,   # [16] int32 (single-nibble freqs)
    block_streams: int = BLOCK_STREAMS,
    tile: int = 0,             # 0 = pair-major (original), >0 = tiled layout
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode FP8 bytes using the pair alphabet.

    Two consecutive exponent nibbles are encoded as a single 256-symbol
    rANS symbol with M=4096. Halves decode steps vs single-nibble.

    If tile > 0, sm_packed is returned in tiled layout [n_tiles, K, tile]
    for vectorized GPU loads with better DRAM row buffer hits.

    Returns (compressed, final_states, block_offsets, sign_mantissa_packed,
    pair_freqs). Pass pair_freqs to gpu_rans_decode / gpu_rans_decode_dump.
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
        # [K, n_tiles, tile] -> [n_tiles, K, tile] for coalesced tiled access
        sm_packed = sm_packed_kn.reshape(K, n_tiles, tile).permute(1, 0, 2).contiguous()
    else:
        sm_packed = sm_packed_kn.t().contiguous()

    # Pair symbols: consecutive exponent nibbles → single 0-255 symbol
    exp_pairs = exp_nibbles.view(K, N // 2, 2)
    pair_syms = (exp_pairs[..., 0] * 16 + exp_pairs[..., 1]).to(torch.uint8)

    # Pair frequencies: outer product of single-nibble distribution
    probs = exp_freqs.float()
    probs = probs / probs.sum()
    pair_probs = (probs.unsqueeze(1) * probs.unsqueeze(0)).flatten().numpy()
    pair_freqs = quantize_freqs(pair_probs, M=M)

    compressed, states, block_offsets = _cpu_encode_interleaved(
        pair_syms.contiguous(), pair_freqs, block_streams
    )
    return compressed, states, block_offsets, sm_packed, pair_freqs

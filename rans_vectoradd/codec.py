"""Python helpers for FP8 rANS encoding with the exponent-pair alphabet.

The decoder on the GPU works on a 256-symbol alphabet where each symbol
bundles two consecutive 4-bit FP8 exponent nibbles. Sign+mantissa
nibbles are stored uncompressed (packed two nibbles per byte).
"""
from __future__ import annotations

import numpy as np
import torch

from rans_vectoradd._C import rans_encode_interleaved as _cpu_encode_interleaved


M = 2048                # must match rans_codec.cpp / rans_decode.cu
BLOCK_STREAMS = 128     # must match rans_decode.cu's BLOCK_STREAMS


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


def pair_freqs_from_single(exp_freqs: torch.Tensor, M: int = M) -> torch.Tensor:
    """Build a 256-entry frequency table for exponent PAIRS from the
    16-entry single-exponent table, assuming independence.

    Pair symbol layout: pair = exp_low | (exp_high << 4) — so flat index
    p decomposes into high = p >> 4, low = p & 0xF. Under the
    independence assumption, p(pair=p) = p(exp_low) * p(exp_high).

    Returned frequencies sum to M (same as the input table).
    """
    assert exp_freqs.numel() == 16, "expected a 16-entry exponent table"
    assert int(exp_freqs.sum()) == M, f"exp_freqs must sum to {M}"
    p = exp_freqs.numpy().astype(np.float64) / M
    p_pair = np.outer(p, p).flatten()  # row-major: index high*16 + low
    return quantize_freqs(p_pair, M)


def batch_encode_fp8(
    fp8_bytes: torch.Tensor,   # [K, N] uint8, FP8 E4M3
    exp_freqs: torch.Tensor,   # [16] int32 — single-exponent freq table
    block_streams: int = BLOCK_STREAMS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode FP8 bytes as (compressed exponent-pair stream, packed
    sign+mantissa nibbles).

    FP8 E4M3 bit layout (MSB → LSB): S | EEEE | MMM.

    The 4-bit exponent from each pair of adjacent FP8 bytes is packed
    into an 8-bit pair symbol (`low | high<<4`) and the 256-symbol
    stream is rANS-coded. The 4-bit sign+mantissa nibble
    `((sign << 3) | mantissa)` is stored uncompressed, packed two
    nibbles per byte into an [N/2, K] layout that matches the decoder's
    [N, K] output stride so warp reads at iteration i coalesce.

    Returns (compressed, final_states, block_offsets, sign_mantissa_packed,
             pair_freqs) where pair_freqs is the 256-entry table the
    decoder needs.
    """
    assert fp8_bytes.dim() == 2 and fp8_bytes.dtype == torch.uint8
    assert fp8_bytes.is_cpu
    K, N = fp8_bytes.shape
    assert N % 2 == 0, "N must be even to pair exponents"

    exp_nibbles = (fp8_bytes >> 3) & 0xF
    sm_nibbles  = ((fp8_bytes >> 4) & 0x8) | (fp8_bytes & 0x7)

    # Pair exponents: pair_symbols[k, i] = exp[k, 2i] | (exp[k, 2i+1] << 4).
    exp_pairs = exp_nibbles.view(K, N // 2, 2)
    pair_symbols = exp_pairs[..., 0] | (exp_pairs[..., 1] << 4)          # [K, N/2]

    # Pack sign+mantissa with the same stride as the pair stream.
    sm_pairs = sm_nibbles.view(K, N // 2, 2)
    sm_packed_kn = sm_pairs[..., 0] | (sm_pairs[..., 1] << 4)            # [K, N/2]
    sm_packed = sm_packed_kn.t().contiguous()                            # [N/2, K]

    # 256-entry pair-frequency table (independence assumption).
    pair_freqs = pair_freqs_from_single(exp_freqs)

    compressed, states, block_offsets = _cpu_encode_interleaved(
        pair_symbols.contiguous(), pair_freqs, block_streams
    )
    return compressed, states, block_offsets, sm_packed, pair_freqs

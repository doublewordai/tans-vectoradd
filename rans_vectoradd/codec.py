"""Python helpers for FP8 rANS encoding (16-symbol exponent alphabet).

Encoding plan for FP8 E4M3:
  Each byte:  S EEEE MMM   (sign 1 bit | exponent 4 bits | mantissa 3 bits)

  - Exponent nibbles are entropy-coded with rANS against the measured
    Qwen-style exponent distribution (non-uniform, peaked around 12-13).
  - Sign+mantissa nibbles (sign | mantissa, 4 bits) are stored
    uncompressed, packed two nibbles per byte.

The encoder emits one independent rANS stream per row of the input. On
the GPU, each stream is decoded by one thread; streams are interleaved
slab-wise so warp refills are coalesced.
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


def batch_encode_fp8(
    fp8_bytes: torch.Tensor,   # [K, N] uint8, FP8 E4M3
    exp_freqs: torch.Tensor,   # [16] int32 summing to M
    block_streams: int = BLOCK_STREAMS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode FP8 bytes into (rANS-coded exponent stream,
    uncompressed sign+mantissa nibbles).

    Returns (compressed, final_states, block_offsets, sign_mantissa_packed).
    Shapes:
        compressed:          [total_bytes]           uint8
        final_states:        [K]                     int32
        block_offsets:       [n_blocks + 1]          int32
        sign_mantissa_packed: [N/2, K]               uint8  (2 nibbles/byte)
    """
    assert fp8_bytes.dim() == 2 and fp8_bytes.dtype == torch.uint8
    assert fp8_bytes.is_cpu
    K, N = fp8_bytes.shape
    assert N % 2 == 0, "N must be even to pack sign+mantissa nibbles"

    exp_nibbles = (fp8_bytes >> 3) & 0xF                           # [K, N]
    sm_nibbles  = ((fp8_bytes >> 4) & 0x8) | (fp8_bytes & 0x7)     # [K, N]

    # Pack sign+mantissa: two nibbles per byte, transposed so warp reads
    # at iteration i coalesce (threads read the i-th nibble-pair byte).
    sm_pairs     = sm_nibbles.view(K, N // 2, 2)                   # [K, N/2, 2]
    sm_packed_kn = sm_pairs[..., 0] | (sm_pairs[..., 1] << 4)      # [K, N/2]
    sm_packed    = sm_packed_kn.t().contiguous()                   # [N/2, K]

    compressed, states, block_offsets = _cpu_encode_interleaved(
        exp_nibbles.contiguous(), exp_freqs, block_streams
    )
    return compressed, states, block_offsets, sm_packed

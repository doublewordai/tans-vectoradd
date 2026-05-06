"""Minimal ncu driver for the fused tANS vecadd kernel.

Usage:
    ncu --target-processes all --kernel-name regex:fp8_vecadd_fused_tans \
        --launch-skip 5 --launch-count 3 --set detailed \
        .venv/bin/python profile_vecadd.py 1024
"""
from __future__ import annotations

import sys

import torch

from rans_vectoradd import (
    QWEN3_14B_FP8_EXP,
    fp8_vecadd_fused_tans,
    random_fp8_bytes,
    tans_codec,
)

TILE_PAIRS = 8


def main() -> None:
    n_fp8_per_stream = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    n_bytes = 1 << 30
    n_streams = n_bytes // n_fp8_per_stream

    exp_freqs = torch.tensor(QWEN3_14B_FP8_EXP, dtype=torch.float64)
    torch.manual_seed(42)
    fp8_a = random_fp8_bytes(n_streams * n_fp8_per_stream, device="cpu")
    fp8_b = random_fp8_bytes(n_streams * n_fp8_per_stream, device="cpu")
    fp8_a = fp8_a.reshape(n_streams, n_fp8_per_stream)
    fp8_b = fp8_b.reshape(n_streams, n_fp8_per_stream)

    ca, sa, oa, pa, sma, pf, sp = tans_codec.encode(
        fp8_a, exp_freqs, tile=TILE_PAIRS)
    cb, sb, ob, pb, smb, _, _ = tans_codec.encode(
        fp8_b, exp_freqs, tile=TILE_PAIRS)
    decode_tbl = tans_codec.build_decode_table(sp.numpy(), pf.numpy())
    args = [t.cuda() for t in (ca, oa, sa, pa, sma, cb, ob, sb, pb, smb, decode_tbl)]

    for _ in range(5):
        fp8_vecadd_fused_tans(*args, n_fp8_per_stream)
    torch.cuda.synchronize()

    for _ in range(3):
        fp8_vecadd_fused_tans(*args, n_fp8_per_stream)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()

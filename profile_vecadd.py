"""Minimal profiler driver for the FP8 vecadd kernels.

Usage:
    ncu --target-processes all --kernel-name regex:fp8_vecadd_fused_tans \
        --launch-skip 5 --launch-count 3 --set detailed \
        .venv/bin/python profile_vecadd.py 1024 --variant register
"""
from __future__ import annotations

import argparse
import os

import torch

from rans_vectoradd import (
    QWEN3_14B_FP8_EXP,
    fp8_vecadd_fused_tans_register,
    fp8_vecadd_fused_tans_shared,
    fp8_vecadd_raw,
    random_fp8_bytes,
    tans_codec,
)

TILE_PAIRS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int, nargs="?", default=1024)
    parser.add_argument("--mode", choices=["fused", "raw"], default="fused")
    parser.add_argument("--size-mib", type=int, default=1024)
    parser.add_argument("--streams-per-block", type=int, choices=[1, 2, 4, 8], default=8)
    parser.add_argument("--variant", choices=["shared", "register"], default="register")
    return parser.parse_args()


def profile_raw(n_bytes: int) -> None:
    a = torch.randint(0, 256, (n_bytes,), dtype=torch.uint8, device="cuda")
    b = torch.randint(0, 256, (n_bytes,), dtype=torch.uint8, device="cuda")

    for _ in range(5):
        fp8_vecadd_raw(a, b)
    torch.cuda.synchronize()

    for _ in range(3):
        fp8_vecadd_raw(a, b)
    torch.cuda.synchronize()


def profile_fused(
    n_bytes: int,
    n_fp8_per_stream: int,
    streams_per_block: int,
    variant: str,
) -> None:
    n_streams = n_bytes // n_fp8_per_stream
    os.environ["TANS_STREAMS_PER_BLOCK"] = str(streams_per_block)

    exp_freqs = torch.tensor(QWEN3_14B_FP8_EXP, dtype=torch.float64)
    torch.manual_seed(42)
    fp8_a = random_fp8_bytes(n_streams * n_fp8_per_stream, device="cpu")
    fp8_a = fp8_a.reshape(n_streams, n_fp8_per_stream)
    ca, sa, oa, pa, sma, pf, sp = tans_codec.encode(
        fp8_a, exp_freqs, tile=TILE_PAIRS)
    del fp8_a

    fp8_b = random_fp8_bytes(n_streams * n_fp8_per_stream, device="cpu")
    fp8_b = fp8_b.reshape(n_streams, n_fp8_per_stream)
    cb, sb, ob, pb, smb, _, _ = tans_codec.encode(
        fp8_b, exp_freqs, tile=TILE_PAIRS)
    del fp8_b
    decode_tbl = tans_codec.build_decode_table(sp.numpy(), pf.numpy())
    args = [t.cuda() for t in (ca, oa, sa, pa, sma, cb, ob, sb, pb, smb, decode_tbl)]
    fn = (
        fp8_vecadd_fused_tans_register
        if variant == "register"
        else fp8_vecadd_fused_tans_shared
    )

    for _ in range(5):
        fn(*args, n_fp8_per_stream)
    torch.cuda.synchronize()

    for _ in range(3):
        fn(*args, n_fp8_per_stream)
    torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    n_bytes = args.size_mib << 20
    if args.mode == "raw":
        profile_raw(n_bytes)
    else:
        profile_fused(n_bytes, args.n, args.streams_per_block, args.variant)


if __name__ == "__main__":
    main()

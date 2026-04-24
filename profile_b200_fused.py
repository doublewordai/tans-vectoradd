"""Profile driver for one raw or fused vecadd case.

This keeps ncu/nsys captures focused on a single steady-state kernel launch
instead of the full benchmark sweep.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from rans_vectoradd import QWEN3_14B_FP8_EXP, encode, quantize_freqs, random_fp8_bytes
from rans_vectoradd._C import fp8_vecadd_fused, fp8_vecadd_raw


def build_sfc(pair_freqs: torch.Tensor) -> torch.Tensor:
    pf_np = pair_freqs.numpy().astype(np.int32)
    cumul = np.zeros(len(pf_np) + 1, dtype=np.int32)
    cumul[1:] = np.cumsum(pf_np)
    sfc = np.zeros(int(cumul[-1]), dtype=np.uint32)
    for sym, f in enumerate(pf_np):
        c = int(cumul[sym])
        sfc[c : c + int(f)] = np.uint32(sym | (int(f) << 8) | (c << 20))
    return torch.from_numpy(sfc.view(np.int32)).cuda()


def prepare_fused(n_bytes: int, n_fp8_per_stream: int):
    k = n_bytes // n_fp8_per_stream
    exp_freqs = quantize_freqs(np.array(QWEN3_14B_FP8_EXP, dtype=np.float64))

    torch.manual_seed(42)
    fp8_a = random_fp8_bytes(k * n_fp8_per_stream, device="cpu").reshape(
        k, n_fp8_per_stream
    )
    fp8_b = random_fp8_bytes(k * n_fp8_per_stream, device="cpu").reshape(
        k, n_fp8_per_stream
    )

    ca, sa, oa, sma, pf = encode(fp8_a, exp_freqs, tile=8)
    cb, sb, ob, smb, _ = encode(fp8_b, exp_freqs, tile=8)
    sfc = build_sfc(pf)

    return (
        ca.cuda(),
        oa.cuda(),
        sa.cuda(),
        sma.cuda(),
        cb.cuda(),
        ob.cuda(),
        sb.cuda(),
        smb.cuda(),
        sfc,
        n_fp8_per_stream,
        835,
    )


def prepare_raw(n_bytes: int):
    torch.manual_seed(42)
    return (
        torch.randint(0, 256, (n_bytes,), dtype=torch.uint8, device="cuda"),
        torch.randint(0, 256, (n_bytes,), dtype=torch.uint8, device="cuda"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["raw", "fused"], default="fused")
    parser.add_argument("--n-bytes", type=int, default=1 << 30)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    if args.mode == "raw":
        raw_args = prepare_raw(args.n_bytes)
        run = lambda: fp8_vecadd_raw(*raw_args)
    else:
        fused_args = prepare_fused(args.n_bytes, args.n)
        run = lambda: fp8_vecadd_fused(*fused_args)

    torch.cuda.synchronize()
    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    for _ in range(args.repeat):
        run()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()

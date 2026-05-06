"""Benchmark fused tANS decompress+vecadd against raw FP8 vecadd."""
from __future__ import annotations

import argparse
import os

import torch
import triton.testing

from rans_vectoradd import (
    QWEN3_14B_FP8_EXP,
    fp8_vecadd_fused_tans_register,
    fp8_vecadd_fused_tans_shared,
    fp8_vecadd_raw,
    random_fp8_bytes,
    tans_codec,
)

TILE_PAIRS = 8


def bench(fn) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    return triton.testing.do_bench(fn, warmup=50, rep=200)


def bench_raw(n_bytes: int) -> dict[str, float]:
    a = torch.randint(0, 256, (n_bytes,), dtype=torch.uint8, device="cuda")
    b = torch.randint(0, 256, (n_bytes,), dtype=torch.uint8, device="cuda")
    ms = bench(lambda: fp8_vecadd_raw(a, b))
    seconds = ms / 1000.0
    return {
        "ms": ms,
        "gb_s": (3 * n_bytes) / 1e9 / seconds,
    }


def prepare_fused_inputs(
    n_bytes: int,
    n_fp8_per_stream: int,
) -> tuple[list[torch.Tensor], int, float]:
    n_streams = n_bytes // n_fp8_per_stream
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

    kernel_args = [
        t.cuda() for t in (ca, oa, sa, pa, sma, cb, ob, sb, pb, smb, decode_tbl)
    ]

    input_bytes = ca.numel() + cb.numel() + sma.numel() + smb.numel()
    metadata_bytes = sa.numel() * 2 + sb.numel() * 2 + oa.numel() * 4
    metadata_bytes += ob.numel() * 4 + pa.numel() + pb.numel()
    hbm_bytes = input_bytes + metadata_bytes + n_bytes
    bits_per_fp8 = (input_bytes + metadata_bytes) * 8 / (2 * n_bytes)

    return kernel_args, hbm_bytes, bits_per_fp8


def bench_fused(
    fn,
    kernel_args: list[torch.Tensor],
    n_fp8_per_stream: int,
    hbm_bytes: int,
    bits_per_fp8: float,
    streams_per_block: int,
) -> dict[str, float]:
    os.environ["TANS_STREAMS_PER_BLOCK"] = str(streams_per_block)
    ms = bench(lambda: fn(*kernel_args, n_fp8_per_stream))
    seconds = ms / 1000.0

    return {
        "ms": ms,
        "gb_s": hbm_bytes / 1e9 / seconds,
        "bits_per_fp8": bits_per_fp8,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mib", type=int, default=1024)
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048],
        help="FP8 bytes per stream.",
    )
    parser.add_argument(
        "--streams-per-block",
        type=int,
        nargs="+",
        choices=[1, 2, 4, 8],
        default=[8],
        help="Fused-kernel encoder-blocks per CTA variants.",
    )
    parser.add_argument(
        "--variant",
        choices=["shared", "register", "both"],
        default="both",
        help="Fused kernel staging variant to benchmark.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_bytes = args.size_mib << 20
    raw = bench_raw(n_bytes)

    print(f"raw: {raw['ms']:.3f} ms  {raw['gb_s']:.1f} GB/s", flush=True)
    for n_fp8_per_stream in args.n:
        if n_bytes % n_fp8_per_stream != 0:
            continue
        kernel_args, hbm_bytes, bits_per_fp8 = prepare_fused_inputs(
            n_bytes, n_fp8_per_stream)
        variants = []
        if args.variant in ("shared", "both"):
            variants.append(("shared", fp8_vecadd_fused_tans_shared))
        if args.variant in ("register", "both"):
            variants.append(("register", fp8_vecadd_fused_tans_register))

        for variant_name, fn in variants:
            for streams_per_block in args.streams_per_block:
                fused = bench_fused(
                    fn,
                    kernel_args,
                    n_fp8_per_stream,
                    hbm_bytes,
                    bits_per_fp8,
                    streams_per_block,
                )
                print(
                    f"{variant_name:<8}  "
                    f"spb={streams_per_block:<2}  "
                    f"N={n_fp8_per_stream:>5}  "
                    f"{fused['ms']:.3f} ms  "
                    f"{raw['ms'] / fused['ms']:.3f}x raw  "
                    f"{fused['gb_s']:.1f} effective GB/s  "
                    f"{fused['bits_per_fp8']:.3f} compressed bits/fp8",
                    flush=True,
                )


if __name__ == "__main__":
    main()

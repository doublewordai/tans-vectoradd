"""Benchmark fused tANS decompress+vecadd against raw FP8 vecadd."""
from __future__ import annotations

import argparse
import os
from collections.abc import Iterator

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
DEFAULT_CACHE_CONFIGS = ["prefer_equal"]
DEFAULT_SHARED_CARVEOUTS = ["default"]
DEFAULT_PREFETCH = ["off"]


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
    cache_config: str,
    shared_carveout: str,
    prefetch: str,
) -> dict[str, float]:
    os.environ["TANS_STREAMS_PER_BLOCK"] = str(streams_per_block)
    os.environ["TANS_CACHE_CONFIG"] = cache_config
    os.environ["TANS_REGISTER_PREFETCH"] = prefetch
    if shared_carveout == "unset":
        os.environ.pop("TANS_SHARED_CARVEOUT", None)
    else:
        os.environ["TANS_SHARED_CARVEOUT"] = shared_carveout
    ms = bench(lambda: fn(*kernel_args, n_fp8_per_stream))
    seconds = ms / 1000.0

    return {
        "ms": ms,
        "gb_s": hbm_bytes / 1e9 / seconds,
        "bits_per_fp8": bits_per_fp8,
    }


def iter_modes(args: argparse.Namespace) -> Iterator[tuple[int, str, str, str]]:
    for streams_per_block in args.streams_per_block:
        for cache_config in args.cache_config:
            for shared_carveout in args.shared_carveout:
                for prefetch in args.prefetch:
                    yield streams_per_block, cache_config, shared_carveout, prefetch


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
    parser.add_argument(
        "--cache-config",
        nargs="+",
        default=DEFAULT_CACHE_CONFIGS,
        help=(
            "Register-kernel cudaFuncSetCacheConfig mode: prefer_l1, "
            "prefer_shared, prefer_equal, or prefer_none."
        ),
    )
    parser.add_argument(
        "--shared-carveout",
        nargs="+",
        default=DEFAULT_SHARED_CARVEOUTS,
        help=(
            "Register-kernel TANS_SHARED_CARVEOUT values: unset, default, "
            "max_shared, max_l1, equal, or integer percentage."
        ),
    )
    parser.add_argument(
        "--prefetch",
        nargs="+",
        default=DEFAULT_PREFETCH,
        help="Register-kernel prefetch mode: on or off.",
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
            for streams_per_block, cache_config, shared_carveout, prefetch in iter_modes(args):
                try:
                    fused = bench_fused(
                        fn,
                        kernel_args,
                        n_fp8_per_stream,
                        hbm_bytes,
                        bits_per_fp8,
                        streams_per_block,
                        cache_config,
                        shared_carveout,
                        prefetch,
                    )
                except Exception as exc:
                    print(
                        f"{variant_name:<8}  "
                        f"spb={streams_per_block:<2}  "
                        f"cache={cache_config:<13}  "
                        f"carveout={shared_carveout:<10}  "
                        f"prefetch={prefetch:<3}  "
                        f"N={n_fp8_per_stream:>5}  "
                        f"FAIL  {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue
                print(
                    f"{variant_name:<8}  "
                    f"spb={streams_per_block:<2}  "
                    f"cache={cache_config:<13}  "
                    f"carveout={shared_carveout:<10}  "
                    f"prefetch={prefetch:<3}  "
                    f"N={n_fp8_per_stream:>5}  "
                    f"{fused['ms']:.3f} ms  "
                    f"{raw['ms'] / fused['ms']:.3f}x raw  "
                    f"{fused['gb_s']:.1f} effective GB/s  "
                    f"{fused['bits_per_fp8']:.3f} compressed bits/fp8",
                    flush=True,
                )


if __name__ == "__main__":
    main()

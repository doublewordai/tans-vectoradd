"""Benchmark: fused decompress + FP8 GEMV vs raw FP8 GEMV.

y = W @ x, W is [M, K] FP8, x is [K] FP16, y is [M] FP16.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import triton.testing

from rans_vectoradd._C import fp8_gemv_raw, fp8_gemv_raw_batch, fp8_gemv_fused
from rans_vectoradd import (
    QWEN3_14B_FP8_EXP,
    encode,
    quantize_freqs,
    random_fp8_bytes,
)


def build_sfc(pair_freqs):
    pf_np = pair_freqs.numpy().astype(np.int32)
    cumul = np.zeros(len(pf_np) + 1, dtype=np.int32)
    cumul[1:] = np.cumsum(pf_np)
    M = int(cumul[-1])
    sfc = np.zeros(M, dtype=np.uint32)
    for sym in range(len(pf_np)):
        f = int(pf_np[sym]); c = int(cumul[sym])
        for slot in range(c, c + f):
            sfc[slot] = np.uint32(sym | (f << 8) | (c << 20))
    return torch.from_numpy(sfc.view(np.int32)).cuda()


def bench_raw(M: int, K: int) -> float:
    W = torch.randint(0, 256, (M, K), dtype=torch.uint8, device="cuda")
    x = torch.randn(K, dtype=torch.half, device="cuda")
    def _run():
        fp8_gemv_raw(W, x)
    for _ in range(5):
        _run()
    torch.cuda.synchronize()
    return triton.testing.do_bench(_run, warmup=50, rep=200)


def bench_raw_batch(M: int, K: int, B: int, *, check: bool = False) -> float:
    W = torch.randint(0, 256, (M, K), dtype=torch.uint8, device="cuda")
    x = torch.randn(B, K, dtype=torch.half, device="cuda")

    def _run():
        return fp8_gemv_raw_batch(W, x)

    if check:
        ref = torch.stack([fp8_gemv_raw(W, x[b]) for b in range(B)])
        got = _run()
        torch.cuda.synchronize()
        if not fp16_equal_with_nan(got, ref):
            max_abs = (got.float() - ref.float()).abs().nan_to_num().max().item()
            raise AssertionError(f"raw batched mismatch B={B}, max_abs={max_abs}")
    for _ in range(5):
        _run()
    torch.cuda.synchronize()
    return triton.testing.do_bench(_run, warmup=50, rep=200)


def fp16_equal_with_nan(a: torch.Tensor, b: torch.Tensor) -> bool:
    return bool(torch.equal(a, b) or torch.equal(torch.nan_to_num(a), torch.nan_to_num(b)))


SEG_PER_ROW = 32


def bench_fused(
    M: int,
    K: int,
    *,
    epb: int | None = None,
    tile: int | None = None,
    rpt: int | None = None,
) -> tuple[float, float]:
    encode_tile = 8 if tile is None else tile
    exp_freqs = quantize_freqs(np.array(QWEN3_14B_FP8_EXP, dtype=np.float64))
    torch.manual_seed(42)
    W_cpu = random_fp8_bytes(M * K, device="cpu").reshape(M, K)

    # Segment each row into SEG_PER_ROW independent rANS streams so the
    # encoder produces M*SEG_PER_ROW short streams → enough parallelism.
    N_seg = K // SEG_PER_ROW
    W_seg = W_cpu.reshape(M, SEG_PER_ROW, N_seg).reshape(M * SEG_PER_ROW, N_seg)

    comp, states, offsets, sm, pf = encode(W_seg, exp_freqs, tile=encode_tile)
    sfc_t = build_sfc(pf)

    W_comp = comp.cuda()
    W_states = states.cuda()
    W_offsets = offsets.cuda()
    W_sm = sm.cuda()
    x = torch.randn(K, dtype=torch.half, device="cuda")

    raw_bytes = M * K
    comp_bytes = W_comp.numel() + W_sm.numel()
    comp_ratio = raw_bytes / comp_bytes

    old_epb = os.environ.get("RANS_GEMV_EPB")
    old_tile = os.environ.get("RANS_GEMV_TILE")
    old_rpt = os.environ.get("RANS_GEMV_RPT")
    if epb is None:
        os.environ.pop("RANS_GEMV_EPB", None)
    else:
        os.environ["RANS_GEMV_EPB"] = str(epb)
    if tile is None:
        os.environ.pop("RANS_GEMV_TILE", None)
    else:
        os.environ["RANS_GEMV_TILE"] = str(tile)
    if rpt is None:
        os.environ.pop("RANS_GEMV_RPT", None)
    else:
        os.environ["RANS_GEMV_RPT"] = str(rpt)

    def _run():
        fp8_gemv_fused(W_comp, W_offsets, W_states, W_sm, sfc_t, x, K)

    try:
        for _ in range(5):
            _run()
        torch.cuda.synchronize()
        ms = triton.testing.do_bench(_run, warmup=50, rep=200)
    finally:
        if old_epb is None:
            os.environ.pop("RANS_GEMV_EPB", None)
        else:
            os.environ["RANS_GEMV_EPB"] = old_epb
        if old_tile is None:
            os.environ.pop("RANS_GEMV_TILE", None)
        else:
            os.environ["RANS_GEMV_TILE"] = old_tile
        if old_rpt is None:
            os.environ.pop("RANS_GEMV_RPT", None)
        else:
            os.environ["RANS_GEMV_RPT"] = old_rpt
    return ms, comp_ratio


def bench_fused_batch(
    M: int,
    K: int,
    B: int,
    *,
    epb: int | None = None,
    tile: int | None = None,
    check: bool = False,
) -> tuple[float, float]:
    encode_tile = 8 if tile is None else tile
    exp_freqs = quantize_freqs(np.array(QWEN3_14B_FP8_EXP, dtype=np.float64))
    torch.manual_seed(42)
    W_cpu = random_fp8_bytes(M * K, device="cpu").reshape(M, K)
    N_seg = K // SEG_PER_ROW
    W_seg = W_cpu.reshape(M, SEG_PER_ROW, N_seg).reshape(M * SEG_PER_ROW, N_seg)
    comp, states, offsets, sm, pf = encode(W_seg, exp_freqs, tile=encode_tile)
    sfc_t = build_sfc(pf)

    W_comp = comp.cuda()
    W_states = states.cuda()
    W_offsets = offsets.cuda()
    W_sm = sm.cuda()
    x = torch.randn(B, K, dtype=torch.half, device="cuda")

    raw_bytes = M * K
    comp_bytes = W_comp.numel() + W_sm.numel()
    comp_ratio = raw_bytes / comp_bytes

    old_epb = os.environ.get("RANS_GEMV_EPB")
    old_tile = os.environ.get("RANS_GEMV_TILE")
    old_rpt = os.environ.get("RANS_GEMV_RPT")
    if epb is None:
        os.environ.pop("RANS_GEMV_EPB", None)
    else:
        os.environ["RANS_GEMV_EPB"] = str(epb)
    if tile is None:
        os.environ.pop("RANS_GEMV_TILE", None)
    else:
        os.environ["RANS_GEMV_TILE"] = str(tile)
    os.environ.pop("RANS_GEMV_RPT", None)

    def _run():
        return fp8_gemv_fused(W_comp, W_offsets, W_states, W_sm, sfc_t, x, K)

    try:
        if check:
            W_cuda = W_cpu.cuda()
            ref = torch.stack([fp8_gemv_raw(W_cuda, x[b]) for b in range(B)])
            got = _run()
            torch.cuda.synchronize()
            if not fp16_equal_with_nan(got, ref):
                max_abs = (got.float() - ref.float()).abs().nan_to_num().max().item()
                raise AssertionError(f"batched fused mismatch B={B}, max_abs={max_abs}")
        for _ in range(5):
            _run()
        torch.cuda.synchronize()
        ms = triton.testing.do_bench(_run, warmup=50, rep=200)
    finally:
        if old_epb is None:
            os.environ.pop("RANS_GEMV_EPB", None)
        else:
            os.environ["RANS_GEMV_EPB"] = old_epb
        if old_tile is None:
            os.environ.pop("RANS_GEMV_TILE", None)
        else:
            os.environ["RANS_GEMV_TILE"] = old_tile
        if old_rpt is None:
            os.environ.pop("RANS_GEMV_RPT", None)
        else:
            os.environ["RANS_GEMV_RPT"] = old_rpt
    return ms, comp_ratio


def run_default(shapes: list[tuple[int, int]]) -> None:
    # Shapes representative of Qwen3-style attention projections at decode.
    # (M, K): M = output dim, K = input dim. GEMV reads M*K FP8 weight bytes.
    print(f"{'M':>6s} {'K':>6s} {'W_MiB':>7s}  "
          f"{'raw_ms':>8s} {'raw_GB/s':>9s}  "
          f"{'fused_ms':>9s} {'fused_GB/s':>11s}  "
          f"{'ratio':>6s} {'comp':>5s}")
    print("-" * 85)
    for M, K in shapes:
        w_mib = M * K / (1 << 20)
        raw_ms = bench_raw(M, K)
        fused_ms, comp_ratio = bench_fused(M, K)
        # Effective bandwidth in terms of uncompressed weight bytes.
        raw_bytes = M * K
        raw_bw = raw_bytes / raw_ms / 1e6        # GB/s
        fused_bw = raw_bytes / fused_ms / 1e6    # GB/s equivalent
        ratio = raw_ms / fused_ms
        print(f"{M:>6d} {K:>6d} {w_mib:>7.1f}  "
              f"{raw_ms:>7.3f} {raw_bw:>8.1f}   "
              f"{fused_ms:>8.3f} {fused_bw:>10.1f}   "
              f"{ratio:>5.3f}x {comp_ratio:>4.2f}x")


def run_variants(M: int, K: int) -> None:
    raw_ms = bench_raw(M, K)
    raw_bytes = M * K
    print(f"\nVariant sweep for M={M}, K={K} (raw {raw_ms:.3f} ms)")
    print(f"{'RPT':>3s} {'EPB':>3s} {'tile':>4s} {'fused_ms':>9s} {'fused_GB/s':>11s} "
          f"{'raw/fused':>9s} {'comp':>5s}")
    print("-" * 59)
    for rpt in (1, 2):
        for tile in (8, 16, 32):
            n_pairs_per_segment = K // SEG_PER_ROW // 2
            if n_pairs_per_segment % tile != 0:
                print(f"{rpt:>3d} {'-':>3s} {tile:>4d} {'skip':>8s} "
                      f"{'n_pairs%tile':>10s} {'':>9s} {'':>5s}")
                continue
            for epb in (1, 2, 4, 8):
                fused_ms, comp_ratio = bench_fused(M, K, epb=epb, tile=tile, rpt=rpt)
                fused_bw = raw_bytes / fused_ms / 1e6
                print(f"{rpt:>3d} {epb:>3d} {tile:>4d} {fused_ms:>8.3f} {fused_bw:>10.1f} "
                      f"{raw_ms / fused_ms:>8.3f}x {comp_ratio:>4.2f}x")


def run_batch_variants(M: int, K: int) -> None:
    raw_ms = bench_raw(M, K)
    fused_b1_ms, comp_ratio = bench_fused(M, K)
    raw_bytes = M * K
    print(f"\nSmall-batch fair sweep for M={M}, K={K}")
    print(f"B=1 raw current: {raw_ms:.3f} ms, {raw_bytes / raw_ms / 1e6:.1f} GB/s")
    print(f"B=1 fused current: {fused_b1_ms:.3f} ms, {raw_bytes / fused_b1_ms / 1e6:.1f} GB/s, comp {comp_ratio:.2f}x")
    print(f"{'B':>2s} {'raw_ms':>8s} {'raw/vec':>8s} {'EPB':>3s} {'tile':>4s} "
          f"{'fused_ms':>9s} {'fused/vec':>10s} {'fused/raw':>9s} {'comp':>5s}")
    print("-" * 83)
    for B in (2, 4, 8):
        raw_b_ms = bench_raw_batch(M, K, B, check=True)
        best: tuple[float, int, int, float] | None = None
        for tile in (8, 16, 32):
            n_pairs_per_segment = K // SEG_PER_ROW // 2
            if n_pairs_per_segment % tile != 0:
                continue
            for epb in (1, 2, 4, 8):
                ms, comp_ratio = bench_fused_batch(M, K, B, epb=epb, tile=tile, check=True)
                if best is None or ms < best[0]:
                    best = (ms, epb, tile, comp_ratio)
        assert best is not None
        ms, epb, tile, comp_ratio = best
        print(f"{B:>2d} {raw_b_ms:>8.3f} {raw_b_ms / B:>8.3f} "
              f"{epb:>3d} {tile:>4d} {ms:>9.3f} {ms / B:>10.3f} "
              f"{raw_b_ms / ms:>8.3f}x {comp_ratio:>4.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        action="store_true",
        help="sweep RANS_GEMV_EPB and RANS_GEMV_TILE on one representative shape",
    )
    parser.add_argument(
        "--skip-default",
        action="store_true",
        help="only run the explicitly requested targeted sweeps",
    )
    parser.add_argument(
        "--batch-variants",
        action="store_true",
        help="sweep fused small-batch GEMV variants for B=2/4/8",
    )
    parser.add_argument("--variant-m", type=int, default=5120)
    parser.add_argument("--variant-k", type=int, default=5120)
    args = parser.parse_args()

    shapes = [
        ( 2048,  2048),
        ( 4096,  4096),
        ( 5120,  5120),   # Qwen3 14B hidden
        ( 8192,  5120),
        ( 5120,  8192),
    ]

    if not args.skip_default:
        run_default(shapes)
    if args.variants:
        run_variants(args.variant_m, args.variant_k)
    if args.batch_variants:
        run_batch_variants(args.variant_m, args.variant_k)


if __name__ == "__main__":
    main()

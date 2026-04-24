"""Benchmark: fused decompress + FP8 GEMV vs raw FP8 GEMV.

y = W @ x, W is [M, K] FP8, x is [K] FP16, y is [M] FP16.
"""
from __future__ import annotations

import numpy as np
import torch
import triton.testing

from rans_vectoradd._C import fp8_gemv_raw, fp8_gemv_fused
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


SEG_PER_ROW = 32


def bench_fused(M: int, K: int) -> tuple[float, float]:
    exp_freqs = quantize_freqs(np.array(QWEN3_14B_FP8_EXP, dtype=np.float64))
    torch.manual_seed(42)
    W_cpu = random_fp8_bytes(M * K, device="cpu").reshape(M, K)

    # Segment each row into SEG_PER_ROW independent rANS streams so the
    # encoder produces M*SEG_PER_ROW short streams → enough parallelism.
    N_seg = K // SEG_PER_ROW
    W_seg = W_cpu.reshape(M, SEG_PER_ROW, N_seg).reshape(M * SEG_PER_ROW, N_seg)

    comp, states, offsets, sm, pf = encode(W_seg, exp_freqs, tile=8)
    sfc_t = build_sfc(pf)

    W_comp = comp.cuda()
    W_states = states.cuda()
    W_offsets = offsets.cuda()
    W_sm = sm.cuda()
    x = torch.randn(K, dtype=torch.half, device="cuda")

    raw_bytes = M * K
    comp_bytes = W_comp.numel() + W_sm.numel()
    comp_ratio = raw_bytes / comp_bytes

    def _run():
        fp8_gemv_fused(W_comp, W_offsets, W_states, W_sm, sfc_t, x, K)

    for _ in range(5):
        _run()
    torch.cuda.synchronize()
    ms = triton.testing.do_bench(_run, warmup=50, rep=200)
    return ms, comp_ratio


def main() -> None:
    # Shapes representative of Qwen3-style attention projections at decode.
    # (M, K): M = output dim, K = input dim. GEMV reads M*K FP8 weight bytes.
    shapes = [
        ( 2048,  2048),
        ( 4096,  4096),
        ( 5120,  5120),   # Qwen3 14B hidden
        ( 8192,  5120),
        ( 5120,  8192),
    ]

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


if __name__ == "__main__":
    main()

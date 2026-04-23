"""Benchmark: fused decompress+vecadd vs raw FP8 vecadd.

Measures wall-clock time for both kernels at matched data sizes.
The fused kernel wins when decode overhead < HBM savings from compression.
"""
from __future__ import annotations

import numpy as np
import torch
import triton.testing

from rans_vectoradd._C import fp8_vecadd_raw, fp8_vecadd_fused
from rans_vectoradd import (
    QWEN3_14B_FP8_EXP,
    encode,
    quantize_freqs,
    random_fp8_bytes,
)


def bench_raw(n_bytes: int) -> dict:
    A = torch.randint(0, 256, (n_bytes,), dtype=torch.uint8, device="cuda")
    B = torch.randint(0, 256, (n_bytes,), dtype=torch.uint8, device="cuda")

    def _run():
        fp8_vecadd_raw(A, B)

    for _ in range(5):
        _run()
    torch.cuda.synchronize()
    ms = triton.testing.do_bench(_run, warmup=50, rep=200)
    s = ms / 1000.0
    # 2 reads + 1 write = 3N bytes
    hbm_bytes = 3 * n_bytes
    return {
        "ms": ms,
        "GB/s": hbm_bytes / 1e9 / s,
        "Gfp8/s": n_bytes / s / 1e9,
    }


def bench_fused(n_bytes: int, N: int, ns: int) -> dict:
    K = n_bytes // N
    exp_freqs = quantize_freqs(np.array(QWEN3_14B_FP8_EXP, dtype=np.float64))

    torch.manual_seed(42)
    fp8_a = random_fp8_bytes(K * N, device="cpu").reshape(K, N)
    fp8_b = random_fp8_bytes(K * N, device="cpu").reshape(K, N)

    tile = 8 if ns == 704 else 0
    ca, sa, oa, sma, pf = encode(fp8_a, exp_freqs, tile=tile)
    cb, sb, ob, smb, _  = encode(fp8_b, exp_freqs, tile=tile)

    # Build sfc table from pair_freqs
    pf_np = pf.numpy().astype(np.int32)
    cumul = np.zeros(len(pf_np) + 1, dtype=np.int32)
    cumul[1:] = np.cumsum(pf_np)
    M = int(cumul[-1])
    sfc = np.zeros(M, dtype=np.uint32)
    for sym in range(len(pf_np)):
        f = int(pf_np[sym])
        c = int(cumul[sym])
        for slot in range(c, c + f):
            sfc[slot] = np.uint32(sym | (f << 8) | (c << 20))
    sfc_t = torch.from_numpy(sfc.view(np.int32)).cuda()

    ca_g, sa_g, oa_g, sma_g = ca.cuda(), sa.cuda(), oa.cuda(), sma.cuda()
    cb_g, sb_g, ob_g, smb_g = cb.cuda(), sb.cuda(), ob.cuda(), smb.cuda()

    n_fp8 = N

    def _run():
        fp8_vecadd_fused(ca_g, oa_g, sa_g, sma_g,
                         cb_g, ob_g, sb_g, smb_g,
                         sfc_t, n_fp8, ns)

    for _ in range(5):
        _run()
    torch.cuda.synchronize()
    ms = triton.testing.do_bench(_run, warmup=50, rep=200)
    s = ms / 1000.0

    comp_bytes = ca.numel() + cb.numel() + sma.numel() + smb.numel()
    out_bytes = n_bytes
    hbm_bytes = comp_bytes + sa.numel() * 4 + sb.numel() * 4 + out_bytes
    ratio = (2 * n_bytes) / (ca.numel() + cb.numel() + sma.numel() + smb.numel())

    return {
        "ms": ms,
        "GB/s": hbm_bytes / 1e9 / s,
        "Gfp8/s": n_bytes / s / 1e9,
        "comp_ratio": ratio,
    }


def main() -> None:
    sizes = [
        (1 << 30, "1 GiB"),
    ]
    configs = [
        (504, "twopass"),
        (704, "tiled"),
    ]

    for N in [128, 256, 512, 1024, 2048]:
        print(f"\n{'=' * 72}")
        print(f"N={N}")
        print(f"{'=' * 72}")
        for n_bytes, size_label in sizes:
            r_raw = bench_raw(n_bytes)
            print(f"\n  {size_label}: raw = {r_raw['ms']:.3f} ms  "
                  f"{r_raw['GB/s']:.1f} GB/s  {r_raw['Gfp8/s']:.1f} Gfp8/s")
            for ns, label in configs:
                r = bench_fused(n_bytes, N, ns)
                ratio = r_raw["ms"] / r["ms"]
                print(f"    {label:<22} {r['ms']:.3f} ms  "
                      f"{ratio:.3f}x raw  "
                      f"{r['GB/s']:.1f} GB/s  "
                      f"comp {r['comp_ratio']:.2f}x")


if __name__ == "__main__":
    main()

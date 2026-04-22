"""Bench the compressed FP8 pairs decoder at N=128.

Figure of merit: fp8 bytes delivered to SMEM per GB of HBM read. The real
use case consumes decoded bytes from SMEM for downstream compute (e.g. MMA),
so HBM writes back to DRAM are benchmark scaffolding, not real cost. We use
the `_dump` kernel variant which folds decoded bytes into a per-thread XOR
digest (blockDim.x bytes per block, ~negligible HBM write) — it does exactly
the same decode work but without the artificial full-size output write.

Memcpy reference is the upper bound of "what if you skipped compression":
   1 fp8 byte delivered per 1 HBM byte read.
We want > 1.0x — more fp8 delivered per HBM byte than memcpy can manage.
"""
from __future__ import annotations

import numpy as np
import torch
import triton.testing

from rans_vectoradd import (
    QWEN3_14B_FP8_EXP,
    batch_encode_fp8,
    gpu_rans_decode_fp8_dump,
    quantize_freqs,
    random_fp8_bytes,
)

PEAK_GBPS = 1008.0


def bench_compressed(K: int, N: int, fp8_cpu: torch.Tensor) -> dict:
    probs = np.array(QWEN3_14B_FP8_EXP, dtype=np.float64)
    exp_freqs = quantize_freqs(probs)
    compressed, states, block_offsets, sm_packed, pair_freqs = batch_encode_fp8(
        fp8_cpu, exp_freqs
    )

    comp = compressed.cuda()
    off  = block_offsets.cuda()
    st   = states.cuda()
    sm   = sm_packed.cuda()

    def _run():
        gpu_rans_decode_fp8_dump(comp, off, st, sm, N, pair_freqs)

    for _ in range(3):
        _run()
    torch.cuda.synchronize()
    ms = triton.testing.do_bench(_run, warmup=50, rep=200)

    n_bytes = K * N  # total fp8 bytes "delivered"
    s = ms / 1000.0
    # HBM read: just the compressed inputs. final_states also but it's 4B/stream,
    # tiny and amortized — include for accuracy. digest write is ignored
    # (~2 MB, << total).
    hbm_read_bytes = int(comp.numel()) + int(sm.numel()) + int(st.numel()) * 4
    hbm_read_gbps  = hbm_read_bytes / 1e9 / s
    gfp8_per_s     = n_bytes / s / 1e9
    # "Compression advantage": delivered fp8 per HBM byte read. > 1.0 beats
    # memcpy per unit HBM.
    bytes_per_fp8 = hbm_read_bytes / n_bytes
    return {
        "ms": ms,
        "Gfp8/s": gfp8_per_s,
        "HBM read GB/s": hbm_read_gbps,
        "% peak read": 100 * hbm_read_gbps / PEAK_GBPS,
        "fp8/HBM-byte": 1.0 / bytes_per_fp8,
        "bits/fp8": bytes_per_fp8 * 8,
    }


def bench_memcpy_digest(n_bytes: int) -> dict:
    """Apples-to-apples reference: HBM-read-only fp8 load, folded into a
    digest. Same HBM access pattern as "load N fp8 bytes into SMEM for
    downstream MMA consumption" — no HBM write-back. This is the ceiling
    we're trying to beat."""
    from rans_vectoradd import fp8_memcpy_digest
    src = torch.empty(n_bytes, dtype=torch.uint8, device="cuda")

    def _run():
        fp8_memcpy_digest(src)

    for _ in range(3):
        _run()
    torch.cuda.synchronize()
    ms = triton.testing.do_bench(_run, warmup=50, rep=200)
    s = ms / 1000.0
    hbm_read_gbps = n_bytes / 1e9 / s
    return {
        "ms": ms,
        "Gfp8/s": n_bytes / s / 1e9,
        "HBM read GB/s": hbm_read_gbps,
        "% peak read": 100 * hbm_read_gbps / PEAK_GBPS,
        "fp8/HBM-byte": 1.0,
        "bits/fp8": 8.0,
    }


def fmt(name: str, r: dict) -> str:
    return (
        f"  {name:<24} {r['ms']:>7.3f} ms  "
        f"{r['Gfp8/s']:>7.1f} Gfp8/s  "
        f"read {r['HBM read GB/s']:>6.1f} GB/s ({r['% peak read']:>5.1f}% peak)  "
        f"{r['fp8/HBM-byte']:>5.3f} fp8/HBM-B  "
        f"({r['bits/fp8']:.2f} bits/fp8)"
    )


def main() -> None:
    bytes_out = 1 << 29  # 512 MiB — fits around other GPU tenants
    print(f"\n=== {bytes_out / (1<<20):.0f} MiB of FP8 delivered ===")

    r_cpy = bench_memcpy_digest(bytes_out)
    print(fmt("memcpy-digest (ref)", r_cpy))

    for N in (128, 256, 512, 1024):
        K = bytes_out // N
        torch.manual_seed(42)
        fp8 = random_fp8_bytes(K * N, device="cpu").reshape(K, N)
        r_dec = bench_compressed(K, N, fp8)
        beat = r_dec["Gfp8/s"] / r_cpy["Gfp8/s"]
        ceiling = r_cpy["HBM read GB/s"] * r_dec["fp8/HBM-byte"]
        note = (f"vs cpy {beat:.2f}x, HBM-bound ceil {ceiling:.0f} "
                f"({ceiling/r_cpy['Gfp8/s']:.2f}x cpy)")
        print(fmt(f"rans-decode N={N}", r_dec) + "  " + note)


if __name__ == "__main__":
    main()

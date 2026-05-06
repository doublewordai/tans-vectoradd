"""Correctness test for fused tANS decompress + FP8 vecadd."""
from __future__ import annotations

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


def untile(out_flat: torch.Tensor, K: int, N: int) -> torch.Tensor:
    n_tiles = (N // 2) // TILE_PAIRS
    return (
        out_flat.view(n_tiles, K, TILE_PAIRS * 2)
        .permute(1, 0, 2)
        .reshape(K, N)
        .contiguous()
    )


def run_case(K: int, N: int, seed: int = 0) -> None:
    exp_freqs = torch.tensor(QWEN3_14B_FP8_EXP, dtype=torch.float64)
    torch.manual_seed(seed)
    fp8_a = random_fp8_bytes(K * N, device="cpu").reshape(K, N)
    fp8_b = random_fp8_bytes(K * N, device="cpu").reshape(K, N)

    ca, sa, oa, pa, sma, pf, sp = tans_codec.encode(fp8_a, exp_freqs, tile=TILE_PAIRS)
    cb, sb, ob, pb, smb, _, _ = tans_codec.encode(fp8_b, exp_freqs, tile=TILE_PAIRS)
    decode_tbl = tans_codec.build_decode_table(sp.numpy(), pf.numpy())

    args = [t.cuda() for t in (ca, oa, sa, pa, sma, cb, ob, sb, pb, smb, decode_tbl)]
    out_raw = fp8_vecadd_raw(fp8_a.flatten().cuda(), fp8_b.flatten().cuda())
    out_raw = out_raw.cpu().reshape(K, N)

    for variant, fn in [
        ("shared", fp8_vecadd_fused_tans_shared),
        ("register", fp8_vecadd_fused_tans_register),
    ]:
        out_flat = fn(*args, N)
        out_fused = untile(out_flat.cpu(), K, N)

        mismatches = (out_fused != out_raw).sum().item()
        status = "OK " if mismatches == 0 else "FAIL"
        print(
            f"K={K:>6} N={N:>5}  {variant:<8}  {status}  "
            f"mismatched bytes: {mismatches}"
        )
        if mismatches:
            bad = (out_fused != out_raw).nonzero()[:5]
            for row, col in bad.tolist():
                print(
                    f"  [{row},{col}] fused={out_fused[row, col].item()} "
                    f"raw={out_raw[row, col].item()}"
                )
            raise SystemExit(1)


def main() -> None:
    print("Fused tANS vecadd correctness:\n")
    for K, N in [
        (1024, 128),
        (10_000, 256),
        (1024, 1024),
    ]:
        run_case(K, N)
    print("\nAll passed.")


if __name__ == "__main__":
    main()

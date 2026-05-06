"""Round-trip test for the GPU tANS decoder. Encode FP8 bytes on CPU,
decode on GPU, verify the output matches the original."""
from __future__ import annotations

import time

import torch

from rans_vectoradd import (
    QWEN3_14B_FP8_EXP,
    gpu_tans_decode,
    random_fp8_bytes,
    tans_codec,
)


def run_case(K: int, N: int, seed: int = 42) -> None:
    exp_freqs = torch.tensor(QWEN3_14B_FP8_EXP, dtype=torch.float64)

    torch.manual_seed(seed)
    fp8 = random_fp8_bytes(K * N, device="cpu").reshape(K, N)

    t0 = time.perf_counter()
    compressed, states, offsets, partial_cnts, sm_packed, pair_freqs, spread = \
        tans_codec.encode(fp8, exp_freqs)
    t_enc = time.perf_counter() - t0

    compressed_g = compressed.cuda()
    states_g     = states.cuda()
    offsets_g    = offsets.cuda()
    partial_g    = partial_cnts.cuda()
    sm_g         = sm_packed.cuda()
    spread_g     = spread.cuda()
    freqs_g      = pair_freqs.cuda()

    comp_bytes = int(compressed.numel())
    sm_bytes   = int(sm_packed.numel())
    bits_per_fp8 = (comp_bytes + sm_bytes) * 8 / (K * N)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    decoded_gpu = gpu_tans_decode(
        compressed_g, offsets_g, states_g, partial_g, sm_g, spread_g, freqs_g, N
    )
    torch.cuda.synchronize()
    t_dec = time.perf_counter() - t0
    decoded = decoded_gpu.cpu().t().contiguous()
    ok = torch.equal(decoded, fp8)
    status = "OK " if ok else "FAIL"
    print(
        f"K={K:>7} N={N:>5}  {status}  "
        f"comp={comp_bytes:>11,}B  sm={sm_bytes:>11,}B  "
        f"bits/fp8={bits_per_fp8:.3f}  "
        f"enc={t_enc * 1000:>6.1f}ms  dec_gpu={t_dec * 1000:>6.2f}ms"
    )
    if not ok:
        diff = (decoded != fp8).any(dim=1).nonzero().flatten()[:5]
        print(f"  first mismatching rows: {diff.tolist()}")
        for k in diff[:3].tolist():
            mismatch = (decoded[k] != fp8[k]).nonzero().flatten()[:10]
            print(f"  row {k}: bytes {mismatch.tolist()}, "
                  f"got {decoded[k, mismatch].tolist()}, "
                  f"want {fp8[k, mismatch].tolist()}")
        raise SystemExit(1)


def main() -> None:
    print("FP8 tANS decompressor round-trip:\n")
    for K, N in [
        (128, 128),
        (1_024, 128),
        (10_000, 128),
        (100_000, 128),
    ]:
        run_case(K, N)
    print("\nAll passed.")


if __name__ == "__main__":
    main()

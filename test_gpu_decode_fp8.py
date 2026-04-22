"""Round-trip test for the FP8 decompressor. Encode on CPU, decode on GPU,
check the output exactly matches the original fp8 bytes."""
from __future__ import annotations

import time

import numpy as np
import torch

from rans_vectoradd import (
    QWEN3_14B_FP8_EXP,
    batch_encode_fp8,
    gpu_rans_decode_fp8,
    quantize_freqs,
    random_fp8_bytes,
)


def run_case(K: int, N: int, seed: int = 42) -> None:
    exp_freqs = quantize_freqs(np.array(QWEN3_14B_FP8_EXP, dtype=np.float64))

    torch.manual_seed(seed)
    fp8 = random_fp8_bytes(K * N, device="cpu").reshape(K, N)

    t0 = time.perf_counter()
    compressed, states, block_offsets, sm_packed = batch_encode_fp8(fp8, exp_freqs)
    t_enc = time.perf_counter() - t0

    compressed_gpu = compressed.cuda()
    states_gpu     = states.cuda()
    offsets_gpu    = block_offsets.cuda()
    sm_gpu         = sm_packed.cuda()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    decoded_gpu = gpu_rans_decode_fp8(
        compressed_gpu, offsets_gpu, states_gpu, sm_gpu, N, exp_freqs
    )
    torch.cuda.synchronize()
    t_dec = time.perf_counter() - t0

    # Kernel writes [N, K]; transpose to [K, N] to compare against source.
    decoded = decoded_gpu.cpu().t().contiguous()
    ok = torch.equal(decoded, fp8)

    comp_bytes = int(compressed.numel())
    sm_bytes   = int(sm_packed.numel())
    bits_per_fp8 = (comp_bytes + sm_bytes) * 8 / (K * N)

    status = "✓" if ok else "FAIL"
    print(
        f"K={K:>7} N={N:>5}  {status}  "
        f"comp={comp_bytes:>11,}B  sm={sm_bytes:>11,}B  "
        f"bits/fp8={bits_per_fp8:.3f}  "
        f"enc={t_enc * 1000:>7.1f}ms  dec_gpu={t_dec * 1000:>6.2f}ms"
    )

    if not ok:
        diff = (decoded != fp8).any(dim=1).nonzero().flatten()[:5]
        print(f"  first mismatching rows: {diff.tolist()}")
        raise SystemExit(1)


def main() -> None:
    print("FP8 decompressor round-trip:\n")
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

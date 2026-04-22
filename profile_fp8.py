"""Minimal script for ncu profiling. Runs the pairs-dump kernel several times
at the active N-of-interest so ncu can skip warmups and capture a steady-state
invocation. Scales down automatically to fit limited GPU memory.
"""
import numpy as np
import torch

from rans_vectoradd import (
    QWEN3_14B_FP8_EXP,
    batch_encode_fp8,
    gpu_rans_decode_fp8_dump,
    quantize_freqs,
    random_fp8_bytes,
)


def main() -> None:
    # Smaller K so we fit around other GPU tenants; still enough waves to
    # measure steady-state (128 SMs × ~11 blocks/SM = ~1400 blocks resident,
    # we want a grid many times larger).
    N = 512
    K = (1 << 28) // N   # 256 MiB of fp8 — matches bench wave count
    probs = np.array(QWEN3_14B_FP8_EXP, dtype=np.float64)
    exp_freqs = quantize_freqs(probs)
    torch.manual_seed(0)
    fp8 = random_fp8_bytes(K * N, device="cpu").reshape(K, N)

    compressed, states, offsets, sm = batch_encode_fp8(fp8, exp_freqs)
    comp = compressed.cuda(); off = offsets.cuda()
    st = states.cuda(); sm = sm.cuda()

    torch.cuda.synchronize()
    for _ in range(20):
        gpu_rans_decode_fp8_dump(comp, off, st, sm, N, exp_freqs)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()

"""Minimal script for ncu profiling. Runs the decoder several times
so ncu can skip warmups and capture a steady-state invocation.
"""
import numpy as np
import torch

from rans_vectoradd import (
    QWEN3_14B_FP8_EXP,
    encode,
    gpu_rans_decode_dump,
    quantize_freqs,
    random_fp8_bytes,
)


def main() -> None:
    N = 512
    K = (1 << 28) // N  # 256 MiB of fp8
    probs = np.array(QWEN3_14B_FP8_EXP, dtype=np.float64)
    exp_freqs = quantize_freqs(probs)
    torch.manual_seed(0)
    fp8 = random_fp8_bytes(K * N, device="cpu").reshape(K, N)

    compressed, states, offsets, sm, pair_freqs = encode(fp8, exp_freqs)
    comp = compressed.cuda()
    off = offsets.cuda()
    st = states.cuda()
    sm = sm.cuda()

    torch.cuda.synchronize()
    for _ in range(20):
        gpu_rans_decode_dump(comp, off, st, sm, N, pair_freqs)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()

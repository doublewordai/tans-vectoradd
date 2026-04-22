"""Minimal script for ncu profiling. Runs the pairs-dump kernel several times
at the active N-of-interest so ncu can skip warmups and capture a steady-state
invocation. Scales down automatically to fit limited GPU memory.
"""
import numpy as np
import torch

from rans_vectoradd import (
    QWEN3_14B_FP8_EXP,
    batch_encode_fp8,
    batch_encode_fp8_pairs,
    gpu_rans_decode_fp8_dump,
    gpu_rans_decode_fp8_ldg_dump,
    gpu_rans_decode_fp8_pair_bl_dump,
    gpu_rans_decode_fp8_pair_ldg_dump,
    quantize_freqs,
    random_fp8_bytes,
)


def main() -> None:
    N = 512
    K = (1 << 28) // N   # 256 MiB of fp8
    probs = np.array(QWEN3_14B_FP8_EXP, dtype=np.float64)
    exp_freqs = quantize_freqs(probs)
    torch.manual_seed(0)
    fp8 = random_fp8_bytes(K * N, device="cpu").reshape(K, N)

    # Single-nibble encoded data
    compressed, states, offsets, sm = batch_encode_fp8(fp8, exp_freqs)
    comp = compressed.cuda(); off = offsets.cuda()
    st = states.cuda(); sm = sm.cuda()

    # Pair-encoded data (separate compressed stream + freqs)
    comp_p, st_p, off_p, sm_p, pair_freqs = batch_encode_fp8_pairs(fp8, exp_freqs)
    comp_p = comp_p.cuda(); off_p = off_p.cuda()
    st_p = st_p.cuda(); sm_p = sm_p.cuda()

    torch.cuda.synchronize()
    for _ in range(20):
        gpu_rans_decode_fp8_dump(comp, off, st, sm, N, exp_freqs)
    for _ in range(20):
        gpu_rans_decode_fp8_ldg_dump(comp, off, st, sm, N, exp_freqs)
    for _ in range(20):
        gpu_rans_decode_fp8_pair_ldg_dump(comp_p, off_p, st_p, sm_p, N, pair_freqs)
    for _ in range(20):
        gpu_rans_decode_fp8_pair_bl_dump(comp_p, off_p, st_p, sm_p, N, pair_freqs)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()

"""Run fp8_vecadd_fused at one N value for ncu profiling.

Usage: profile_vecadd.py <N>
"""
import sys
import numpy as np
import torch

from rans_vectoradd._C import fp8_vecadd_fused
from rans_vectoradd import (
    QWEN3_14B_FP8_EXP,
    encode,
    quantize_freqs,
    random_fp8_bytes,
)


def main() -> None:
    N = int(sys.argv[1])
    n_bytes = 1 << 30  # 1 GiB
    K = n_bytes // N

    exp_freqs = quantize_freqs(np.array(QWEN3_14B_FP8_EXP, dtype=np.float64))
    torch.manual_seed(42)
    fp8_a = random_fp8_bytes(K * N, device="cpu").reshape(K, N)
    fp8_b = random_fp8_bytes(K * N, device="cpu").reshape(K, N)

    ca, sa, oa, sma, pf = encode(fp8_a, exp_freqs, tile=8)
    cb, sb, ob, smb, _  = encode(fp8_b, exp_freqs, tile=8)

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

    torch.cuda.synchronize()
    # warmups (skipped by --launch-skip)
    for _ in range(5):
        fp8_vecadd_fused(ca_g, oa_g, sa_g, sma_g, cb_g, ob_g, sb_g, smb_g,
                         sfc_t, N, 835)
    torch.cuda.synchronize()
    # profiled launches
    for _ in range(3):
        fp8_vecadd_fused(ca_g, oa_g, sa_g, sma_g, cb_g, ob_g, sb_g, smb_g,
                         sfc_t, N, 835)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()

"""Correctness tests for raw and fused FP8 GEMV kernels."""
from __future__ import annotations

import numpy as np
import torch

from rans_vectoradd import QWEN3_14B_FP8_EXP, encode, quantize_freqs
from rans_vectoradd._C import fp8_gemv_fused, fp8_gemv_raw, fp8_gemv_raw_batch


def build_sfc(pair_freqs: torch.Tensor) -> torch.Tensor:
    pf_np = pair_freqs.numpy().astype(np.int32)
    cumul = np.zeros(len(pf_np) + 1, dtype=np.int32)
    cumul[1:] = np.cumsum(pf_np)
    sfc = np.zeros(int(cumul[-1]), dtype=np.uint32)
    for sym, f in enumerate(pf_np):
        c = int(cumul[sym])
        sfc[c : c + int(f)] = np.uint32(sym | (int(f) << 8) | (c << 20))
    return torch.from_numpy(sfc.view(np.int32)).cuda()


def finite_fp8_bytes(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    # torch.float8_e4m3fn treats 0x7f and 0xff as NaN. Avoid them so the
    # reference comparison has deterministic finite arithmetic.
    torch.manual_seed(seed)
    valid = torch.tensor(
        [i for i in range(256) if i not in (0x7F, 0xFF)], dtype=torch.uint8
    )
    return valid[torch.randint(valid.numel(), shape)]


def reference_gemv(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return (
        W.view(torch.float8_e4m3fn).to(torch.float32)
        * x.to(torch.float32).unsqueeze(0)
    ).sum(dim=1).to(torch.float16)


def test_raw_gemv_matches_torch_reference() -> None:
    M, K = 16, 512
    W = finite_fp8_bytes((M, K), seed=0).cuda()
    x = torch.randn(K, dtype=torch.float16, device="cuda")

    y = fp8_gemv_raw(W, x)
    ref = reference_gemv(W, x)
    torch.cuda.synchronize()

    assert torch.equal(y, ref)


def test_fused_gemv_matches_raw_gemv() -> None:
    M, K = 16, 512
    seg_per_row = 32
    W = finite_fp8_bytes((M, K), seed=1)
    x = torch.randn(K, dtype=torch.float16, device="cuda")

    W_seg = W.reshape(M, seg_per_row, K // seg_per_row).reshape(
        M * seg_per_row, K // seg_per_row
    )
    exp_freqs = quantize_freqs(np.array(QWEN3_14B_FP8_EXP, dtype=np.float64))
    comp, states, offsets, sm, pair_freqs = encode(W_seg, exp_freqs, tile=8)

    raw = fp8_gemv_raw(W.cuda(), x)
    fused = fp8_gemv_fused(
        comp.cuda(),
        offsets.cuda(),
        states.cuda(),
        sm.cuda(),
        build_sfc(pair_freqs),
        x,
        K,
    )
    torch.cuda.synchronize()

    assert torch.equal(fused, raw)


def test_fused_batch_gemv_matches_raw_gemv() -> None:
    M, K = 16, 512
    seg_per_row = 32
    W = finite_fp8_bytes((M, K), seed=3)
    x = torch.randn(8, K, dtype=torch.float16, device="cuda")

    W_seg = W.reshape(M, seg_per_row, K // seg_per_row).reshape(
        M * seg_per_row, K // seg_per_row
    )
    exp_freqs = quantize_freqs(np.array(QWEN3_14B_FP8_EXP, dtype=np.float64))
    comp, states, offsets, sm, pair_freqs = encode(W_seg, exp_freqs, tile=8)
    sfc = build_sfc(pair_freqs)

    W_cuda = W.cuda()
    for B in (2, 4, 8):
        raw = fp8_gemv_raw_batch(W_cuda, x[:B].contiguous())
        raw_loop = torch.stack([fp8_gemv_raw(W_cuda, x[b]) for b in range(B)])
        torch.cuda.synchronize()
        assert torch.equal(raw, raw_loop), B
        fused = fp8_gemv_fused(
            comp.cuda(),
            offsets.cuda(),
            states.cuda(),
            sm.cuda(),
            sfc,
            x[:B].contiguous(),
            K,
        )
        torch.cuda.synchronize()
        assert torch.equal(fused, raw), B


if __name__ == "__main__":
    test_raw_gemv_matches_torch_reference()
    test_fused_gemv_matches_raw_gemv()
    test_fused_batch_gemv_matches_raw_gemv()
    print("GEMV correctness tests passed.")

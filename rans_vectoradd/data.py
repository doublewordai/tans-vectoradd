"""Representative FP8 data generators.

For the raw (uncompressed) vector-add benchmark the data content doesn't
matter: HBM bandwidth is content-independent. But compressed variants
will measure effective bandwidth through a decompressor whose output rate
depends on the compression ratio, which depends on the actual entropy
distribution of the input. Pure-random bytes are incompressible by
construction, so they would make any compressed variant look no better
than the raw path. We need data that compresses like the real thing.

``QWEN3_14B_FP8_EXP`` is the aggregate exponent distribution measured
across all FP8 linear weights in Qwen3-14B-FP8 (13.2 GB of weights, 280
tensors). Exponent entropy ~2.787 bits. See ``analyze_fp8.py``.
"""
from __future__ import annotations

import torch

# Measured 2026-04-21 on Qwen3-14B-FP8, aggregated across all 280 FP8
# tensors (13,212,057,600 bytes). Exponent entropy = 2.787 bits / 4.
QWEN3_14B_FP8_EXP: tuple[float, ...] = (
    0.0005,  # exp=0
    0.0005,  # exp=1
    0.0009,  # exp=2
    0.0018,  # exp=3
    0.0034,  # exp=4
    0.0062,  # exp=5
    0.0104,  # exp=6
    0.0167,  # exp=7
    0.0283,  # exp=8
    0.0480,  # exp=9
    0.0838,  # exp=10
    0.1502,  # exp=11
    0.2417,  # exp=12
    0.2715,  # exp=13
    0.1273,  # exp=14
    0.0088,  # exp=15
)


def random_fp8_bytes(
    n: int,
    *,
    exp_probs: tuple[float, ...] = QWEN3_14B_FP8_EXP,
    device: str | torch.device = "cuda",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate N bytes of FP8 E4M3 with a specified exponent distribution.

    Sign+mantissa (4 bits) sampled uniformly — empirical entropy ~3.979/4.
    Exponent (4 bits) sampled from ``exp_probs`` — defaults to Qwen3-14B-FP8.

    Returns a uint8 tensor on ``device``.
    """
    probs = torch.tensor(exp_probs, dtype=torch.float32, device=device)
    probs = probs / probs.sum()
    cdf = torch.cumsum(probs, dim=0)

    u = torch.rand(n, device=device, generator=generator)
    exp = torch.searchsorted(cdf, u).clamp_max_(15).to(torch.uint8)

    # Uniform over the 16 possible (sign, mantissa) combinations.
    sm = torch.randint(
        0, 16, (n,), dtype=torch.uint8, device=device, generator=generator
    )
    sign = (sm >> 3) & 1
    mantissa = sm & 7

    return (sign << 7) | (exp << 3) | mantissa

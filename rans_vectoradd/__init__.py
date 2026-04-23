# Import torch first so its shared libraries (libc10.so, libtorch.so) are
# loaded before our extension tries to link against them.
import torch  # noqa: F401

from rans_vectoradd._C import (
    fp8_memcpy_digest,
    gpu_rans_decode_fp8,
    gpu_rans_decode_fp8_dump,
    gpu_rans_decode_fp8_ldg,
    gpu_rans_decode_fp8_ldg_dump,
    gpu_rans_decode_fp8_pair_bl,
    gpu_rans_decode_fp8_pair_bl_dump,
    gpu_rans_decode_fp8_triple,
    gpu_rans_decode_fp8_triple_dump,
    gpu_tans_decode_fp8,
    gpu_tans_decode_fp8_dump,
    gpu_rans_decode_fp8_pair_ldg,
    gpu_rans_decode_fp8_pair_ldg_dump,
    gpu_rans_decode_fp8_pair_ldg_q4,
    gpu_rans_decode_fp8_pair_ldg_q4_dump,
    gpu_rans_decode_fp8_regscan,
    gpu_rans_decode_fp8_regscan_dump,
)
from rans_vectoradd.codec import (
    BLOCK_STREAMS,
    batch_encode_fp8,
    batch_encode_fp8_pairs,
    batch_encode_fp8_triples,
    quantize_freqs,
)
from rans_vectoradd.codec import M_TRIPLE  # noqa: F401
from rans_vectoradd.data import QWEN3_14B_FP8_EXP, random_fp8_bytes

__all__ = [
    "BLOCK_STREAMS",
    "QWEN3_14B_FP8_EXP",
    "batch_encode_fp8",
    "batch_encode_fp8_pairs",
    "fp8_memcpy_digest",
    "gpu_rans_decode_fp8",
    "gpu_rans_decode_fp8_dump",
    "gpu_rans_decode_fp8_ldg",
    "gpu_rans_decode_fp8_ldg_dump",
    "gpu_rans_decode_fp8_pair_bl",
    "gpu_rans_decode_fp8_pair_bl_dump",
    "gpu_rans_decode_fp8_triple",
    "gpu_rans_decode_fp8_triple_dump",
    "batch_encode_fp8_triples",
    "gpu_tans_decode_fp8",
    "gpu_tans_decode_fp8_dump",
    "gpu_rans_decode_fp8_triple",
    "gpu_rans_decode_fp8_triple_dump",
    "gpu_rans_decode_fp8_pair_ldg",
    "gpu_rans_decode_fp8_pair_ldg_dump",
    "gpu_rans_decode_fp8_pair_ldg_q4",
    "gpu_rans_decode_fp8_pair_ldg_q4_dump",
    "gpu_rans_decode_fp8_regscan",
    "gpu_rans_decode_fp8_regscan_dump",
    "quantize_freqs",
    "random_fp8_bytes",
]

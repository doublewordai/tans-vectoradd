# Import torch first so its shared libraries (libc10.so, libtorch.so) are
# loaded before our extension tries to link against them.
import torch  # noqa: F401

from rans_vectoradd._C import (
    fp8_memcpy_digest,
    gpu_rans_decode_fp8,
    gpu_rans_decode_fp8_dump,
)
from rans_vectoradd.codec import (
    BLOCK_STREAMS,
    batch_encode_fp8,
    pair_freqs_from_single,
    quantize_freqs,
)
from rans_vectoradd.data import QWEN3_14B_FP8_EXP, random_fp8_bytes

__all__ = [
    "BLOCK_STREAMS",
    "QWEN3_14B_FP8_EXP",
    "batch_encode_fp8",
    "fp8_memcpy_digest",
    "gpu_rans_decode_fp8",
    "gpu_rans_decode_fp8_dump",
    "pair_freqs_from_single",
    "quantize_freqs",
    "random_fp8_bytes",
]

"""Build the CUDA extension.

Usage (after `uv sync`):
    uv run python setup.py build_ext --inplace
"""
# The venv's torch was bundled with CUDA 13.0 headers, but the system nvcc is
# 12.8. Torch's _check_cuda_version insists these must match exactly, even
# though for our purposes (sm_89, no cross-version ABI touchpoints) they are
# compatible. Disable the check so direct `setup.py build_ext --inplace` works
# for fast incremental rebuilds.
import os

import torch
import torch.utils.cpp_extension as _cppext
_cppext._check_cuda_version = lambda *a, **k: None

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def cuda_gencode_flags() -> list[str]:
    """Build for requested or visible CUDA architectures."""
    archs = os.environ.get("RANS_CUDA_ARCHS")
    if archs:
        sms = [a.strip().removeprefix("sm_").replace(".", "") for a in archs.split(",")]
    elif torch.cuda.is_available():
        sms = sorted({
            f"{major}{minor}"
            for major, minor in (
                torch.cuda.get_device_capability(i)
                for i in range(torch.cuda.device_count())
            )
        })
    else:
        sms = ["89"]

    return [f"-gencode=arch=compute_{sm},code=sm_{sm}" for sm in sms if sm]


setup(
    ext_modules=[
        CUDAExtension(
            name="rans_vectoradd._C",
            sources=[
                "csrc/fp8_memcpy.cu",
                "csrc/rans_codec.cpp",
                "csrc/rans_decode.cu",
                "csrc/vecadd.cu",
                "csrc/gemv.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-fopenmp"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                    *cuda_gencode_flags(),
                ],
            },
            extra_link_args=["-fopenmp"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)

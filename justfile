set dotenv-load := false

py := ".venv/bin/python"

# Fast inner loop: rebuild, verify correctness, bench the decoder.
iterate: build test bench

# Incremental CUDA/C++ rebuild (seconds, not minutes).
build:
    CC=gcc CXX=g++ {{py}} setup.py build_ext --inplace

# Minimal round-trip test — confirms the kernel still decodes correctly.
test:
    {{py}} test_gpu_decode_fp8.py

# Throughput bench: rans-decode (dump) vs HBM-read-only memcpy reference.
bench:
    {{py}} bench_fp8_compare.py

# Kernel profile via ncu — requires sudo for GPU perf counters.
profile:
    sudo -n env PATH=$PATH ncu --target-processes all \
        --kernel-name regex:rans_decode_fp8 \
        --launch-skip 5 --launch-count 1 \
        --set detailed \
        $(pwd)/{{py}} profile_fp8.py

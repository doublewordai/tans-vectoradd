set dotenv-load := false

py := ".venv/bin/python"

build:
    CC=gcc CXX=g++ {{py}} setup.py build_ext --inplace

test:
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} {{py}} test_gpu_decode_tans.py
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} {{py}} test_vecadd_tans.py

bench:
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} {{py}} bench_vecadd.py

profile N="1024":
    sudo -n env PATH=$PATH CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
        ncu --target-processes all \
        --kernel-name regex:fp8_vecadd_fused_tans \
        --launch-skip 5 --launch-count 3 \
        --set detailed \
        $(pwd)/{{py}} profile_vecadd.py {{N}}

iterate: build test bench

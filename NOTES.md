# tANS FP8 Vector Add

This repo is a small CUDA harness for testing bandwidth amplification from
lossless FP8 compression.

The active path is:

1. Generate FP8-like bytes with the Qwen3-14B-FP8 exponent distribution.
2. Split each byte into exponent and sign+mantissa nibbles.
3. tANS-code exponent pairs on CPU.
4. Decode the exponent pairs inside a fused CUDA kernel, reconstruct FP8
   bytes, add two tensors as E4M3, and write the result.

The toy workload is vector add. It is deliberately memory-bound, so beating
the raw FP8 vector-add baseline is evidence that the fused decompressor is
recovering more logical FP8 bytes per second than raw HBM can deliver.

## Files

- `rans_vectoradd/tans_codec.py` - Python-side tANS table construction and
  FP8 pair-stream preparation.
- `csrc/tans_codec.cpp` - CPU tANS encoder into the GPU slab layout.
- `csrc/tans_decode.cu` - standalone GPU tANS decoder used by the round-trip
  test.
- `csrc/vecadd.cu` - raw FP8 vecadd baseline and the fused tANS vecadd kernel.
- `bench_vecadd.py` - local benchmark.
- `profile_vecadd.py` - minimal Nsight Compute driver.

## Current 4090 Baseline

On this machine, `CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench_vecadd.py`
previously measured raw FP8 vecadd at roughly 908 GB/s. The fused tANS path
was faster than raw for the main sweep, peaking at about 1.10x raw around
`N=1024`.

## Commands

```bash
just build
just test
just bench
```

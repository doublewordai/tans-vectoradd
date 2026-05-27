# rans-vectoradd

A small CUDA harness for measuring **bandwidth amplification** from in-kernel
lossless decompression of FP8 weights, accompanying the blog post
[*Pushing memory bound kernels beyond the speed of light with lossless decompression*](https://fergusfinn.com/blog/faster-than-speed-of-light/).

## The experiment

The toy workload is FP8 vector add. It is deliberately memory-bound: two reads
and one write per FLOP, which puts it firmly into the regime where HBM
bandwidth is the binding constraint.

The fused kernel reads each input as a tANS-compressed stream of FP8 exponent
nibbles, decodes them on-chip, reconstructs the FP8 bytes, and performs the
add. On an RTX 4090 the fused kernel runs at roughly 1.10× the uncompressed
baseline at ~6.95 bits per FP8 byte, with decode overhead essentially zero.
The decoder is encoding-aware: only the exponent nibble is tANS-coded (the
sign and mantissa bits are close to uniform across modern model weights and
pass through unchanged).

## Quickstart

```bash
uv sync           # install pinned PyTorch + Triton
just build        # compile the CUDA extension
just test         # decoder round-trip + fused vecadd correctness
just bench        # raw vs fused on a 1 GiB working set
```

`just bench` prints the raw FP8 vecadd baseline followed by the fused kernel
at a sweep of stream lengths; the headline number is the `× raw` column.

## File map

- `csrc/vecadd.cu` — `fp8_vecadd_raw_kernel` (uncompressed baseline) and
  `fp8_vecadd_fused_tans_kernel` (the fused decode-and-add).
- `csrc/tans_codec.cpp` — CPU-side tANS encoder into the GPU slab layout.
- `csrc/tans_decode.cu` — standalone GPU tANS decoder, exercised by the
  round-trip test.
- `rans_vectoradd/tans_codec.py` — Python-side table construction and
  FP8 pair-stream preparation.
- `rans_vectoradd/data.py` — Qwen3-14B-FP8 exponent distribution used as
  the encoding source.
- `bench_vecadd.py` — local benchmark sweep.
- `profile_vecadd.py` — minimal Nsight Compute driver.

## License

MIT, see [LICENSE](LICENSE).

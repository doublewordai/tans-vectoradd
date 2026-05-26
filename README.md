# rans-vectoradd

A small CUDA harness for measuring **bandwidth amplification** from in-kernel
lossless decompression of FP8 weights, accompanying the blog post
[*rANS on GPUs: at-bandwidth decoding*](https://fergusfinn.com/blog/gpu-rans-decoding/).

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

- `csrc/vecadd.cu` — the kernels. `fp8_vecadd_raw_kernel` is the
  uncompressed baseline. There are two fused variants:
  `fp8_vecadd_fused_tans_shared_kernel` (shared-memory symbol staging) and
  `fp8_vecadd_fused_tans_register_kernel` (register-resident staging). On
  the 4090, `shared` is the one that hits the 1.10× number quoted above;
  `register` was being tuned for GH200 when this branch settled and is
  slightly off-pace at 4090 shape.
- `csrc/tans_codec.cpp` — CPU-side tANS encoder into the GPU slab layout.
- `csrc/tans_decode.cu` — standalone GPU tANS decoder, exercised by the
  round-trip test.
- `rans_vectoradd/tans_codec.py` — Python-side table construction and
  FP8 pair-stream preparation.
- `rans_vectoradd/data.py` — Qwen3-14B-FP8 exponent distribution used as
  the encoding source.
- `bench_vecadd.py` — local benchmark sweep.
- `profile_vecadd.py` — minimal Nsight Compute driver.

## Experimental branches

The result on this branch (`main`) is the clean 4090 path described in
the blog post. The companion branches hold work that didn't make the
cut for the post:

- `experiments` — GH200 / Hopper tuning of the same kernel family, with
  decode-table layout variants and bit-buffer-state-index alternatives.
- `hopper-bw-ceiling` — deeper Hopper exploration including alternative
  coders (Huffman, pair-Huffman, Tunstall, hybrid), warp-split kernel
  designs, and ~60 NCU-driven micro-variants. None of the variants
  crossed the amplification line on Hopper at FP8 vector-add shape;
  the branch is preserved as a record of what was tried.

These branches will lag main as a deliberate snapshot of the work
described in the *Outlook: bigger GPUs* section of the blog post.

## License

MIT, see [LICENSE](LICENSE).

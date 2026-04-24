# NOTES

## What this is

A fused kernel that decompresses rANS-encoded FP8 weights and does a
vector add in one pass. The goal is "bandwidth amplification" — a
kernel that processes compressed weights faster than raw HBM can
deliver uncompressed bytes, because the compressed input is smaller.

## Current state

On RTX 4090, the fused kernel's output throughput is 3–8% higher than
the same vector add run on uncompressed inputs. The uncompressed
reference already saturates HBM (~904 GB/s, ~91% of DRAM peak), so
this is genuine bandwidth amplification — the kernel produces
decompressed FP8 faster than the card can deliver raw FP8 for the
equivalent operation. `python bench_vecadd.py` reproduces.

## Open questions

### What's the deal with N & the current evidence of bandwidth amplification

Currently, we can get 1.08x bandwidth amplification with N=128. The theoretical
ceiling, if we treat the memcpy variant as the hardware limit, is like 1.06 or
something? Probably, we've optimized the DRAM accesses in the fused vecadd
kernel as well as leveraged the fact that there's less data to transfer. 

We should break this out, and also, get it working more efficiently at larger
$N$, where the ceiling is higher.

### Does this work on Blackwell?

4090 is memory-bound; B200 has 8x the HBM bandwidth but similar per-SM
decode compute. An unmodified port is expected to underperform memcpy
on B200 by a lot. Open question: what does it take to get bandwidth
amplification on B200?

### Does fusing into a real workload work?

The vecadd is a toy. The real claim is that we can decompress weights
inline with a consumer kernel (GEMM, attention) and beat the
uncompressed version end-to-end. Open question: can we fuse this into
a competitive FlashAttention kernel and beat uncompressed FA?

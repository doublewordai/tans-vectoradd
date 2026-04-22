# rANS FP8 GPU Decompressor — Engineering Plan

## Goal

Demonstrate **bandwidth amplification**: a GPU kernel that delivers decompressed
FP8 bytes faster than raw HBM can deliver uncompressed bytes. Compressed weights
are smaller → less HBM traffic → net throughput above memcpy, if the decoder is
fast enough.

**Targets:**
- **4090 (Ada, SM 89):** Cross the memcpy line. Deliver >862 Gfp8/s with a real
  compression ratio. This is "bandwidth amplification" — the decoder runs faster
  than uncompressed reads.
- **B200 (Blackwell, SM 100):** Achieve >50% of peak HBM bandwidth (~4 TB/s
  effective) via compressed decode. Memcpy baseline is ~6.9 TB/s at scale.

## Context

We have conclusive proof from a reliable source that bandwidth amplification is
achievable with this exact problem structure (rANS/ANS entropy decoding of FP8
exponent nibbles on GPU). This is not speculative — we are replicating a known
result. The challenge is pure engineering: finding the right combination of
algorithm variant, table layout, register management, and hardware-specific
tuning.

## Current state (as of 2026-04-22)

### Best standalone result: 4090

**pair-bl at N=128: 771 Gfp8/s = 0.89× memcpy (862 Gfp8/s)**

Optimizations stacked to get here:
1. `__ldg` for sfc table (L1 cache, no SMEM bank conflicts): +9%
2. Pair alphabet (256 symbols, M=4096, 2 nibbles/step): +41%
3. Branchless renorm (predicated, zero warp divergence): +4%

Compression ratio: 1.075× (7.44 bits/fp8). HBM-bound ceiling: ~927 Gfp8/s.
Gap to memcpy: 11%. Gap to ceiling: 17%.

Profile: compute 52%, memory 48%, neither saturated. IPC 1.99/4.0. Zero
divergent branches. Latency-limited — warps stall on ~30-cycle L1 reads for the
sfc table, with insufficient independent work to fill the gap.

### B200 result

786 Gfp8/s standalone — barely different from 4090 despite 8× HBM bandwidth.
Decoder is compute-bound: throughput scales with SM×clock, not HBM. At
production data sizes (4 GB), memcpy reaches ~6.9 TB/s, decoder stays at ~840
Gfp8/s = 0.12× memcpy.

### Dead ends (explored and measured)

| Approach | Result | Why it failed |
|----------|--------|---------------|
| Register-scan (regscan) | 246 Gfp8/s | 60 ALU instructions/step → compute-bound |
| Multi-stream (ms-4 to ms-16) | 577→194 Gfp8/s | Register pressure kills occupancy; fewer warps > more ILP |
| Split-phase multi-stream | 636→205 Gfp8/s | Compiler sees ILP but 4 LDGs cover only 4 of 30 latency cycles |
| Bufferless renorm | 502 Gfp8/s | More frequent global reads + more branches > register savings |
| Quad-stream (pair-q4) | 666 Gfp8/s at N=128 | Occupancy drop (100%→67%) outweighs ILP at small N |
| Factored decode (digit decomposition) | Incorrect | Joint cumulative doesn't decompose into independent base-M digits for non-uniform distributions. Theoretically impossible within standard rANS. |

### Key insight from dead ends

The 4090 is latency-bound with excellent occupancy (100% theoretical, 74%
achieved). The bottleneck is the serial dependency: each decode step waits ~30
cycles for an L1 table read before it can compute the next state. More streams
per thread trade occupancy for ILP, but the 4090 scheduler benefits more from
occupancy (48 warps hiding each other's stalls) than from per-warp ILP.

On B200, the same serial chain runs at the same absolute speed (~800 Gfp8/s)
while HBM is 8× faster. The gap is structural: decode throughput ∝ SM×clock,
HBM throughput ∝ memory technology. The solution must either radically reduce
per-step cost or increase parallelism without occupancy loss.

---

## Plan 1: Explicit joint triple table

**Hypothesis:** 3 nibbles per decode step (vs 2 for pairs) reduces L1 latency
per symbol from 15 to 10 cycles. ~33% throughput gain, potentially crossing the
memcpy line on 4090.

**Approach:** Build a 4096-symbol (16³) joint sfc table with M=8192. Each entry
is uint64 to fit sym(12) + f(13) + c(13). Table = 64 KB (half of L1). One
`__ldg` read per step, producing 3 exponent nibbles.

**Parameter space to explore:**

| Parameter | Values to try | Trade-off |
|-----------|--------------|-----------|
| M_total | 4096, 8192, 12288, 16384 | Quantization quality vs table size |
| Entry type | uint32 (M≤4096 only), uint64 | L1 bandwidth vs packing simplicity |
| N (stream length) | 126, 252, 504 (divisible by 6) | Compression ratio vs decode speed |
| Renorm style | Branchless, branched | Divergence vs instruction count |

**Risks:**
- Quantization overhead at M=8192 for 4096 symbols (~0.12 bits/nibble). May
  erode the compression ratio enough to offset the decode speedup.
- 64 KB table may cause L1 pressure, evicting compressed data / sign_mantissa
  from cache.
- uint64 reads use 2× the L1 bandwidth per lookup.

**Measurements needed:**
- Compression ratio at each M (actual, not estimated — the f≥1 floor constraint
  dominates for rare triples)
- Throughput (Gfp8/s) and profile (compute/memory/stall breakdown)
- L1 hit rate (does the larger table cause cache thrashing?)

**Success criteria:** >862 Gfp8/s on 4090 with a compression ratio >1.05×.

---

## Plan 2: Multi-stream with aggressive register minimization

**Hypothesis:** The multi-stream approach failed because register pressure
killed occupancy. If we can get 8+ streams per thread at ≤40 registers, the
split-phase pipeline fills the L1 latency gap without sacrificing warps.

**Approach:** Systematically reduce per-stream register usage through:

1. **Architectural changes to the decode step:**
   - Combine slab_idx and buf_avail into a single register (3-bit avail packed
     into slab_idx)
   - Use int32 for all offsets (block_base, tid_s)
   - Pack have flags into a bitmask
   - Explore minimal buffer sizes (1-entry pending vs 4-chunk buffer)

2. **Compiler-directed register management:**
   - `__launch_bounds__(128, N)` to force register ceiling
   - Measure spill cost vs occupancy gain empirically
   - Profile with `--ptxas-options=-v` at each configuration

3. **Manual instruction scheduling:**
   - Inline PTX for the critical decode step to control register allocation
   - Explicit `asm volatile` for the split-phase LDG issue
   - Ensure the compiler doesn't undo the split-phase structure

4. **Explore smaller decode step variants:**
   - Can we reduce the 12 ALU instructions per step?
   - Which instructions are on the critical path vs parallelizable?

**Parameter space:**

| Parameter | Values | Trade-off |
|-----------|--------|-----------|
| NS (streams/thread) | 4, 6, 8, 12 | ILP vs register pressure |
| Buffer size | 0, 1, 4 chunks | Registers vs global read frequency |
| launch_bounds min blocks | 8, 10, 12 | Forced register ceiling vs spill cost |
| Split-phase | yes/no | Compiler ILP vs code complexity |

**Target:** NS=8 at ≤40 registers (12 blocks, 48 warps, 100% occupancy on 4090).
8 LDGs in Phase A covers 8 of ~30 latency cycles. Combined with 48 warps of
multi-warp scheduling, this may be enough.

---

## Plan 3: tANS (table ANS) variant

**Hypothesis:** tANS replaces rANS's multiply+divide+renorm with a single
table lookup: `(next_state, symbol) = table[state][input_bits]`. Per-step ALU
drops from ~12 to ~3 instructions. This doesn't help on 4090 (latency-bound)
but could be transformative on B200 (compute-bound, 6.6× gap).

**Approach:**
- Implement a tANS encoder/decoder for the 16-symbol exponent alphabet
- State table: 2^R entries (R=10-12). Table = 4-16 KB.
- Each decode step: read table[state] → (symbol, nbits, next_state_base).
  Consume nbits from input. new_state = next_state_base + consumed_bits.
- Per-step cost: 1 L1 read + 3 ALU + bit extraction. No multiply, no divide.

**Why this matters for multi-stream:**
- tANS context = (state, bit_position) = 2 registers per stream (vs 7 for rANS)
- At 2 regs/stream: 16 streams = 32 regs for contexts. Leaves ~30 regs for
  temps. Easily fits in 12 blocks at 128 threads.
- 16 split-phase LDGs cover half the 30-cycle pipeline. Combined with multi-warp
  scheduling: potentially zero stalls.

**Risks:**
- tANS compression is slightly worse than rANS for the same table size
- The bit-extraction logic (variable-width reads from a bitstream) adds
  complexity and potential divergence
- tANS tables are distribution-specific (need to rebuild for each weight tensor,
  or per distribution class)

---

## Plan 4: B200-specific tuning

**Hypothesis:** Our B200 test was a zero-effort port (recompile for SM 100,
no tuning). B200 has meaningfully different hardware that we haven't exploited.

**B200 vs 4090 hardware differences:**

| Feature | 4090 | B200 | Implication |
|---------|------|------|-------------|
| SMs | 128 | 148 | 16% more compute |
| Max warps/SM | 48 | 64 | 33% more latency hiding |
| SMEM/SM | 100 KB | 228 KB | Room for prefetch tiles + tables |
| L2 cache | ~48 MB | 126 MB | Larger tables could live in L2 |
| Clock | 2.23 GHz | 1.97 GHz | 12% slower per-SM |
| HBM | 1 TB/s | ~8 TB/s | 8× more bandwidth to saturate |
| TMA | No | Yes | Hardware-accelerated async loads |

**Specific opportunities:**
- 64 warps/SM: multi-stream approaches that failed at 48 warps might work at 64
  (the occupancy-vs-ILP curve shifts)
- 228 KB SMEM: SMEM prefetch for compressed data (we have 0 SMEM usage currently)
  + sfc table in SMEM (no bank conflicts for the small triple/tANS tables)
- TMA: hardware-managed data movement for compressed stream → SMEM, freeing the
  thread for decode compute
- L2-resident tables: if tANS or large triple tables don't fit in L1, B200's
  126 MB L2 might serve them at acceptable latency (~100 cycles vs ~30 for L1)

**The 50% target:** 50% of ~6.9 TB/s = ~3.45 TB/s. Current decoder delivers
~0.84 TB/s. Need ~4× improvement. This likely requires combining multiple
optimizations (tANS + multi-stream + B200-specific features).

---

## Plan 5: Fused kernel (parallel track)

**Hypothesis:** The standalone decoder may never match memcpy on B200, but in
fusion with a consumer kernel (GEMV/GEMM), the decode overhead is hidden behind
the consumer's compute.

**Approach:**
- ThunderKittens framework: producer-consumer warpgroup pattern
- Producer warpgroup loads compressed weights, decodes into SMEM
- Consumer warpgroups read decompressed FP8 from SMEM, run MMA
- The producer's decode compute overlaps with the consumer's MMA compute

**When to pursue:** Once any standalone variant demonstrates decode throughput
within 2× of the target bandwidth. The fusion multiplier can cover a 2× gap but
not a 6× gap.

---

## Execution order

1. **Plan 1 (triple table)** — fastest to implement, direct extension of working
   pair code. Tests whether 3 nibbles/step crosses the line on 4090.

2. **Plan 3 (tANS)** — if triples don't cross the line, tANS's lower per-step
   cost is the next lever. Also enables Plan 2 (multi-stream) by reducing
   register pressure.

3. **Plan 2 (multi-stream)** — revisit with tANS contexts (2 regs/stream vs 7).
   The occupancy-ILP trade-off may flip with cheaper per-stream state.

4. **Plan 4 (B200 tuning)** — apply the winning 4090 approach to B200 with
   hardware-specific optimizations. The 50% target is achievable if the per-step
   cost is low enough (tANS) and parallelism is high enough (multi-stream + 64
   warps/SM).

5. **Plan 5 (fusion)** — once standalone throughput is within range, integrate
   into a ThunderKittens GEMV kernel for the end-to-end demonstration.

## Principles

- **Measure before theorizing.** Profile every variant. The ncu numbers have
  consistently surprised us (regscan was ALU-bound, multi-stream was
  occupancy-bound, B200 was compute-bound). Theory guides experiments;
  measurements settle them.
- **One variable at a time.** Each experiment changes one thing and measures the
  effect. Stack wins; don't guess at interactions.
- **Patience.** This is a hard engineering problem with a known solution. Every
  dead end narrows the search space. The wins have been incremental (9%, 41%,
  4%) and they compound.

"""tANS (table ANS / FSE) encoder and decode-table builder.

tANS replaces rANS's per-step multiply+divide+renorm with a single table
lookup: (symbol, nbits, next_state_base) = table[state]. The decoder reads
nbits from the bitstream and computes next_state = base + bits. ~3 ALU
instructions per step vs ~12 for rANS.

The trade-off: tANS needs a precomputed table per distribution (vs rANS
which works from raw frequencies). For weight decompression where the
distribution is fixed per tensor, this is fine.

Table size: L = 2^tableLog entries. Each entry is a uint32:
  bits  0..3  : symbol (0-15 for exponent nibbles)
  bits  4..7  : nbBits (0-tableLog, how many bits to consume)
  bits  8..23 : nextState base (add consumed bits to get next state)
"""
from __future__ import annotations

import numpy as np
import torch

from rans_vectoradd._C import rans_encode_interleaved as _cpu_encode_interleaved
from rans_vectoradd.codec import BLOCK_STREAMS, quantize_freqs


def build_tans_decode_table(
    freqs: np.ndarray,  # [n_symbols] int, summing to L = 2^tableLog
    table_log: int,
) -> np.ndarray:
    """Build tANS decode table. Returns uint32 array of L entries.

    Each entry packs: sym(4) | nbBits(4) | nextBase(16).
    State range during decode: [L, 2L). Table is indexed by (state - L).
    """
    L = 1 << table_log
    n_sym = len(freqs)
    assert freqs.sum() == L, f"freqs must sum to {L}, got {freqs.sum()}"
    assert all(f > 0 for f in freqs), "all freqs must be > 0"

    # Step 1: Spread symbols across L positions using coprime step.
    # The step must be odd (coprime to power-of-2 L).
    step = (L >> 1) + (L >> 3) + 3
    if step % 2 == 0:
        step += 1
    symbol_table = np.zeros(L, dtype=np.int32)
    position = 0
    for s in range(n_sym):
        for _ in range(int(freqs[s])):
            # Skip positions in high-prob area if needed (simplified: just step)
            symbol_table[position] = s
            position = (position + step) & (L - 1)

    # Step 2: Build decode entries.
    # For each table position x (0 to L-1):
    #   sym = symbol_table[x]
    #   substate = how many times we've seen this symbol so far
    #   ns_base = freq[sym] + substate  (in range [freq[sym], 2*freq[sym]))
    #   nbBits = tableLog - floor(log2(ns_base))
    #   nextState = (ns_base << nbBits) - L
    #
    # At decode time: state ∈ [L, 2L). Index = state - L.
    #   entry = table[index]
    #   sym, nbBits, nextBase = unpack(entry)
    #   bits = read_bits(nbBits)
    #   new_state = nextBase + bits   (guaranteed ∈ [L, 2L))

    sym_count = np.zeros(n_sym, dtype=np.int32)
    table = np.zeros(L, dtype=np.uint32)

    for x in range(L):
        s = symbol_table[x]
        substate = sym_count[s]
        sym_count[s] += 1

        ns_base = int(freqs[s]) + substate
        if ns_base == 0:
            # Shouldn't happen since all freqs > 0
            table[x] = 0
            continue

        high_bit = int(ns_base).bit_length() - 1  # floor(log2(ns_base))
        nb_bits = table_log - high_bit
        next_state = (ns_base << nb_bits) - L

        assert 0 <= next_state < L, f"nextState {next_state} out of range for x={x}"
        assert 0 <= nb_bits <= 15, f"nbBits {nb_bits} out of range"
        assert 0 <= s <= 15, f"symbol {s} out of range"

        entry = (s & 0xF) | ((nb_bits & 0xF) << 4) | ((next_state & 0xFFFF) << 8)
        table[x] = entry

    return table


def tans_encode_stream(
    symbols: np.ndarray,  # [N] uint8, symbol values 0-15
    freqs: np.ndarray,    # [n_sym] int, summing to L
    table_log: int,
) -> tuple[int, bytes]:
    """Encode a single stream of symbols using tANS. Returns (final_state, bitstream).

    Encoding processes symbols in REVERSE order (like rANS). Bits are
    accumulated into a bitstream. The decoder reads bits from the end
    backwards.
    """
    L = 1 << table_log
    n_sym = len(freqs)

    # Build cumulative frequencies
    cum = np.zeros(n_sym + 1, dtype=np.int64)
    for i in range(n_sym):
        cum[i + 1] = cum[i] + freqs[i]

    # Build encode table: for each (state, symbol) → (nbBits_to_output, new_state)
    # State range: [L, 2L).
    # For symbol s with freq f[s]:
    #   threshold = f[s] << (table_log + 1 - f[s].bit_length())
    #   While state >= threshold: output low bit, state >>= 1
    #   Then: state maps to the correct position in the spread table.
    #
    # Simpler approach: use the decode table in reverse.
    # Build a mapping: for each symbol s and substate k (0 to f[s]-1):
    #   encode_table[cum[s] + k] = state value from decode table

    # Build the symbol spread (same as decode table construction)
    step = (L >> 1) + (L >> 3) + 3
    if step % 2 == 0:
        step += 1
    symbol_table = np.zeros(L, dtype=np.int32)
    position = 0
    for s in range(n_sym):
        for _ in range(int(freqs[s])):
            symbol_table[position] = s
            position = (position + step) & (L - 1)

    # For encoding, we need: given (state, symbol) → (bits_to_output, new_state)
    # The tANS encode uses the state transition:
    #   1. While state >= (freq[s] << (tableLog + 1 - freq[s].bit_length())):
    #      output LSB, state >>= 1
    #   2. Find the correct table position for this symbol and substate

    # Build reverse lookup: for symbol s, the k-th occurrence in the
    # spread table is at position sorted_positions[s][k].
    sym_positions = [[] for _ in range(n_sym)]
    for x in range(L):
        sym_positions[symbol_table[x]].append(x)

    # Encode: process symbols in reverse
    state = L  # initial state
    bits_out = []  # list of (bit_value, ...) — we'll pack later

    for i in range(len(symbols) - 1, -1, -1):
        s = int(symbols[i])
        f = int(freqs[s])

        # Compute how many bits to output to bring state into [f, 2f).
        nb_out = 0
        temp = state
        while temp >= 2 * f:
            nb_out += 1
            temp >>= 1
        # Output nb_out low bits of state, MSB first (so LIFO reads LSB first).
        if nb_out > 0:
            bits_value = state & ((1 << nb_out) - 1)
            for b in range(nb_out - 1, -1, -1):
                bits_out.append((bits_value >> b) & 1)
            state >>= nb_out

        # Now state ∈ [f, 2f). The substate is state - f ∈ [0, f).
        substate = state - f

        # Map to table position: sym_positions[s][substate]
        # The decode table at this position will recover the symbol and
        # produce nextState = state (after reading the right bits).
        table_pos = sym_positions[s][substate]

        # New state = table_pos + L (decode state range is [L, 2L))
        state = table_pos + L

    # Pack bits into bytes (LSB first)
    bitstream = bytearray()
    for bit_idx in range(0, len(bits_out), 8):
        byte_val = 0
        for b in range(8):
            if bit_idx + b < len(bits_out):
                byte_val |= (bits_out[bit_idx + b] << b)
        bitstream.append(byte_val)

    return state, bytes(bitstream), len(bits_out)


def tans_decode_stream(
    final_state: int,
    bitstream: bytes,
    n_symbols: int,
    decode_table: np.ndarray,  # [L] uint32
    table_log: int,
    n_bits: int | None = None,  # actual number of data bits (excludes padding)
) -> np.ndarray:
    """Decode a stream using tANS. CPU reference implementation."""
    L = 1 << table_log

    # Read bits from the end of the bitstream backwards
    all_bits = []
    for byte_val in bitstream:
        for b in range(8):
            all_bits.append((byte_val >> b) & 1)
    # Start from the last DATA bit, not the last padded bit
    bit_pos = (n_bits - 1) if n_bits is not None else (len(all_bits) - 1)

    state = final_state
    symbols_out = []

    for _ in range(n_symbols):
        # Look up decode table
        idx = state - L
        entry = int(decode_table[idx])
        sym = entry & 0xF
        nb_bits = (entry >> 4) & 0xF
        next_base = (entry >> 8) & 0xFFFF

        symbols_out.append(sym)

        # Read nb_bits from bitstream (backwards)
        bits = 0
        for b in range(nb_bits):
            if bit_pos >= 0:
                bits |= (all_bits[bit_pos] << b)
                bit_pos -= 1

        state = next_base + L + bits  # next state in [L, 2L)

    return np.array(symbols_out, dtype=np.uint8)


def batch_encode_fp8_tans(
    fp8_bytes: torch.Tensor,   # [K, N] uint8
    exp_freqs: torch.Tensor,   # [16] int32 (single-nibble freqs)
    table_log: int = 11,       # L = 2^tableLog = 2048
    block_streams: int = BLOCK_STREAMS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, int]:
    """tANS encoder for FP8 exponent nibbles.

    Returns (compressed, final_states, block_offsets, sign_mantissa_packed,
    decode_table, table_log).
    """
    assert fp8_bytes.dim() == 2 and fp8_bytes.dtype == torch.uint8
    K, N = fp8_bytes.shape
    assert N % 2 == 0

    exp_nibbles = (fp8_bytes >> 3) & 0xF
    sm_nibbles  = ((fp8_bytes >> 4) & 0x8) | (fp8_bytes & 0x7)

    sm_pairs     = sm_nibbles.view(K, N // 2, 2)
    sm_packed_kn = sm_pairs[..., 0] | (sm_pairs[..., 1] << 4)
    sm_packed    = sm_packed_kn.t().contiguous()

    # Quantize to tANS table size
    L = 1 << table_log
    probs = exp_freqs.float()
    probs = probs / probs.sum()
    tans_freqs = quantize_freqs(probs.numpy(), M=L)

    # Build decode table
    decode_table_np = build_tans_decode_table(tans_freqs.numpy(), table_log)
    decode_table = torch.from_numpy(decode_table_np.astype(np.int32))

    # For the encoder, we reuse the rANS encoder with tANS frequencies.
    # The rANS encoder produces a valid compressed stream for the SAME
    # frequency distribution. The tANS decoder can't decode rANS data
    # directly, but we can build a tANS-specific encoder.
    #
    # For now, use a Python-based tANS encoder (slow but correct).
    # TODO: optimize with C++ if this approach proves viable.

    freqs_np = tans_freqs.numpy().astype(np.int64)
    all_compressed = []
    all_states = []

    for k in range(K):
        nibs = exp_nibbles[k].numpy().astype(np.uint8)
        state, bitstream = tans_encode_stream(nibs, freqs_np, table_log)
        all_states.append(state)
        all_compressed.append(bitstream)

    # Pack into interleaved slab format compatible with the GPU decoder.
    # For now, pack as a flat concatenation with per-stream offsets.
    # The GPU decoder will need a different compressed data layout than
    # the rANS slab format — tANS reads BITS not BYTES.
    #
    # Simple format: each stream's bitstream is byte-aligned, padded to
    # 4 bytes. Stored contiguously in encoder blocks of block_streams.
    # The GPU reads uint32s and extracts bits.

    n_blocks = (K + block_streams - 1) // block_streams

    # Pad each stream's bitstream to 4-byte alignment
    padded = []
    padded_lens = []
    for bs in all_compressed:
        pad = (4 - len(bs) % 4) % 4
        padded.append(bs + b'\x00' * pad)
        padded_lens.append(len(bs) + pad)

    # Compute block layout (same slab structure as rANS)
    block_offsets = [0]
    block_Gs = []
    for b in range(n_blocks):
        start = b * block_streams
        end = min(start + block_streams, K)
        max_len = max(padded_lens[start:end])
        if max_len == 0:
            max_len = 4
        G = max_len // 4
        if G & 1:
            G += 1  # even for 2-slab refills
        block_Gs.append(G)
        block_offsets.append(block_offsets[-1] + G * block_streams * 4)

    # Interleave into slabs
    total_bytes = block_offsets[-1]
    compressed = bytearray(total_bytes)

    for k in range(K):
        b = k // block_streams
        tid = k % block_streams
        G_b = block_Gs[b]
        stream_bytes = padded[k]
        G_s = len(stream_bytes) // 4

        base_offset = block_offsets[b]
        for g in range(G_s):
            slab_idx = G_b - G_s + g  # right-aligned
            dst = base_offset + slab_idx * block_streams * 4 + tid * 4
            compressed[dst:dst+4] = stream_bytes[g*4:(g+1)*4]

    compressed_t = torch.frombuffer(bytearray(compressed), dtype=torch.uint8).clone()
    states_t = torch.tensor(all_states, dtype=torch.int32)
    offsets_t = torch.tensor(block_offsets, dtype=torch.int32)

    return compressed_t, states_t, offsets_t, sm_packed, decode_table, table_log

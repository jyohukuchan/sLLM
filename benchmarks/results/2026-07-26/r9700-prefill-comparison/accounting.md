# Prefill logical traffic and FLOP accounting

## Purpose and boundary

This retains the policy of
`../sq8-r9700-handwritten-kernel-phase0-v0.1/efficiency.md`: it is a
KV-inclusive **logical streaming lower-bound**, not a claim that every counted
byte crossed HBM.  Hardware HBM/TCC byte counters were not captured, so
physical-HBM efficiency is **unconfirmed**.  The same normalized numerator is
used for uLLM SQ8_0, llama.cpp Q8_0/F32-KV, and llama.cpp Q8_0/F16-KV so that
the table's GB/s and percentage use one denominator.

The R9700 comparison roof is the inherited decimal `640,000,000,000 B/s`.

## Normalized common byte denominator

For prompt length `N`, let `K = ceil(N / 128)`.  The 128-token chunk matches
uLLM's fixed-M128 prefill mode and llama.cpp's `-ub 128` internal ubatch.

```text
B_common(N) = K * (B_projection_payload + B_projection_BF16_scales)
              + B_LM_head_BF16
              + B_causal_KV_read(N)
              + B_KV_write(N)

B_projection_payload     = 13,212,057,600 B
B_projection_BF16_scales =      1,612,800 B
B_LM_head_BF16           =  1,555,824,640 B

B_causal_KV_read(N)
 = 40 layers * 40 Q heads * (128 K + 128 V) * 4 B * sum(i, i=1..N)
 = 1,638,400 * N*(N + 1)/2 B

B_KV_write(N)
 = 40 layers * 8 KV heads * (128 K + 128 V) * 4 B * N
 = 327,680 * N B

logical_GB_per_s       = B_common(N) / elapsed_seconds / 1e9
logical_roof_ratio_640 = logical_GB_per_s / 640
```

The causal read expression exactly reproduces the prior 1024-token attention
accounting: `859,832,320,000 B`.  It expands GQA K/V across Q heads as logical
attention consumption, as the earlier policy does for decode's logical KV
read.  It is intentionally F32-equivalent even for llama.cpp F16 KV; this is
what makes the denominator identical across the F32 and F16 rows.  The table
must therefore not be read as a physical-byte bandwidth measurement.

For this reason `logical_roof_ratio_640` is **not a physical-HBM efficiency**.
It can exceed 1.0 when an implementation reuses or fuses GQA operands that
the common Q-head-expanded policy counts multiple times.  Such rows are
reported as a useful same-policy logical-work rate, while physical HBM
efficiency remains **unconfirmed** without TCC/HBM counters.  Calling a
greater-than-100% row an efficiency would be incorrect.

The weight terms are the existing SQ8_0 accounting applied uniformly: 280
projection payloads plus their BF16 `[128,128]` scales, once per 128-token
chunk, and one final BF16 LM-head evaluation.  The output-head term is one
per prompt because both implementations select only the final prompt output.
Embedding lookup, norms, activations, RoPE, softmax, page-table traffic,
workspace traffic, dequantization temporaries, cache reuse, copies, and launch
overhead are outside the lower bound.

This common denominator uses `ceil(N/128)` as the canonical M128 work shape.
The raw result additionally records each engine's observed execution-unit
count.  At N=4095, uLLM's fixed-M128 planner processes 31 full 128-token
chunks then falls back to 127 M=1 units because the remainder is less than
128, yielding 158 advances rather than the canonical 32.  llama.cpp records
32 internal 128-token ubatches.  This is a real execution-path difference,
not folded into the shared numerator; it is reported separately so that the
cross-engine work-normalized rate is not mistaken for physical traffic.

`summary.json` also records a format-aware alternative lower bound.  For
llama.cpp its projection and output-head byte fields come directly from the
GGUF tensor blocks (Q8_0 blocks include their scales) and its physical KV
storage is F32 or F16.  It is useful context but is deliberately not the
cross-engine efficiency numerator.

## Achieved FLOPS

The cross-engine FLOP numerator is likewise common and deliberately limited
to dense linear algebra plus QK/AV dot products:

```text
F(N) = N * 2 * 13,212,057,600
       + 2 * 777,912,320
       + 4 * 40 layers * 40 Q heads * 128 head_dim * N*(N + 1)/2

achieved_TFLOP_per_s = F(N) / elapsed_seconds / 1e12
```

The first term is the 280 projection matrices; the second is one final
LM-head matrix-vector; the last is causal QK plus AV (one multiply and one add
per operand).  It omits norms, RoPE, softmax, SiLU/gating, quantization,
reductions, and all non-matmul work.  It is an achieved-work indicator, not a
peak-FLOPS efficiency: a defensible common R9700 peak for these mixed
quantized kernels is not asserted here.

## Source quantities

| quantity | value | evidence |
| --- | ---: | --- |
| layers / Q heads / KV heads / head dimension | 40 / 40 / 8 / 128 | uLLM Qwen3-14B SQ8 runtime constants |
| SQ8 projection elements / payloads | 13,212,057,600 / 13,212,057,600 B | SQ8 artifact manifest (280 tensors) |
| SQ8 projection scales | 1,612,800 B | SQ8 artifact manifest |
| BF16 LM head | 777,912,320 elements / 1,555,824,640 B | SQ8 package manifest |
| GGUF Q8_0 block projection storage | 14,037,811,200 B | 280 rank-2 `blk.*` tensors via GGUF reader |
| GGUF Q8_0 output storage | 826,531,840 B | `output.weight` via GGUF reader |

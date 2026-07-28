# Gemma4 BF16 batched-matmul M=16 EE v0.1

## Result

Candidate B landed as a Gemma-only change: one 256-thread CTA now computes
sixteen F32 activation rows for one BF16 weight row. The first eight waves
retain the existing mapping and then compute rows 8--15 before the next LDS
weight strip. Each output dot keeps the prior F32 accumulation and wave
reduction order. No WMMA/MFMA, activation conversion, shared Qwen kernel, or
attention softmax code changed.

| context | promoted M=8 prefill | M=16 prefill | change | M=8 decode reference | M=16 decode |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 56.74 | 59.02 | 1.040x | 15.86 | 22.62 |
| 512 | 54.85 | 58.99 | 1.075x | 12.62 | 18.32 |
| 2048 | 52.79 | 54.04 | 1.024x | 12.08 | 15.53 |

The decode route is M=1 and does not dispatch this prefill matmul. The fresh
decode figures are non-regressing against the promoted reference; their
absolute uplift is attributed to a different measurement window, not to an
M=16 decode-path claim.

## Dispatch result

The new trace remains deliberately well-populated, not occupancy-starved:
the 12,288-row / M=128 MLP dispatch is `3,145,728 x 8` global threads with a
256-thread block: 98,304 CTAs, or **1,536 CTA/CU**. Across all matmuls the
smallest grid is 2,048 CTAs / **32 CTA/CU**. The extra accumulator pair raises
the dispatch record from 32 to 40 VGPR (still 512 B LDS, zero scratch).

For the representative 12,288 x 1,536 x 128 MLP projection, the M=16 trace
takes 3.017891 s over 640 launches, or **1.024 TFLOPS** for the direct
`2*M*N*K` FMA count. The promoted M=8 trace took 3.263121 s, or 947.0 GFLOPS;
this is a 1.081x kernel improvement. Summed Gemma matmul trace time fell from
8.167429 s to 7.601238 s (1.074x).

## GPU acceptance

All evidence is retained at
`benchmarks/results/2026-07-28/gemma4-bf16-m16-ee/raw/`.

- Real six-token activation differential vs unchanged host path: layer max
  abs `2.288818359375e-5`, final-norm max abs `3.0517578125e-5`, and logits
  max abs `1.621246337890625e-5`; resident and host logits top-1 are both
  `9079 / 22.510112762451172`.
- Two multi-step greedy cases match the BL token sequences and cached versus
  full-reprefill routes.
- Future-token and sliding `j-512` causal probes are bit-exact; N=2048 ring
  rollover matches its full M=N reprefill top-1.
- Shared-KV snapshot maps sliding consumers to source 13 and full consumers
  to source 14.
- The protected Qwen3.5 AQ4_0 output SHA-256 is
  `30865287e7525f4b24449ec24be3aa7619bfbbbbf48522cf2f67f9e58379b588`, with
  top-1 `220 / 8.529029846191406`; the frozen production binary was not
  changed.

The runtime translation-unit baseline was re-recorded in this same change:
`475204184566b0798883a931c1f1528b86dec79b6b1aeb8310a1637d2153f699`.


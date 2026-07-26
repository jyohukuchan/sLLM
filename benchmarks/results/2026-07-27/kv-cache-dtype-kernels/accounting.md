# Accounting boundary

The reported values are full-model engine-loop **tok/s**. They are not
profiler-range durations, kernel-only ratios, physical-HBM bandwidth, or an
estimate reconstructed from KV byte counts.

The timer boundary is intentionally aligned with the prior prefill comparison:
it starts after the same-length warm-up and model/request setup, includes the
actual AQ4_0 full-model prompt/decode work and final logit selection, and ends
after device synchronization. It excludes model load and reset.

The byte/FLOP formulas in
[`../../2026-07-26/r9700-prefill-comparison/accounting.md`](../../2026-07-26/r9700-prefill-comparison/accounting.md) are specific to Qwen3-14B SQ8_0 versus GGUF Q8_0. They
must not be copied numerically into this Qwen3.5-9B AQ4_0 dtype experiment:
the model has hybrid layers and a different projection inventory. No physical
HBM counter was captured, so physical-bandwidth efficiency is **unconfirmed**.

The exact KV allocation values are separately reported because they are a
storage result rather than a timing denominator. They include FP8's two
independent FP16 scale planes and page rounding:

| row | eight full-attention layers |
|---|---:|
| F32, context 4,096 | 256 MiB |
| F16, context 8,192 | 256 MiB |
| FP8 E4M3FN + FP16 scales, requested context 16,256 | 258 MiB (16,384 physical tokens) |

The raw `capacity-*.json` files provide the exact bytes per payload and scale
plane; no inferred admission limit is substituted for a successful model load.

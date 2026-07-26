# SQ8 artifact-FP32 reference feasibility — 2026-07-26

`feasibility.json` is a CPU-only preflight receipt, not a reference capture. It records that the
frozen v0.2 contract is bound to Qwen3-14B-FP8 while the requested/local 9B input is a different,
32-layer Qwen3.5 model with only a partial v0.1 SQ8 overlay. No GPU, service lifecycle,
activation, campaign, active manifest, `/opt/ullm`, or existing measurement result was touched.

The existing canonical decoder was independently run against the matching 14B artifact: all 280
weight/scale payload pairs were checksum-verified, and one `[128,128]` block was decoded to F32.
That confirms decoder reuse only; it is not a substitute for the unavailable 9B full-model timing.

Consequently this directory intentionally contains no logits, hidden states, token stream, or
reference capture. The 1-token and 8-step CPU timings, peak RSS, and 4,096-position extrapolation
are `not_run`/`not_computed`, rather than substituted with a source-model or projection-only proxy.

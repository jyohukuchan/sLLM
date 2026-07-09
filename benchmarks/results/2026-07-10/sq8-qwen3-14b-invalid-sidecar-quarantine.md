# Qwen3-14B-FP8 SQ8 Invalid Sidecar Quarantine

Date: 2026-07-10

## Decision

The affected uLLM rows remain connection diagnostics, but they are invalid for implementation,
quality, and performance conclusions. Their original `status=ok`, commands, logs, and metrics are
preserved. Each affected JSONL row carries a top-level `result_validity` quarantine marker.

vLLM rows are not quarantined by this decision.

## Reason

The Qwen3-14B-FP8 source checkpoint stores every FP8 projection weight with a 128x128 BF16
`weight_scale_inv` tensor. The v0.1 sidecar builder converted raw F8 values to F32 and requantized
them without applying the source scale. The resulting sidecar was therefore not mathematically the
source Qwen3-14B-FP8 model.

Reason code:

```text
source_fp8_weight_scale_inv_not_applied
```

## Affected Inventory

The row selector is:

```text
engine.name == uLLM
model.name == Qwen3-14B-FP8
workload.sq_artifact == /tmp/ullm-qwen3-14b-fp8-full-sq8-artifact
```

It matches 21 rows across these 8 JSONL files:

- `benchmarks/results/2026-07-09/sq8-vllm-fp8-comparison/results.jsonl`
- `benchmarks/results/2026-07-09/sq8-qwen3-14b-full-mixed-real-batch-smoke/results.jsonl`
- `benchmarks/results/2026-07-09/sq8-qwen3-14b-full-mixed-real-batch-d2d-pack-smoke/results.jsonl`
- `benchmarks/results/2026-07-09/sq8-qwen3-14b-full-mixed-real-batch-device-handoff-smoke/results.jsonl`
- `benchmarks/results/2026-07-09/sq8-qwen3-14b-full-mixed-real-batch-no-final-logits-smoke/results.jsonl`
- `benchmarks/results/2026-07-10/sq8-qwen3-14b-normalized-kernel-family-refresh/results.jsonl`
- `benchmarks/results/2026-07-10/sq8-qwen3-14b-post-host-pack-refresh/results.jsonl`
- `benchmarks/results/2026-07-10/sq8-qwen3-14b-no-host-staging-refresh/results.jsonl`

## Gate Behavior

- normalized throughput comparison always requires explicit
  `result_validity.performance_comparison_valid=true`;
- `--require-implementation-valid` requires explicit `implementation_valid=true` and a manifest
  SHA-256;
- quarantined rows are hidden from default summaries;
- `--include-quarantined` includes them for audit and shows their validity classification.

The prompt guard attached to some affected rows is retained as plumbing evidence only. It used the
same summary as reference and candidate and is not an independent quality oracle.

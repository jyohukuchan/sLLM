# SQ R9700 State Freeze v0.1

## Summary

This freezes the R9700-only SQ candidate evaluation state after the cached-prefix FlashAttention2-style prerequisite work.

- Target GPU: R9700/RDNA4, runtime device index `2`.
- Source model: `/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B`.
- AQ4 baseline: `aq4-lmhead-g8-weighted-lmhead-calib32-r9700-2026-07-07`.
- First SQ candidate: `sq-fp8-w8a16-r9700-v0`.
- Cached-prefix default executor: `cached_prefix_rdna4_fp8_auto`.

Machine-readable freeze:

- `benchmarks/results/2026-07-08/sq-r9700-state-freeze-v0.1.json`

## AQ4 Baseline Anchor

| field | value |
| --- | ---: |
| summary | `benchmarks/results/2026-07-07/engine/prompt-suite-aq4-lmhead-g8-weighted-lmhead-calib32-r9700/summary.json` |
| package | `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d` |
| prefill tok/s mean | 54.812211942 |
| decode tok/s mean | 62.240923434 |
| decode tok/s min | 51.249075594 |
| decode tok/s max | 64.468935027 |
| verified all | true |
| output ok / warn / not evaluated | 5 / 1 / 1 |

## SQ Candidate Contract

`sq-fp8-w8a16-r9700-v0` is the first SQ candidate. It is not the final SQ format.

| field | value |
| --- | --- |
| weight payload dtype | `fp8_e4m3` |
| activation dtype | `bf16_or_f32` |
| scale granularity | `row` |
| scale dtype | `f32` |
| initial target | language model 2D projection, embedding, and lm_head weights |
| deferred | visual tower, MTP, V620/RDNA2 dequant path, tensor parallel |

Required artifact metadata:

- `sq_manifest.json`
- FP8 tensor list
- passthrough tensor list and reason
- compact resident bytes
- materialized working-set bytes estimate
- scale dtype/layout
- source model path
- source AQ4 baseline package when applicable

## Measurement Contract

- Batch result rows use `inference-benchmark-result-v0.1`.
- SQ comparison rows use `sq-candidate-runtime-result-v0.1`.
- Batch workload manifests use `ullm-batch-throughput-workload-v0.1`.
- Result rows must keep `prefill_total_input_tps`, `decode_total_generated_tps`, and `end_to_end_total_tps` separate.
- Memory rows must include VRAM baseline/peak/consumed plus KV cache bytes when available.
- Cached-prefix rows using an auto executor must preserve both requested `executor` and concrete `resolved_executor`.

## Current T0-T2 Status

| task | status | note |
| --- | --- | --- |
| T0 state freeze | done | This file and the paired JSON are the frozen anchor. |
| T1 JSONL/schema preservation | done for v0.1 package batch rows | `run-external-benchmark.py` preserves total-throughput, prefix/chunk/context, executor, and KV cache accounting for `package-batch-throughput-bench-v0.1`; guarded by `phase-t1-jsonl-preservation-v1.md`. |
| T1 real batch runner | not done | Needed before SQ throughput comparison. |
| T2 artifact metadata path | partial done | `sq-fp8-w8a16-r9700-v0` manifest and writer are staged. |
| T2 runtime load path | partial done | `sq-fp8-materialize-smoke` validates the artifact boundary; `sq-fp8-token-ids-logits-smoke` validates one selected tensor overlay in the package path. |
| T2 short prompt guard | partial done with narrower boundary found | One `q_proj` overlay and layer 3 projection set passed top1 guards; layers `3,7` changed top1. Family split points to `q/v/down` as risky. Row-block scale recovers `q` and `down`, but not `v`. `v` fallback + `q/k/o/gate/up/down` row-block32 passes layers `3,7,11,15` on 3/3 short prompts and layers `3,7,11,15,19` on len4, but fails layers `3,7,11,15,19,23` and all self-attention probe layers. Full-target SQ guard is still pending. |

## Next Action

1. Keep `sq-fp8-materialize-smoke` as the runtime artifact-boundary guard.
2. Keep `v` fallback + `q/k/o/gate/up/down` row-block32 as the current partial-quality candidate and regression guard.
3. Treat the layers `3,7,11,15,19,23` failure as the next T2 boundary.
4. Define the T2 short-guard acceptance rule: strict top1 match, top-k overlap, or text-level tolerance.
5. If strict top1 remains required, test additional fallback families, per-layer fallback, or stronger scale/layout for the 6-layer cumulative drift.
6. Implement T1 real batch executor before using total throughput rows for SQ performance decisions.
7. Move to T5 throughput comparison only after the full-target guard satisfies the acceptance rule or the accepted quality tolerance is documented.

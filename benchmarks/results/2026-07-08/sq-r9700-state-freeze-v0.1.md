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
| T2 artifact metadata path | partial done with policy input | `sq-fp8-w8a16-r9700-v0` manifest and writer are staged. `tools/build-sq-fp8-w8a16-artifact.py` accepts `--policy-json`; dry-run confirmed `kup6_gate5_down5` selects `22` FP8 tensors and `753` passthrough tensors. |
| T2 runtime load path | partial done | `sq-fp8-materialize-smoke` validates the artifact boundary; `sq-fp8-token-ids-logits-smoke` validates one selected tensor overlay in the package path. |
| T2 short prompt guard | partial done with six-layer prompt bundle subset found | One `q_proj` overlay and layer 3 projection set passed top1 guards; layers `3,7` changed top1. Family split points to `q/v/down` as risky. Row-block scale recovers `q` and `down`, but not `v`. `v` fallback + `q/k/o/gate/up/down` row-block32 passes layers `3,7,11,15` on 3/3 short prompts and layers `3,7,11,15,19` on len4, but fails layers `3,7,11,15,19,23` and all self-attention probe layers. Six-layer split shows `k/up` row-block32 passes 3/3 short prompts, while `o/gate/down` fail individually at 6 layers. Per-layer combination search found `kup6_gate5_down5`, which passes len4/case_a/case_b strict top1. Case_a top8 overlap is only `2 / 8`, so this remains a regression subset, not full SQ policy. The selected FP8/fallback policy is saved as `sq-fp8-kup6-gate5-down5-policy-v0.1.json`. T2 promotion rule v0.1 is strict top1; full-target SQ guard is still pending. |

## Next Action

1. Keep `sq-fp8-materialize-smoke` as the runtime artifact-boundary guard.
2. Keep `kup6_gate5_down5` as the current six-layer strict-top1 regression subset.
3. Keep `kup6_ogatedown5` as a near-miss failure guard.
4. Use `sq-fp8-kup6-gate5-down5-policy-v0.1.json` through `--policy-json` as the current SQ policy representation for selected FP8 and fallback families.
5. Keep T2 promotion rule v0.1 as strict top1 until a text-level guard is implemented and accepted.
6. Use top-k overlap, AQ4 top1 rank, and logit gap as diagnostic-only fields.
7. Treat the current prefill/cached-prefix attention speed as sufficient to resume SQ candidate evaluation; defer extra FlashAttention2-like work unless SQ comparison exposes a blocker.
8. Implement T1 real batch executor before using total throughput rows for SQ performance decisions.
9. Use native FP8 or materialization-aware runtime paths for throughput comparison; do not use SQ overlay load timing as an SQ speed result.
10. Move to T5 AQ4/FP8 throughput comparison after the T2 prompt-bundle subset is encoded and the accepted quality boundary is documented.
11. Run vLLM comparison only after uLLM R9700 `batch=1/4/8` AQ4 and FP8 rows share the same schema.

# SQ R9700 State Freeze v0.1

## Summary

This freezes the R9700-only SQ candidate evaluation state after the cached-prefix FlashAttention2-style prerequisite work.

- Target GPU: R9700/RDNA4, runtime device index `2`.
- Source model: `/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B`.
- AQ4 baseline: `aq4-lmhead-g8-weighted-lmhead-calib32-r9700-2026-07-07`.
- First SQ candidate: `sq-fp8-w8a16-r9700-v0`.
- Current partial SQ policy artifact: `kup6_gate5_down5`, generated under `/tmp/ullm-sq-fp8-kup6-gate5-down5-policy-v0.1-artifact`.
- Current SQ format candidate matrix: `benchmarks/results/2026-07-08/sq-fp8-format-candidate-matrix-v0.1.json`.
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
| scale granularity | `row_or_row_block` |
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
| T1 component real-batch JSONL parser | done for prefill component rows | `run-external-benchmark.py --parse ullm-component-prefill` preserves `batching.mode=real`, `prefill_real_batch=true`, request/token parallelism, total input tok/s, attention pair/s, and sampled correctness for component prefill rows; guarded by `phase-t1-component-prefill-real-batch-parser-v1.md`. |
| T1 package-backed prefill component runner | done for v1 smoke and flattened batch grid | `tools/run-package-prefill-component-workload.py` runs package-backed component smokes and emits `inference-benchmark-result-v0.1` rows; R9700 AQ4 k-proj smoke verified `batching.mode=real` and `prefill_real_batch=true`; B=4 flattened token-parallel row preserves `batch_size=4`, `component_total_input_tokens=8`, and `prefill_executor_token_parallelism=8`; guarded by `phase-t1-package-prefill-component-runner-v1.md` and `phase-t1-package-prefill-component-batch-grid-v1.md`. |
| T1 model-loop hybrid throughput parser | partial done for selected-layer stack | `package-self-attn-mlp-block-model-loop-smoke` now emits timed prefill/decode/end-to-end token fields and CSV workload fields. `run-external-benchmark.py --parse ullm-model-loop-throughput` converts it into JSONL; R9700 layers `3,7` smoke preserved `batching.mode=hybrid`, `prefill_real_batch=false`, `decode_real_batch=true`, and `decode_executor_request_parallelism=2`; guarded by `phase-t1-model-loop-hybrid-throughput-smoke-v1.md`. |
| T1 token-id model-loop bridge | partial done for AQ4 selected-layer guard | `package-token-ids-model-loop-smoke` feeds embedding token IDs into the selected-layer model-loop scheduler and emits final lm_head top1 tokens. R9700 layers `3,7`, `batch=2`, `prompt=2`, `generated=1` preserved `input_source=embedding_token_ids`, `batching.mode=hybrid`, `decode_real_batch=true`, and final top1 tokens `155793,23175`; guarded by `phase-t1-token-id-model-loop-hybrid-smoke-v1.md`. |
| T1 token-id model-loop real-prefill bridge | partial done for AQ4 selected-layer request-batch prefill guard | `package-token-ids-model-loop-smoke` now groups prefill by layer/timestep across active requests and emits `prefill_batch_request_counts_csv`. R9700 layers `3,7`, `batch=2`, `prompt=2`, `generated=1` preserved `batching.mode=real`, `prefill_real_batch=true`, `prefill_executor=stack_prefill_request_batch_step`, `prefill_batch_request_counts=2,2,2,2`, `decode_real_batch=true`, and final top1 tokens `155793,23175`; guarded by `phase-t1-token-id-model-loop-real-prefill-smoke-v1.md`. |
| T1 full package total throughput runner | partial done with logical full-package grid | `package-batch-throughput-bench` full-package rows are saved for AQ4 `batch=1/4/8` in `phase-t1-full-package-logical-batch-small-grid-v1.md`. They preserve prefill/decode/end-to-end total throughput, KV cache bytes, VRAM, and correctness, but `batching.mode=logical`, `prefill_real_batch=false`, `decode_real_batch=false`, and weights are reloaded per request. Real request-batch prefill/decode/end-to-end rows are still needed before SQ throughput comparison. |
| T2 artifact metadata path | partial done with policy artifact verified | `sq-fp8-w8a16-r9700-v0` manifest and writer are staged. `tools/build-sq-fp8-w8a16-artifact.py` accepts `--policy-json`; actual `kup6_gate5_down5` payload artifact generated under `/tmp` with `22` FP8 tensors and `753` passthrough tensors. |
| T2 candidate matrix | done v0.1, quality/throughput pending | `tools/build-sq-fp8-candidate-matrix.py` builds `sq-fp8-format-candidate-matrix-v0.1.json` and `.md` from the current policy and artifact result. It fixes W8A16 F32-scale, scale16, scale8, W8A8, and hybrid fallback candidate axes. Strict top1 remains the promotion rule, and overlay load timing remains excluded from speed results. |
| T2 runtime load path | partial done with policy artifact materialize verified | `sq-fp8-materialize-smoke` validates the artifact boundary; `sq-fp8-token-ids-logits-smoke` validates selected tensor overlays in the package path. The current policy artifact materialized `model.language_model.layers.3.self_attn.k_proj.weight` on R9700 with `roundtrip_max_abs_diff=0` and `verified=true`. |
| T2 SQ token-id model-loop bridge | partial done for selected-layer real-prefill guard | `sq-fp8-token-ids-model-loop-smoke` connects the current `kup6_gate5_down5` SQ FP8 artifact to the token-id model-loop request-batch prefill path. R9700 layers `3,7`, `batch=2`, `prompt=2`, `generated=1` preserved `batching.mode=real`, `prefill_real_batch=true`, `decode_real_batch=true`, and AQ4-matching final top1 tokens `155793,23175`; guarded by `phase-t2-sq-fp8-token-id-model-loop-real-prefill-smoke-v1.md`. |
| T2 SQ token-id model-loop prompt bundle | partial done, strict-top1 not promoted | The same SQ FP8 artifact was run through the six-layer token-id model-loop prompt bundle on R9700 with layers `3,7,11,15,19,23`, batch `3`, and top-k `8`. Rows preserved `batching.mode=real`, `prefill_real_batch=true`, `decode_real_batch=true`, `workload.final_topk_tokens`, and `workload.final_topk_logits`. AQ4/SQ top1 matched only `case_b`; `len4` and `case_a` drifted while AQ4 top1 remained in SQ top8 at ranks `3` and `2`. Guarded by `phase-t2-sq-fp8-token-id-model-loop-prompt-bundle-v1.md`; `kup6_gate5_down5` remains a regression subset, not a promoted SQ policy. |
| T2 SQ model-loop coverage reduction | partial done, mixed scale/layout required next | k/up row-block32 coverage was reduced from six layers down to layer3 only on the same R9700 token-id model-loop prompt bundle. Every k/up row-block32 variant still failed `case_a`; `up_proj` layer3 row-block32 and `k_proj` layer3 row-block16 passed as single-tensor probes, but combined k/up layer3 row-block16 failed. Guarded by `phase-t2-sq-fp8-token-id-model-loop-coverage-reduction-v1.md`; next T2 step is per-family/per-tensor scale-layout support before testing mixed scale policies such as k row-block16 + up row-block32. |
| T2 SQ mixed scale layout | partial done, layer coverage interaction remains | `tools/build-sq-fp8-w8a16-artifact.py` now supports policy `scale.overrides[]` and emits candidate `scale_granularity=mixed`, `scale_layout=per_tensor`, with per-entry row-block widths. R9700 model-loop prompt bundle shows `kup1-layer3-k16-up32` passes `3 / 3`, but `kup2-k16-up32` fails `case_a` with AQ4 top1 at SQ top8 rank `2`. Guarded by `phase-t2-sq-fp8-token-id-model-loop-mixed-scale-v1.md`; next T2 step is isolating layer7 coverage/fallback. |
| T2 SQ layer7 isolation | partial done, layer7 up interaction isolated | R9700 model-loop prompt bundle shows `layer7-k16-up32` passes `3 / 3`, and `layer3-kup-plus-layer7-k16` also passes `3 / 3`. `layer3-kup-plus-layer7-up32` fails `case_a`, with AQ4 top1 at SQ top8 rank `3`. Guarded by `phase-t2-sq-fp8-token-id-model-loop-layer7-isolation-v1.md`; next T2 step is probing layer7 `up_proj` row-block16/64 or fallback while keeping layer3 k16/up32 and layer7 k16 fixed. |
| T2 SQ layer7 up scale | partial done, layer7 up fallback selected for current boundary | R9700 model-loop prompt bundle shows `layer7-up-fallback` passes `3 / 3`, while `layer7-up16` and `layer7-up64` both fail `case_a`, with AQ4 top1 at SQ top8 rank `2`. Guarded by `phase-t2-sq-fp8-token-id-model-loop-layer7-up-scale-v1.md`; current passing subset is layer3 `k_proj` row-block16, layer3 `up_proj` row-block32, and layer7 `k_proj` row-block16, with layer7 `up_proj` fallback. |
| T2 SQ layer7 add family | partial done, layer7 o/gate pass individually but not together | R9700 model-loop prompt bundle shows `layer7-plus-o32` and `layer7-plus-gate32` both pass `3 / 3`. `layer7-plus-down64` fails `case_a`, and `layer7-plus-o32-gate32` also fails `case_a` despite both families passing individually. Guarded by `phase-t2-sq-fp8-token-id-model-loop-layer7-add-family-v1.md`; current boundary keeps layer7 `up_proj` and `down_proj` fallback, and treats `o_proj`/`gate_proj` as branch candidates rather than a combined promoted subset. |
| T2 SQ layer7 o/gate scale | partial done, row-block16 does not recover o+gate | R9700 model-loop prompt bundle shows `o16+gate32`, `o32+gate16`, and `o16+gate16` all fail `case_a` with SQ top1 `193706` and AQ4 top1 at SQ top8 rank `2`. Guarded by `phase-t2-sq-fp8-token-id-model-loop-layer7-ogate-scale-v1.md`; current boundary keeps `o+gate` combined coverage as a failure guard and uses either `o32` or `gate32` as a single-branch extension. |
| T2 SQ layer7 o32 branch layer11 | partial done, layer11 k/o passes on o32 branch | R9700 model-loop prompt bundle shows `layer7-o32-plus-layer11-k16` and `layer7-o32-plus-layer11-k16-o32` both pass `3 / 3` strict top1 with AQ4 top1 `110784,237950,182949`. Guarded by `phase-t2-sq-fp8-token-id-model-loop-layer7-o32-branch-layer11-v1.md`; current passing branch is layer3 `k16/up32` + layer7 `k16/o32` + layer11 `k16/o32`, with layer7 `up_proj`/`down_proj` fallback. |
| T2 SQ layer7 o32 branch layer15 | partial done, layer15 k/o passes on o32 branch | R9700 model-loop prompt bundle shows `layer7-o32-layer11-o32-plus-layer15-k16` and `layer7-o32-layer11-o32-plus-layer15-k16-o32` both pass `3 / 3` strict top1 with AQ4 top1 `110784,237950,182949`. Guarded by `phase-t2-sq-fp8-token-id-model-loop-layer7-o32-branch-layer15-v1.md`; current passing branch is layer3 `k16/up32` + layer7 `k16/o32` + layer11 `k16/o32` + layer15 `k16/o32`, with layer7 `up_proj`/`down_proj` fallback. |
| T2 SQ layer7 o32 branch layer19 | partial done, layer19 k/o passes on o32 branch | R9700 model-loop prompt bundle shows `layer7-o32-layer11-o32-layer15-o32-plus-layer19-k16` and `layer7-o32-layer11-o32-layer15-o32-plus-layer19-k16-o32` both pass `3 / 3` strict top1 with AQ4 top1 `110784,237950,182949`. Guarded by `phase-t2-sq-fp8-token-id-model-loop-layer7-o32-branch-layer19-v1.md`; current passing branch is layer3 `k16/up32` + layer7 `k16/o32` + layer11 `k16/o32` + layer15 `k16/o32` + layer19 `k16/o32`, with layer7 `up_proj`/`down_proj` fallback. |
| T2 SQ layer7 o32 branch layer23 | partial done, selected-layer k/o branch passes through layer23 | R9700 model-loop prompt bundle shows `layer7-o32-layer11-o32-layer15-o32-layer19-o32-plus-layer23-k16` and `layer7-o32-layer11-o32-layer15-o32-layer19-o32-plus-layer23-k16-o32` both pass `3 / 3` strict top1 with AQ4 top1 `110784,237950,182949`. Guarded by `phase-t2-sq-fp8-token-id-model-loop-layer7-o32-branch-layer23-v1.md`; current passing branch is layer3 `k16/up32` + layer7/11/15/19/23 `k16/o32`, with layer7 `up_proj`/`down_proj` fallback. |
| T2 SQ selected-layer k/o layer3 o32 | partial done, layer3 o joins selected-layer k/o branch | R9700 model-loop prompt bundle shows `selected-layer-ko-plus-layer3-o32` passes `3 / 3` strict top1 with AQ4 top1 `110784,237950,182949`. Guarded by `phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer3-o32-v1.md`; current passing branch is layer3 `k16/o32/up32` + layer7/11/15/19/23 `k16/o32`, with layer7 `up_proj`/`down_proj`/`gate_proj` fallback. |
| T2 SQ selected-layer k/o layer3 gate scale | partial done, layer3 gate remains fallback | R9700 model-loop prompt bundle shows `selected-layer-ko-plus-layer3-o32-gate32` and `selected-layer-ko-plus-layer3-o32-gate16` both pass only `2 / 3` strict top1; len4 changes from AQ4 `110784` to SQ `102446`, with AQ4 top1 at SQ top8 rank `3`. Guarded by `phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer3-gate-scale-v1.md`; current passing branch remains `selected-layer-ko-plus-layer3-o32`, and layer3 `gate_proj` row-block32/16 are failure guards. |
| T2 SQ selected-layer k/o layer3 down64 | partial done, layer3 down joins current branch | R9700 model-loop prompt bundle shows `selected-layer-ko-plus-layer3-o32-down64` passes `3 / 3` strict top1 with AQ4 top1 `110784,237950,182949`. Guarded by `phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer3-down64-v1.md`; current passing branch is layer3 `k16/o32/up32/down64` + layer7/11/15/19/23 `k16/o32`, with layer3 `gate_proj` and layer7 `up_proj`/`down_proj`/`gate_proj` fallback. |
| T2 SQ selected-layer k/o layer11 up scale | partial done, layer11 up remains fallback | R9700 model-loop prompt bundle shows `selected-layer-ko-layer3-down64-plus-layer11-up32` and `selected-layer-ko-layer3-down64-plus-layer11-up16` both pass only `2 / 3` strict top1; len4 changes from AQ4 `110784` to SQ `102446`, with AQ4 top1 at SQ top8 rank `2`. Guarded by `phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer11-up-scale-v1.md`; current passing branch remains `selected-layer-ko-plus-layer3-o32-down64`, and layer11 `up_proj` row-block32/16 are failure guards. |
| T2 short prompt guard | partial done with six-layer prompt bundle subset found | One `q_proj` overlay and layer 3 projection set passed top1 guards; layers `3,7` changed top1. Family split points to `q/v/down` as risky. Row-block scale recovers `q` and `down`, but not `v`. `v` fallback + `q/k/o/gate/up/down` row-block32 passes layers `3,7,11,15` on 3/3 short prompts and layers `3,7,11,15,19` on len4, but fails layers `3,7,11,15,19,23` and all self-attention probe layers. Six-layer split shows `k/up` row-block32 passes 3/3 short prompts, while `o/gate/down` fail individually at 6 layers. Per-layer combination search found `kup6_gate5_down5`, which passes len4/case_a/case_b strict top1. Case_a top8 overlap is only `2 / 8`, so this remains a regression subset, not full SQ policy. The selected FP8/fallback policy is saved as `sq-fp8-kup6-gate5-down5-policy-v0.1.json`. T2 promotion rule v0.1 is strict top1; full-target SQ guard is still pending. |

## Next Action

1. Keep `sq-fp8-materialize-smoke` as the runtime artifact-boundary guard.
2. Keep `kup6_gate5_down5` only as a direct-logits regression subset; do not promote it as the SQ quality policy because the six-layer token-id model-loop prompt bundle fails strict top1 on `len4` and `case_a`.
3. Keep `kup6_ogatedown5` as a near-miss failure guard.
4. Use `sq-fp8-kup6-gate5-down5-policy-v0.1.json` through `--policy-json` as the current SQ policy representation for selected FP8 and fallback families; the generated `/tmp` artifact is a reproducible boundary-check artifact, not a committed package.
5. Use `sq-fp8-format-candidate-matrix-v0.1.json` as the current machine-readable SQ format candidate matrix.
6. Keep T2 promotion rule v0.1 as strict top1 until a text-level guard is implemented and accepted.
7. Use top-k overlap, AQ4 top1 rank, logit gap, and saved `workload.final_topk_tokens` / `workload.final_topk_logits` as diagnostic-only fields.
8. Treat the current prefill/cached-prefix attention speed as sufficient to resume SQ candidate evaluation; defer extra FlashAttention2-like work unless SQ comparison exposes a blocker.
9. Use synthetic and package-backed component real-batch rows only for kernel/schema validation, not final SQ package throughput.
10. Extend T1 from logical full-package rows to real full-package request-batch prefill/decode/end-to-end rows before using total throughput rows for SQ performance decisions.
11. Use native FP8 or materialization-aware runtime paths for throughput comparison; do not use SQ overlay load timing as an SQ speed result.
12. Keep `layer7-k16-up32` and `layer3-kup-plus-layer7-k16` as passing probes, and `layer3-kup-plus-layer7-up32` as the isolation failure guard.
13. Keep `layer7-up-fallback` as the current passing boundary; keep `layer7-up16` and `layer7-up64` as failure guards.
14. Keep `layer7-plus-o32` and `layer7-plus-gate32` as individual passing branches; keep `layer7-plus-down64` and `layer7-plus-o32-gate32` as failure guards.
15. Keep `layer7-ogate-o16-gate32`, `layer7-ogate-o32-gate16`, and `layer7-ogate-o16-gate16` as evidence that row-block16 does not recover combined o/gate coverage.
16. Keep `layer7-o32-plus-layer11-k16-o32` as the current passing branch and `layer7-o32-plus-layer11-k16` as its rollback guard.
17. Keep `layer7-o32-layer11-o32-plus-layer15-k16-o32` as the current passing branch and `layer7-o32-layer11-o32-plus-layer15-k16` as its rollback guard.
18. Keep `layer7-o32-layer11-o32-layer15-o32-plus-layer19-k16-o32` as the current passing branch and `layer7-o32-layer11-o32-layer15-o32-plus-layer19-k16` as its rollback guard.
19. Keep `layer7-o32-layer11-o32-layer15-o32-layer19-o32-plus-layer23-k16-o32` as the current passing branch and `layer7-o32-layer11-o32-layer15-o32-layer19-o32-plus-layer23-k16` as its rollback guard.
20. Keep `selected-layer-ko-plus-layer3-o32` as the current passing branch.
21. Keep layer3 `gate_proj` row-block32 and row-block16 as failure guards.
22. Keep `selected-layer-ko-plus-layer3-o32-down64` as the current passing branch.
23. Keep layer11 `up_proj` row-block32 and row-block16 as failure guards.
24. Probe layer11 `down_proj` row-block64 on top of the current branch.
25. Move to T5 AQ4/FP8 throughput comparison after the T1 full package total-throughput runner exists and a T2 model-loop-safe subset is encoded in the runtime path.
26. Run vLLM comparison only after uLLM R9700 `batch=1/4/8` AQ4 and FP8 rows share the same schema.

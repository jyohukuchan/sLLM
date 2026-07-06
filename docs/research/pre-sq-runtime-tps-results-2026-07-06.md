# Pre-SQ Runtime TPS Results 2026-07-06

## 前回の要点

- `docs/plans/pre-sq-runtime-tps-plan-v0.1.md` は、sq format策定前に実推論に近いtoken/sを取ることを目的にしている。
- 数tokenのsmokeだけではなく、最低 `prompt_tokens=512`、`generated_tokens=256` をR9700/RDNA4とV620/RDNA2で測る必要がある。
- tensor parallel、batch処理、server APIは今回の範囲外にしている。

## 今回の変更点

- `ullm-engine package-token-ids-bench` を追加し、既存のhybrid incremental generate経路を正式なpre-sq計測入口として使えるようにした。
- incremental self-attentionのKV cache bytesをJSONへ記録するようにした。
- `tools/summarize-runtime-tps.py` を追加し、raw smoke JSONからMarkdown summaryと `inference-benchmark-result-v0.1` 風JSONLを生成できるようにした。
- `tools/run-external-benchmark.py` に `--parse ullm-token-ids-generate` を追加し、uLLM stdout JSON、rocm-smi VRAM監視、correctness summaryを同じJSONL行にまとめられるようにした。
- R9700とV620で `prompt_tokens=512`, `generated_tokens=256` のVRAM監視付きrunを完走した。
- materialized-AQ baseline packageでR9700 `512/256` を完走した。V620側の同一長decodeは、R9700/V620ともdecode約 `0.14 tok/s` に張り付くことが既に確認できたため、途中で意図的に停止した。
- BF16 baseline feasibilityを確認し、現行package/runtimeだけでは真のBF16 baselineを作れないと判断した。
- R9700/V620で短いgolden prefix reference guardを実行し、accepted packageが12層のfixture比較でverifiedになることを確認した。

## 次の行動

1. T6 decision packを作り、sq format策定へ入るための判断材料をまとめる。
2. 以後のTPS測定は、長いprefillと短いdecodeを分ける。decodeが約 `0.14 tok/s` の経路で長時間測定を繰り返さない。
3. sq format案では、F32常駐を避ける保存形式とdecode時のmaterialize範囲を最優先で検討する。

## Artifacts

- Raw runtime smoke summary: `benchmarks/results/2026-07-06/engine/pre-sq-runtime-bench-summary.md`
- Raw runtime smoke JSONL: `benchmarks/results/2026-07-06/engine/pre-sq-runtime-bench-summary.jsonl`
- VRAM-monitored benchmark JSONL: `benchmarks/results/2026-07-06/engine/pre-sq-runtime-bench-vram.jsonl`
- VRAM-monitored benchmark summary: `benchmarks/results/2026-07-06/engine/pre-sq-runtime-bench-vram-summary.md`
- Materialized-AQ baseline JSONL: `benchmarks/results/2026-07-06/engine/pre-sq-runtime-baseline-vram.jsonl`
- Materialized-AQ baseline summary: `benchmarks/results/2026-07-06/engine/pre-sq-runtime-baseline-vram-summary.md`
- BF16 baseline feasibility note: `docs/research/pre-sq-bf16-baseline-feasibility-2026-07-06.md`
- T4 reference guard summary: `benchmarks/results/2026-07-06/engine/pre-sq-runtime-t4-reference-guard-summary.md`
- T4 R9700 reference guard JSONL: `benchmarks/results/2026-07-06/engine/package-golden-prefix-t4-r9700-actual-prefix0-12-accepted-qwen35-hidden3994-v1.jsonl`
- T4 V620 reference guard JSONL: `benchmarks/results/2026-07-06/engine/package-golden-prefix-t4-v620-actual-prefix0-12-accepted-qwen35-hidden3994-v1.jsonl`
- SQ design input memo: `docs/plans/sq-format-design-input-v0.1.md`

Local raw logs are under `benchmarks/results/2026-07-06/engine/logs/`, but that directory is intentionally ignored by git. The tracked JSONL above contains the comparable metrics, memory summary, correctness summary, and artifact paths.

## Primary Result

Package:

```text
/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-qwen35-hidden3994-v1-row-scale-layer6-layer10.ullm.d
```

Conditions:

- model: Qwen3.5-9B package
- quantization policy: `qwen35_9b_p4p46_hidden3994_v1`
- layers: all decoder layers
- decode mode: `hybrid_incremental_greedy`
- prompt tokens: `512`
- generated tokens: `256`
- batch size: `1`
- tensor parallel: `1`
- sampling: greedy
- KV cache dtype in current runtime: f32

| target | uLLM device | rocm-smi card | prefill tok/s | decode tok/s | total tok/s | total wall s | TTFT ms | TPOT ms | decode p50 ms | decode p95 ms | consumed GiB | peak total GiB | KV bytes | verified |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| R9700/RDNA4 | `2` | `card2` | 2.912 | 0.141 | 0.387 | 1984.922 | 175802.661 | 7084.688 | 7037.410 | 7625.910 | 26.257 | 26.419 | 50331648 | true |
| V620/RDNA2 | `1` | `card1` | 2.520 | 0.139 | 0.377 | 2037.830 | 203177.131 | 7182.688 | 7151.547 | 7709.009 | 26.247 | 26.409 | 50331648 | true |

V620 note: the engine `device_index=1` run mapped to rocm-smi `card1` in the memory log. The initial `card0` hint in the command was corrected in the saved JSONL metadata.

## Interpretation

- The required `512/256` grid is now proven on both R9700 and V620 with the same JSONL schema.
- Decode throughput is effectively the same on R9700 and V620, around `0.14 tok/s`. This means current runtime overhead dominates GPU generation differences.
- R9700 is faster on prefill, about `2.912 tok/s` vs `2.520 tok/s`, but the end-to-end run is still decode dominated.
- KV cache is only about `48 MiB`. The measured VRAM pressure, about `26.25 GiB` consumed, is dominated by resident f32 materialized weights and runtime buffers rather than KV.
- These numbers are not a product-speed target. They are a pre-sq lower-bound measurement from the current proof path.

## SQ Design Implications

- The first sq format should prioritize avoiding whole-layer f32 residency.
- Weight storage and decode-time materialization granularity matter more than KV compression for this specific `512/256` single-request case.
- The current decode path should not be used to judge final RDNA2 vs RDNA4 hardware potential because lm_head/top-k and per-step host/runtime orchestration are still heavy.
- Because decode is already pathologically slow and stable, repeating long `256` token decode runs on the same current path has low value. Future measurement should split long prefill pressure from short decode probes until runtime or sq implementation changes.
- A useful sq prototype should make the following visible in the benchmark record:
  - compact resident bytes
  - materialized working-set bytes
  - prefill TPS
  - decode TPS
  - materialization time
  - steady-state decode time

## Materialized-AQ Baseline Update

Baseline package:

```text
/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer10.ullm.d
```

R9700 `512/256` completed with the same benchmark schema:

| target | uLLM device | rocm-smi card | prefill tok/s | decode tok/s | total wall s | consumed GiB | peak total GiB | KV bytes | verified |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| R9700/RDNA4 | `2` | `card2` | 2.912 | 0.140 | 1998.935 | 26.257 | 26.424 | 50331648 | true |

V620 `512/256` was intentionally stopped after the R9700 result showed the same decode bottleneck already seen in the accepted package runs. The partial V620 memory log has 294 samples and reached `card1` used `26.268 GiB`, total used `26.411 GiB`. This is enough to confirm that the baseline reaches the same resident-memory regime, but it is not a formal V620 TPS record.

Conclusion:

- The materialized-AQ baseline is not meaningfully faster than the accepted package on the current runtime path.
- The bottleneck to address before sq format measurement is runtime decode cost and f32 materialized residency, not RDNA2 vs RDNA4 selection.
- Long decode runs should resume only after the execution path changes or when a publication-quality sustained number is specifically needed.

## BF16 Baseline Feasibility

The existing full Qwen3.5 package artifacts are not BF16/passthrough-only runtime baselines. The checked materialized-AQ baseline package contains `255` quantized tensors and `520` passthrough tensors. Its large decoder matrix families are under quantized `tensors/`, while passthrough is mostly `embed`, `lm_head`, and `other` small/support tensors.

The loader can read BF16 passthrough payloads, but `read_named_passthrough_f32*` expands them to f32 host values. The decoder matrix path uses `materialize_selected_aq4_matrix`, which dequantizes AQ4 tensors into f32 runtime buffers. Therefore, running the current package is not a true BF16 compute baseline.

For this pre-sq stage, the true BF16 baseline is deferred rather than implemented. Implementing it would require at least a passthrough-only full decoder package, loader branches that select passthrough decoder matrices, and runtime kernels or buffer paths that preserve the intended BF16 baseline semantics.

## T4 Reference Guard

Fixture:

```text
benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16
```

Package:

```text
/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-qwen35-hidden3994-v1-row-scale-layer6-layer10.ullm.d
```

Condition:

- command: `package-golden-prefix-smoke`
- run mode: `actual_prefix`
- layers: `0..12`
- sequence length: `16`
- rotary dim: `64`
- rope base: `10000000`

| target | uLLM device | backend | layers | max MSE | max mean abs diff | max abs diff | min cosine similarity | verified |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | :---: |
| R9700/RDNA4 | `2` | hip | 12 | 0.003055947561 | 0.043167665239 | 0.629638672 | 0.994777962 | true |
| V620/RDNA2 | `1` | hip | 12 | 0.003055947561 | 0.043167665239 | 0.629638672 | 0.994777962 | true |

This is not a full logits or generated-token reference check. It is a short hidden-state fixture guard proving that the accepted package still matches the existing golden prefix fixture within the known AQ error envelope on both RDNA4 and RDNA2.

## T6 Decision Pack

Decision:

- Proceed to sq format design input with the current measurements.
- Do not spend more time on long `256` token decode runs for the current f32-materialized path.
- Treat current TPS as lower-bound proof-path numbers, not as a product-speed target.

Performance and memory facts:

| item | R9700/RDNA4 | V620/RDNA2 | note |
| --- | ---: | ---: | --- |
| accepted package prefill tok/s | 2.912 | 2.520 | `512` prompt tokens |
| accepted package decode tok/s | 0.141 | 0.139 | `256` generated tokens |
| accepted package consumed GiB | 26.257 | 26.247 | VRAM consumed over baseline |
| accepted package KV bytes | 50331648 | 50331648 | about 48 MiB |
| materialized-AQ baseline decode tok/s | 0.140 | deferred | R9700 long-run anchor only |

Bottleneck classification:

- The run is decode dominated. TPOT is about `7.1 s/token`, so total wall time is dominated by generated token count.
- R9700 prefill is faster than V620 prefill, but decode is effectively identical.
- KV cache is not the memory bottleneck for this workload. Resident f32 materialized weights and runtime buffers dominate VRAM.
- The current runtime path is too slow to use repeated long decode runs as an optimization signal.

Correctness guard:

- R9700 and V620 both passed the short golden prefix fixture guard for layers `0..12`.
- This is enough to avoid measuring an obviously broken path for pre-sq TPS records.
- It is not a substitute for final logits or generated-token agreement in a later product-quality benchmark.

Baseline decision:

- materialized-AQ f32 residency is the current lower-bound baseline.
- True BF16 baseline is deferred because current package artifacts and runtime do not support full decoder BF16 baseline semantics.
- BF16 should be revisited after either a passthrough-only full decoder package exists or the runtime has a clean BF16 matrix path.

SQ design implications:

- The first sq format should avoid whole-model f32 residency.
- The format and loader should make resident compact bytes, materialized working-set bytes, and materialization time visible in benchmark records.
- The first sq candidate should preserve the accepted correctness policy as a reference point, including row-scale override capability.
- Performance probes should split long prefill pressure from short decode probes until the decode path is no longer pathologically slow.

## Remaining Plan Items

- T3 is now substantially satisfied for the minimum `512/256` grid, including VRAM.
- T4 is satisfied for this pre-sq scope by the short golden prefix reference guard on R9700 and V620.
- T5 is closed for the current pre-sq scope: materialized-AQ has an R9700 long-run anchor, and true BF16 baseline is explicitly deferred because current artifacts/runtime do not support it.
- T6 is satisfied by the decision pack above and `docs/plans/sq-format-design-input-v0.1.md`.
- Stretch context runs should be deferred unless a faster path is introduced.

## Runtime Decode Bottleneck Fix Update

Follow-up debugging showed that the earlier `~0.14 tok/s` decode result did not represent the
raw FP32/AQ GPU path. It was dominated by the CPU/chunked lm_head top-k path and by smoke-only
self-attention verification work inside the incremental decode loop.

Implemented fixes:

- Added `gpu_resident_f32` lm_head mode for `package-token-ids-generate-smoke` and
  `package-token-ids-bench`, so lm_head weights are loaded to GPU once instead of scanned from
  package chunks on CPU every token.
- Added prefill/decode timing breakdowns, including per-layer step timings and lm_head timings.
- Moved linear-attention gate/beta parameters to resident GPU buffers.
- Removed the expensive per-value-head GPU RMSNorm loop from the linear-attention decode step by
  computing that small headwise norm on the host after recurrent output readback.
- Made pure HIP runtime kernels enqueue asynchronously; host synchronization is now left to
  explicit readback/synchronization points.
- Added opt-in rocBLAS SGEMV for `matvec_f32`, with fallback to the existing HIP kernel.
- Added a lighter self-attention incremental prepare path that skips smoke-only q/k/RoPE/attention
  reference recomputation.

Short R9700 validation:

| target | device | prompt | generated | lm_head mode | decode tok/s | prefill tok/s | decode p50 ms | layers p50 ms | lm_head p50 ms | verified |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | :---: |
| R9700/RDNA4 | `2` | `16` | `8` | `gpu_resident_f32` | 5.254 | 4.712 | 191.185 | 182.609 | 7.175 | true |

Short V620 compatibility check:

| target | device | prompt | generated | lm_head mode | decode tok/s | prefill tok/s | decode wall ms | layers wall ms | lm_head wall ms | verified |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | :---: |
| V620/RDNA2 | `1` | `1` | `2` | `gpu_resident_f32` | 3.725 | 1.204 | 268.451 | 257.248 | 10.088 | true |

Updated interpretation:

- The current R9700 FP32 path now exceeds the minimum debug target of `5 tok/s` on the short
  `prompt=16/generated=8` probe.
- This is still not near the expected R9700 product-speed range. The remaining main cost is the
  per-layer decode body, especially repeated GEMV and host/runtime orchestration.
- AQ vs SQ should not be judged from the old `0.14 tok/s` number. That number was a runtime path
  artifact, not a useful memory-bandwidth-bound AQ result.

New artifacts:

- `benchmarks/results/2026-07-06/engine/package-token-ids-generate-gpu-lm-head-r9700-prompt16-gen8-self-prepare-fast.json`
- `benchmarks/results/2026-07-06/engine/package-token-ids-generate-gpu-lm-head-v620-prompt1-gen2-self-prepare-fast.json`

## Warmup-Aware Decode Follow-Up

User feedback noted that GPU warmup can make token/s look artificially low. The benchmark JSON now
includes `decode.step_wall_summary`, which reports all-step TPS plus `warmup_skip_1`,
`warmup_skip_2`, `last_4`, and `last_8` step TPS.

R9700 `prompt=16/generated=16` validation with the default HIP matvec path:

| target | device | prompt | generated | lm_head mode | all-step tok/s | skip-1 tok/s | skip-2 tok/s | last-4 tok/s | p50 step ms | verified |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | :---: |
| R9700/RDNA4 | `2` | `16` | `16` | `gpu_resident_f32` | 5.220 | 5.210 | 5.199 | 5.136 | 191.523 | true |

Interpretation:

- The first timed decode steps were the fastest measured steps (`186.422 ms`, then `186.343 ms`),
  so this result is not being suppressed by first-token warmup.
- Decode step time slowly increases from about `186 ms` to `195 ms`; this is consistent with
  cache/position-dependent work, especially self-attention, rather than GPU warmup.
- The average layer body cost is about `183.3 ms/token`. Linear-attention layers alone account for
  about `154.0 ms/token`, which already exceeds the `50 ms/token` budget required for `20 tok/s`.
- rocBLAS SGEMV was measured slightly slower for this 1-token decode path, so `matvec_f32` now uses
  the existing HIP kernel by default. rocBLAS remains available through
  `ULLM_ENABLE_ROCBLAS_MATVEC=1` or `ULLM_REQUIRE_ROCBLAS_MATVEC=1`.

New artifact:

- `benchmarks/results/2026-07-06/engine/package-token-ids-generate-gpu-lm-head-r9700-prompt16-gen16-warmup-summary-hip-matvec.json`

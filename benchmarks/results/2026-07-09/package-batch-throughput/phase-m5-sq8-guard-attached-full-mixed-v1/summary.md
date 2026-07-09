# M5 SQ8_0 Guard-Attached Full Mixed Row

## 前回の要点

- M5 requires full mixed SQ8_0 result rows to preserve implementation metadata, artifact timing,
  throughput, and prompt-suite regression status.
- `run-external-benchmark.py` can now attach a prompt-suite guard bundle through
  `--prompt-guard-bundle-json`.
- The behavioral guard bundle for `sq-layer23-k16` already existed at
  `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-prompt-suite-text-guard-smoke-v0.1/guard-behavioral/guard-bundle-summary.json`.

## 今回の変更点

- Re-ran the full mixed request-state SQ8_0 prompt bundle with the behavioral guard bundle attached.
- Saved the `inference-benchmark-result-v0.1` row to `results.schema.jsonl`.
- Verified the row carries the M5 minimum fields: `SQ8_0` format ID, implementation ID, SQ artifact,
  FP8/passthrough tensor counts, direct execution mode, projection counters, artifact timing,
  total throughput, and prompt-suite regression status.

## Result

| Status | Engine | Model | Family | Quant | SQ mode | Impl | Target | Workload | Batching | Prefill total tok/s | Decode total tok/s | End-to-end tok/s | Consumed GiB | Decode x GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ok | uLLM | Qwen3.5-9B | FP8 | SQ8_0 | direct_fp8_dequant_matvec | single=sq8_0_matvec_rdna4_direct,triple=sq8_0_matvec_triple_rdna4_direct | R9700 | pp20/tg3/b3 | real | 58.30 | 73.10 | 30.70 | 4.29 | 313.51 |

Key fields:

- `workload.sq_execution_mode`: `direct_fp8_dequant_matvec`
- `workload.sq_projection_boundary`: `single+triple`
- `workload.sq_fp8_single_matvec_count`: `23`
- `workload.sq_fp8_triple_matvec_count`: `115`
- `metrics.artifact_load_wall_time_seconds`: `13.056968994`
- `metrics.load_excluded_total_wall_time_seconds`: `0.749147515`
- `metrics.load_included_total_wall_time_seconds`: `14.085140174`
- `quality.prompt_suite_regression_status`: `passed`
- `guards.prompt_guard_bundle.acceptance_mode`: `behavioral`
- `guards.prompt_guard_bundle.behavioral_passed`: `true`
- `guards.prompt_guard_bundle.strict_passed`: `false`

`metrics.artifact_materialization_wall_time_seconds` is `null` because this direct resident row did
not report a separate materialization timer. The field is intentionally not aliased from layer load.

## 次の行動

- Use this row as the first M5 guard-attached uLLM SQ8_0 row.
- For M10, create matching prompt/generation/concurrency rows before comparing against vLLM + FP8.

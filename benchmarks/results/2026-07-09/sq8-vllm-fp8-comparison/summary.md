# SQ8 vLLM FP8 Comparison Prep

## 前回の要点

- M10 compares uLLM SQ8_0 rows with an external `vLLM + FP8` baseline after uLLM rows become
  implementation-valid.
- The vLLM smoke template uses `prompt_tokens=16`, `generated_tokens=8`, and
  `concurrent_requests=1`.
- The current uLLM SQ8_0 package path is Qwen3.5-9B, while the planned vLLM external baseline is
  Qwen3-14B-FP8.

## 今回の変更点

- Added a uLLM SQ8_0 smoke-shape row with `pp16/tg8/b1`.
- Attached the existing behavioral prompt-suite guard bundle.
- Preserved direct SQ8_0 projection metadata and artifact timing in the same
  `inference-benchmark-result-v0.1` row.

## Result

| Status | Engine | Model | Family | Quant | SQ mode | Impl | Target | Workload | Batching | Prefill total tok/s | Decode total tok/s | End-to-end tok/s | Consumed GiB | Decode x GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ok | uLLM | Qwen3.5-9B | FP8 | SQ8_0 | direct_fp8_dequant_matvec | single=sq8_0_matvec_rdna4_direct,triple=sq8_0_matvec_triple_rdna4_direct | R9700 | pp16/tg8/b1 | single | 49.89 | 73.16 | 35.83 | 4.17 | 304.89 |

Key fields:

- `quality.prompt_suite_regression_status`: `passed`
- `guards.prompt_guard_bundle.acceptance_mode`: `behavioral`
- `workload.sq_projection_boundary`: `single+triple`
- `workload.sq_fp8_single_matvec_count`: `24`
- `workload.sq_fp8_triple_matvec_count`: `120`
- `metrics.artifact_load_wall_time_seconds`: `11.941487135`
- `metrics.load_excluded_total_wall_time_seconds`: `0.669757925`
- `metrics.load_included_total_wall_time_seconds`: `12.868902806`

## 次の行動

- Run the vLLM smoke row from
  `docs/plans/r9700-qwen3-14b-fp8-external-engine-plan-v0.1.md`.
- Keep the model mismatch explicit until a same-model uLLM SQ8_0 row exists.

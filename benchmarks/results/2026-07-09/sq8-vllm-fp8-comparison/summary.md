# SQ8 vLLM FP8 Smoke Comparison

## 前回の要点

- M10 compares uLLM SQ8_0 rows with an external `vLLM + FP8` baseline after uLLM rows become
  implementation-valid.
- The vLLM smoke template uses `prompt_tokens=16`, `generated_tokens=8`, and
  `concurrent_requests=1`.
- The current uLLM SQ8_0 package path is Qwen3.5-9B, while the planned vLLM external baseline is
  Qwen3-14B-FP8.

## 今回の変更点

- Added a uLLM SQ8_0 smoke-shape row with `pp16/tg8/b1`.
- Ran the vLLM Qwen3-14B-FP8 smoke baseline on R9700 with `ROCR_VISIBLE_DEVICES=1`.
- Ran the vLLM Qwen3-14B-FP8 representative `pp512/tg128/b1` row on R9700.
- Attached the existing behavioral prompt-suite guard bundle.
- Preserved both rows in the same `inference-benchmark-result-v0.1` JSONL file.

## Result

| Status | Engine | Model | Family | Quant | SQ mode | Impl | Target | Workload | Batching | Prefill total tok/s | Decode total tok/s | End-to-end tok/s | Consumed GiB | Decode x GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ok | uLLM | Qwen3.5-9B | FP8 | SQ8_0 | direct_fp8_dequant_matvec | single=sq8_0_matvec_rdna4_direct,triple=sq8_0_matvec_triple_rdna4_direct | R9700 | pp16/tg8/b1 | single | 49.89 | 73.16 | 35.83 | 4.17 | 304.89 |
| ok | vLLM | Qwen3-14B-FP8 | FP8 | FP8 | - | - | R9700 | pp16/tg8/b1 | - | 31.19 | 15.59 | 46.78 | 28.71 | 447.63 |
| ok | vLLM | Qwen3-14B-FP8 | FP8 | FP8 | - | - | R9700 | pp512/tg128/b1 | - | 90.18 | 22.54 | 112.72 | 28.72 | 647.34 |

uLLM key fields:

- `quality.prompt_suite_regression_status`: `passed`
- `guards.prompt_guard_bundle.acceptance_mode`: `behavioral`
- `workload.sq_projection_boundary`: `single+triple`
- `workload.sq_fp8_single_matvec_count`: `24`
- `workload.sq_fp8_triple_matvec_count`: `120`
- `metrics.artifact_load_wall_time_seconds`: `11.941487135`
- `metrics.load_excluded_total_wall_time_seconds`: `0.669757925`
- `metrics.load_included_total_wall_time_seconds`: `12.868902806`

vLLM smoke key fields:

- `engine.version`: `0.23.1rc1.dev618+g8cf7c4d8a.rocm723`
- `engine.commit`: `8cf7c4d8ad602d73ff2ec72a101420d47163c136`
- `model.name`: `Qwen3-14B-FP8`
- `artifacts.elapsed_seconds`: `84.5010472680442`
- `metrics.requests_per_second`: `1.95`
- `memory.vram_consumed_bytes`: `30830026752`

vLLM representative key fields:

- `case_id`: `vllm-r9700-qwen3-14b-fp8-rep-pp512-tg128-tp1-rocr`
- `artifacts.elapsed_seconds`: `59.06105652800761`
- `metrics.requests_per_second`: `0.18`
- `metrics.prefill_tokens_per_second`: `90.17614034497254`
- `metrics.decode_tokens_per_second`: `22.54`
- `metrics.total_tokens_per_second`: `112.72`
- `memory.vram_consumed_bytes`: `30837428224`

Important limitation:

- These rows have matching smoke workload shape, target GPU, and result schema, but not matching
  model architecture/size. Treat this as a local feasibility and measurement-path comparison, not a
  same-model throughput conclusion.
- Same-model readiness is tracked in `same-model-readiness.md`; tensor namespace compatibility is
  handled in runtime lookup. The current blocker has moved from basic package connectivity to a
  full 40-layer `Qwen3-14B-FP8` uLLM row: the BF16 thin package plus layer0 SQ8_0 sidecar overlay
  reaches `verified=true`, but no full same-model throughput row has been produced yet.

## 次の行動

- Build the full 40-layer same-model uLLM SQ8_0 row before making final throughput claims.
- If same-model uLLM is not ready, keep future rows labeled as external feasibility baselines.

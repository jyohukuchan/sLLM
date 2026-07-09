# SQ8 vLLM FP8 Smoke Comparison

## 前回の要点

- M10 compares uLLM SQ8_0 rows with an external `vLLM + FP8` baseline after uLLM rows become
  implementation-valid.
- The first comparison rows used uLLM `Qwen3.5-9B` SQ8_0 and vLLM `Qwen3-14B-FP8`, so they were not
  same-model throughput evidence.
- The later work produced a BF16 thin package plus SQ8_0 sidecar artifact for local
  `Qwen3-14B-FP8`.

## 今回の変更点

- Added full 40-layer uLLM `Qwen3-14B-FP8` SQ8_0 rows with the same smoke and representative shapes
  as the vLLM FP8 baseline.
- Added config-aligned uLLM rows using local Qwen3 config values: `rotary_dim=128` and
  `rope_base=1000000`.
- Attached a self-behavioral prompt-suite smoke guard to the config-aligned uLLM rows.
- Refreshed the latest config-aligned smoke and representative rows after R9700 projection dispatch
  descriptors; both refreshed rows now report `*_r9700_direct` SQ8_0 projection implementation IDs.
- Added a full 40-layer mixed request-state real-batch uLLM diagnostic row with `TOP_K=0`, so final
  logits are excluded from total latency.
- Preserved the earlier `rotary_dim=32` / `rope_base=10000000` uLLM rows as preliminary connectivity
  rows, not final same-model rows.

## Result

| Status | Engine | Model | Quant | Config | SQ mode | Target | Workload | Prefill tok/s | Decode tok/s | End-to-end tok/s | Consumed GiB | Decode x GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ok | uLLM | Qwen3.5-9B | SQ8_0 | native package | direct_fp8_dequant_matvec | R9700 | pp16/tg8/b1 | 49.89 | 73.16 | 35.83 | 4.17 | 304.89 |
| ok | vLLM | Qwen3-14B-FP8 | FP8 | HF config | - | R9700 | pp16/tg8/b1 | 31.19 | 15.59 | 46.78 | 28.71 | 447.63 |
| ok | vLLM | Qwen3-14B-FP8 | FP8 | HF config | - | R9700 | pp512/tg128/b1 | 90.18 | 22.54 | 112.72 | 28.72 | 647.34 |
| ok | uLLM | Qwen3-14B-FP8 | SQ8_0 | preliminary rope32/theta1e7 | direct_fp8_dequant_matvec | R9700 | pp16/tg8/b1 | 2.77 | 3.02 | 0.33 | 12.82 | 38.69 |
| ok | uLLM | Qwen3-14B-FP8 | SQ8_0 | preliminary rope32/theta1e7 | direct_fp8_dequant_matvec | R9700 | pp512/tg128/b1 | 2.84 | 2.68 | 2.19 | 13.26 | 35.60 |
| ok | uLLM | Qwen3-14B-FP8 | SQ8_0 | config rope128/theta1e6 | direct_fp8_dequant_matvec | R9700 | pp16/tg8/b1 | 2.75 | 2.70 | 0.32 | 12.82 | 34.64 |
| ok | uLLM | Qwen3-14B-FP8 | SQ8_0 | config rope128/theta1e6 | direct_fp8_dequant_matvec | R9700 | pp512/tg128/b1 | 2.97 | 2.86 | 2.27 | 13.26 | 37.92 |
| ok | uLLM | Qwen3-14B-FP8 | SQ8_0 | config rope128/theta1e6 | direct_fp8_dequant_matvec | R9700 | pp16/tg8/b2 real-batch no-final | 15.42 | 15.71 | 15.51 | 12.49 | 196.19 |

Current same-model uLLM key fields:

- package: `/tmp/ullm-qwen3-14b-fp8-bf16-thin.ullm.d`
- artifact: `/tmp/ullm-qwen3-14b-fp8-full-sq8-artifact`
- `workload.sq_fp8_tensor_count`: `281`
- `workload.sq_passthrough_tensor_count`: `442`
- `workload.sq_projection_boundary`: `single+triple`
- `workload.sq_projection_implementation_ids` (latest config-aligned rows):
  `single=sq8_0_matvec_r9700_direct,triple=sq8_0_matvec_triple_r9700_direct`
- config-aligned smoke row:
  `ullm-r9700-qwen3-14b-fp8-sq8-smoke-pp16-tg8-b1-rope128-theta1e6-r9700dispatch`
- config-aligned representative row:
  `ullm-r9700-qwen3-14b-fp8-sq8-rep-pp512-tg128-b1-rope128-theta1e6-r9700dispatch`
- prompt-suite smoke:
  `benchmarks/results/2026-07-09/sq8-vllm-fp8-comparison/qwen3-14b-sq8-prompt-suite-smoke-rope128-theta1e6/summary.json`
- prompt guard bundle:
  `benchmarks/results/2026-07-09/sq8-vllm-fp8-comparison/qwen3-14b-sq8-prompt-suite-smoke-rope128-theta1e6/guard-self-behavioral/guard-bundle-summary.json`
- guard status: `quality.prompt_suite_regression_status=passed`,
  `scope=self_behavioral_prompt_suite_smoke`, `output_health=not_evaluated`
- prompt-suite smoke metrics: `verified_all=true`, `output_not_evaluated_count=1`, generated preview
  `准准`
- real-batch no-final-logits diagnostic row:
  `benchmarks/results/2026-07-09/sq8-qwen3-14b-full-mixed-real-batch-no-final-logits-smoke/results.jsonl`
- no-final-logits diagnostic key fields: `final_logits_in_total=false`,
  `sq_fp8_batch_matvec_count=6720/6720`, `sq_diagnostic_host_staging_read_count=0`,
  `sq_diagnostic_host_staging_write_count=72`

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

- The `rope128/theta1e6` uLLM rows are now same model, same GPU, same prompt/generated shape, and
  config-aligned with local `Qwen3-14B-FP8`.
- They are still not a final serving-performance conclusion. The b1 uLLM rows are measured through
  the current token-id model-loop path with final logits included and `prefill_real_batch=false` /
  `decode_real_batch=false`, while vLLM is measured through its throughput benchmark.
- The new `pp16/tg8/b2` uLLM row is real-batch and excludes final logits, but it does not yet have a
  matching vLLM `concurrent_requests=2` row and still uses the CLI model-loop harness.
- Multi-request mixed-state uLLM runs are classified as `batching_mode=grouped`, not real-batch,
  until batched projection kernels are actually used.
- The Qwen3-14B-FP8 uLLM rows have sampled `verified=true`, and the config-aligned rows now have a
  self-behavioral prompt-suite smoke guard. This verifies prompt-suite/guard plumbing only; it is not
  an external reference quality check because the current smoke suite has `output_health=false`.

## 次の行動

- Add a non-self behavioral guard or health-evaluated prompt suite before using the rows as final
  quality-regression evidence.
- Add a matched vLLM `concurrent_requests=2` row or a server-style uLLM path before using the
  real-batch row as final serving comparison.
- Keep the preliminary rope32/theta1e7 rows only as connectivity history.

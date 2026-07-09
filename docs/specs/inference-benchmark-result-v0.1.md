# Inference benchmark result v0.1

## Purpose

This schema records token/s measurements from existing inference engines before uLLM implements Qwen3 or locks any `aq` / `sq` quantization details.

The goal is to make every result comparable across engines, hardware, context length, batch settings, tensor parallelism, pipeline parallelism, and unsupported cases.

## Result Format

Results are stored as JSON Lines under `benchmarks/results/`.

Each line is one benchmark case. A case may be successful, failed, unsupported, or skipped.

## Required Fields

```json
{
  "schema_version": "inference-benchmark-result-v0.1",
  "run_id": "2026-06-30-llamacpp-v620-qwen3",
  "case_id": "llamacpp-qwen3-14b-q4km-ctx4096-b1-pp512-tg128",
  "status": "ok",
  "engine": {
    "name": "llama.cpp",
    "version": "0.15.3",
    "commit": "86b94708f224"
  },
  "model": {
    "name": "Qwen3-14B",
    "source": "huggingface",
    "revision": null,
    "format": "gguf",
    "quantization": "Q4_K_M"
  },
  "hardware": {
    "host": "homelab1-WRX80-Creator",
    "gpu_count": 1,
    "gpus": [
      {
        "name": "AMD Radeon Pro V620",
        "gfx": "gfx1030",
        "vram_bytes": 32195477504
      }
    ],
    "cpu": null,
    "driver": "6.16.13",
    "runtime": "ROCm 7.2.1"
  },
  "parallelism": {
    "tensor_parallel": 1,
    "pipeline_parallel": 1,
    "data_parallel": 1
  },
  "workload": {
    "context_length": 4096,
    "prompt_tokens": 512,
    "generated_tokens": 128,
    "batch_size": 1,
    "concurrent_requests": 1,
    "prompt_tokens_per_request": [512],
    "generated_tokens_per_request": [128],
    "kv_cache_dtype": "f16"
  },
  "metrics": {
    "prefill_tokens_per_second": 0.0,
    "decode_tokens_per_second": 0.0,
    "total_tokens_per_second": 0.0,
    "prefill_total_input_tokens": 512,
    "decode_total_generated_tokens": 128,
    "end_to_end_total_tokens": 640,
    "prefill_total_input_tokens_per_second": 0.0,
    "decode_total_generated_tokens_per_second": 0.0,
    "end_to_end_total_tokens_per_second": 0.0,
    "latency_p50_ms": null,
    "latency_p95_ms": null,
    "time_to_first_token_ms_p50": null,
    "time_to_first_token_ms_p95": null,
    "time_per_output_token_ms_p50": null,
    "time_per_output_token_ms_p95": null,
    "vram_baseline_bytes": null,
    "vram_peak_bytes": null,
    "vram_consumed_bytes": null,
    "decode_tokens_per_second_times_vram_consumed_gib": null,
    "power_watts_avg": null
  },
  "memory": {
    "backend": "rocm-smi",
    "sample_interval_seconds": 1.0,
    "sample_count": 0,
    "baseline_total_bytes": null,
    "peak_total_bytes": null,
    "consumed_total_bytes": null,
    "baseline_by_card_bytes": {},
    "peak_by_card_bytes": {},
    "consumed_by_card_bytes": {},
    "log": null
  },
  "artifacts": {
    "command": "llama-bench ...",
    "stdout_log": null,
    "stderr_log": null,
    "memory_log": null
  },
  "error": null,
  "notes": []
}
```

### Optional Prompt Guard Bundle Attachment (inference-benchmark-result-v0.1)

`run-external-benchmark` can add a `quality.prompt_suite_regression_status` and guard bundle details when
`--prompt-guard-bundle-json` is provided.

- `quality.prompt_suite_regression_status`: `passed` / `failed` / `not_attached`
- `guards.prompt_guard_bundle.status`: `ok`
- `guards.prompt_guard_bundle.artifact`: path to the prompt-suite guard JSON
- `guards.prompt_guard_bundle.passed`
- `guards.prompt_guard_bundle.acceptance_mode`
- `guards.prompt_guard_bundle.strict_passed`
- `guards.prompt_guard_bundle.behavioral_passed`
- `guards.prompt_guard_bundle.compared_case_count`
- `guards.prompt_guard_bundle.generated_token_match_count`
- `guards.prompt_guard_bundle.generated_text_match_count`
- `guards.prompt_guard_bundle.generated_without_stop_text_match_count`
- `guards.prompt_guard_bundle.top_logits_match_count`
- `guards.prompt_guard_bundle.max_prefill_top_logit_abs_diff`
- `guards.prompt_guard_bundle.max_decode_last_top_logit_abs_diff`
- `artifacts.prompt_guard_bundle_json`

## Status Values

- `ok`: the benchmark ran and metrics are valid.
- `unsupported`: the engine or hardware does not support the requested condition.
- `oom`: the case ran out of memory.
- `failed`: the case failed for another reason.
- `skipped`: intentionally not run.

## Unsupported Cases

Unsupported cases must still be written as JSONL rows. This matters for TP/PP, multi-GPU, and V620 limitations.

Example:

```json
{
  "schema_version": "inference-benchmark-result-v0.1",
  "run_id": "2026-06-30-v620-reference-engines",
  "case_id": "vllm-v620-tp1-ctx4096",
  "status": "unsupported",
  "engine": { "name": "vLLM", "version": null, "commit": "5b4cb6952310" },
  "model": { "name": "Qwen3-14B", "source": "huggingface", "revision": null, "format": "safetensors", "quantization": "bf16" },
  "hardware": { "host": "homelab1-WRX80-Creator", "gpu_count": 1, "gpus": [{ "name": "AMD Radeon Pro V620", "gfx": "gfx1030", "vram_bytes": 32195477504 }], "cpu": null, "driver": "6.16.13", "runtime": "ROCm 7.2.1" },
  "parallelism": { "tensor_parallel": 1, "pipeline_parallel": 1, "data_parallel": 1 },
  "workload": { "context_length": 4096, "prompt_tokens": 512, "generated_tokens": 128, "batch_size": 1, "concurrent_requests": 1, "kv_cache_dtype": "f16" },
  "metrics": null,
  "artifacts": { "command": null, "stdout_log": null, "stderr_log": null },
  "error": { "type": "unsupported_hardware", "message": "V620 is not an early execution target for vLLM." },
  "notes": []
}
```

## Required Comparison Axes

- engine
- engine commit
- model
- model format
- quantization
- context length
- prompt tokens
- generated tokens
- batch size
- concurrent requests
- tensor parallelism
- pipeline parallelism
- GPU count
- GPU model
- backend/runtime
- KV cache dtype
- quantization family: `K-Quant`, `I-Quant`, `UD`, `FP8`, or another explicit family
- VRAM baseline, peak, and consumed memory

## Metrics

At minimum:

- prefill tokens/s
- decode tokens/s
- total tokens/s
- prefill total input tokens/s for batch throughput runs
- decode total generated tokens/s for batch throughput runs
- end-to-end total tokens/s for batch throughput runs
- prefill wall time in seconds
- decode wall time in seconds
- total wall time in seconds
- VRAM baseline before the command
- peak VRAM during the command
- consumed VRAM, defined as peak total used bytes minus baseline total used bytes
- `decode_tokens_per_second_times_vram_consumed_gib`
- unsupported/OOM reason if metrics are unavailable

Latency and power metrics are optional in v0.1.

For uLLM pre-sq runtime runs, extend `metrics` with these optional fields:

```json
{
  "prefill_wall_time_seconds": 0.0,
  "decode_wall_time_seconds": 0.0,
  "total_wall_time_seconds": 0.0,
  "time_to_first_token_ms": null,
  "time_per_output_token_ms": null,
  "prefill_total_input_tokens": 0,
  "decode_total_generated_tokens": 0,
  "generated_tokens_total": 0,
  "end_to_end_total_tokens": 0,
  "prefill_total_input_tokens_per_second": null,
  "decode_total_generated_tokens_per_second": null,
  "end_to_end_total_tokens_per_second": null,
  "time_to_first_token_ms_p50": null,
  "time_to_first_token_ms_p95": null,
  "request_latency_ms_p50": null,
  "request_latency_ms_p95": null,
  "time_per_output_token_ms_p50": null,
  "time_per_output_token_ms_p95": null
}
```

## Batch Throughput Semantics

For batch throughput rows, `total_tokens_per_second` is a compatibility field. The comparison key
is the explicit total-throughput field that matches the phase:

- `prefill_total_input_tokens_per_second`: total prompt/input tokens processed during prefill divided
  by prefill wall time.
- `decode_total_generated_tokens_per_second`: total timed generated tokens produced during decode
  divided by decode wall time.
- `end_to_end_total_tokens_per_second`: prompt/input plus generated tokens divided by command wall
  time.

For uLLM `package-batch-throughput-bench-v0.1` reports, the raw field names are
`metrics.prefill_total_input_tps`, `metrics.decode_total_generated_tps`, and
`metrics.end_to_end_total_tps`. When these reports are converted into
`inference-benchmark-result-v0.1` JSONL rows, map them to the corresponding
`*_tokens_per_second` fields above.

Raw uLLM package batch reports also preserve prefill workload accounting fields used by the FP8/SQ
planning grid:

- `workload.prefill_mode`: `cold`, `cached_prefix`, or `decode` when supported by that runner.
- `workload.prefill_executor`: requested prefill executor policy when supplied by the workload.
- `workload.resolved_prefill_executor`: concrete executor used when an auto policy is resolved.
- `workload.cached_prefix_tokens_per_request`
- `workload.new_prefill_tokens_per_request`
- `workload.total_context_tokens_after_prefill_per_request`
- `metrics.cached_prefix_total_tokens`
- `metrics.total_context_tokens_after_prefill`
- `metrics.estimated_prefill_attention_work_tokens`
- `batching.prefill_executor`
- `batching.prefill_real_batch`
- `batching.decode_executor`
- `memory.kv_cache_bytes_total`

For cold prefill, cached-prefix fields are zero and `estimated_prefill_attention_work_tokens` is the
sum of `N * (N + 1) / 2` over requests. These fields are diagnostic/context columns; the comparison
throughput keys remain the explicit total-throughput fields above.

For cached-prefix component-derived SQ rows, record both `executor` and `resolved_executor` when the
source report has an auto executor. `executor` names the requested policy, while
`resolved_executor` names the concrete kernel path used for that case.

For uLLM component prefill real-batch rows parsed with `--parse ullm-component-prefill`, preserve:

- `metrics.prefill_total_input_tokens_per_second`
- `metrics.attention_pair_tps_mean`
- `workload.prompt_tokens_per_request`
- `workload.new_prefill_tokens_per_request`
- `workload.estimated_prefill_attention_work_tokens`
- `batching.mode`
- `batching.prefill_real_batch`
- `batching.prefill_executor_request_parallelism`
- `batching.prefill_executor_token_parallelism`

These rows are real-batch component rows, not full package throughput rows. They can prove a kernel
or component is using request/token parallelism, but they must not be used as final SQ package
throughput until connected to the package runner.

The same parser also accepts package-backed component smoke output such as
`package-prefill-aq4-matvec-batch-smoke`. Those rows should preserve `batching.component_package`
and may derive `prefill_total_input_tokens_per_second` from `token_tps_mean` when the component
stdout does not emit the package-batch field name directly. These rows prove that the real-batch
component path is connected to a `.ullm.d` package, but they are still not whole-model package total
throughput rows.

SQ8_0 package-backed component rows, such as `sq-fp8-package-self-attn-layer-batch-smoke`, should
also preserve SQ projection telemetry under `workload`: `sq_execution_mode`,
`sq_projection_boundary`, `sq_projection_implementation_ids`, `sq_fp8_batch_matvec_count`, and
`sq_fp8_expected_all_batch_matvec_count`. A row with `batching.mode="real"` and
`sq_fp8_batch_matvec_count == sq_fp8_expected_all_batch_matvec_count` proves the selected component
used the direct SQ8_0 batch projection boundary. It is still not a full-package serving row.

For package-backed component rows that flatten a workload batch into token parallelism, preserve the
requested workload batch and the executor's actual parallelism separately. A row may therefore have
`workload.batch_size=4`, `workload.prompt_tokens_per_request=[2,2,2,2]`, and
`workload.component_total_input_tokens=8`, while `batching.prefill_executor_request_parallelism=1`
and `batching.prefill_executor_token_parallelism=8`.

Model-loop stack smokes may be converted with `--parse ullm-model-loop-throughput`. These rows are
selected-layer stack rows, not final full language-model throughput rows. They may use
`batching.mode="hybrid"` when prefill is still sequential per request/token but decode uses a real
ready-batch executor. Such rows must preserve:

- `batching.prefill_real_batch`
- `batching.decode_real_batch`
- `batching.decode_executor_request_parallelism`
- `workload.layers_csv`
- `workload.prompt_tokens_per_request`
- `workload.generated_tokens_per_request`
- `workload.final_top1_tokens` when a final LM head guard is present
- `workload.final_topk_tokens` and `workload.final_topk_logits` when top-k logits are emitted
- `metrics.prefill_total_input_tokens_per_second`
- `metrics.decode_total_generated_tokens_per_second`
- `metrics.end_to_end_total_tokens_per_second`
- `metrics.artifact_load_wall_time_seconds`
- `metrics.artifact_materialization_wall_time_seconds`
- `metrics.load_excluded_total_wall_time_seconds`
- `metrics.load_included_total_wall_time_seconds`

For model-loop rows, `load_excluded_total_wall_time_seconds` is the measured command section used
for throughput, while `load_included_total_wall_time_seconds` includes the surrounding layer or
artifact load section when the runner reports it. `artifact_load_wall_time_seconds` should use an
explicit artifact-load timer when present and may fall back to the legacy `layer_load_ms` timer.
`artifact_materialization_wall_time_seconds` is only populated when the runner reports a separate
materialization timer; it must not alias `layer_load_ms`.

`hybrid` rows are useful for connecting scheduler/runtime request batching to the result schema, but
they must not be mixed with final `real` full-package throughput rows in SQ/vLLM comparisons.

SQ8_0 resident stack diagnostics, such as `sq-fp8-package-self-attn-stack-batch-smoke`, may also
use the model-loop parser. These rows prove that the stack path can avoid F32 materialized SQ8_0
weights and use resident direct projection boundaries. Rows with `batching.mode="real"`,
`prefill_real_batch=true`, and `decode_real_batch=true` prove real-batch projection for the reported
boundary. If `sq_fp8_batch_matvec_count < sq_fp8_expected_all_batch_matvec_count`, only part of the
selected SQ8_0 projection set is batched, so the row remains a resident stack diagnostic rather than
a final all-projection row. Even when those counters match, selected-layer stack rows remain
diagnostic until the same execution path is represented in full-package or server-style rows.

Full mixed request-state rows, such as `sq-fp8-token-ids-mixed-request-state-smoke`, are not
selected-layer diagnostics when they use `layers_csv` covering the full model. If these rows report
`batching.mode="real"` and `sq_fp8_batch_matvec_count == sq_fp8_expected_all_batch_matvec_count`,
they are implementation-valid full model-loop real-batch rows. They should still remain separate
from serving rows when final logits are included in total latency or when host staging is part of the
diagnostic resident path.
Rows may expose that staging under `workload.sq_diagnostic_host_staging_read_count`,
`workload.sq_diagnostic_host_staging_write_count`, `workload.sq_diagnostic_host_staging_read_bytes`,
and `workload.sq_diagnostic_host_staging_write_bytes`. Nonzero values mean the row is still
diagnostic for serving comparison even when all SQ8_0 projection boundaries are batched.
Strict no-host-staging gates require all four fields to be present and equal to zero, so rows that
predate these counters cannot silently satisfy final SQ8_0/vLLM comparison checks.

SQ candidate rows may also carry a top-level `candidate` object:

- `candidate.id`: e.g. `sq-fp8-w8a16-r9700-v0`
- `candidate.artifact`: artifact directory or package path used for the run

`batch_size` is the number of requests in one scheduling step. `concurrent_requests` is the number
of live requests in the run. They are equal for fixed prompt/decode benchmark grids, but they may
diverge once dynamic scheduling is implemented.

uLLM may record `batching.mode` outside this generic schema. `logical` means requests are accounted
as a batch but executed through sequential single-request paths. `hybrid` means only part of the
path, usually decode, shares kernels or scheduler state across requests. `real` means the measured
full-package path uses request-batch execution for the relevant phase. Only `real` rows should be
used to compare production batch throughput against vLLM.

Raw uLLM package reports may also include `prefill.layer_step_summary`. This is diagnostic data for
prefill optimization, not a top-level comparison key. It records layer-by-layer token-loop timing and
optional component timing when component synchronization is enabled.

`prefill.executor` identifies the uLLM prefill path used by a raw package report. Current values
include `layer_major_host_token_loop` for the original host-readback loop and `device_token_loop` for
the experimental device-to-device token loop. `device_token_loop` reduces host boundaries but is not
real batch prefill by itself.

uLLM reports may also include executor-granularity fields such as
`prefill.real_batch`, `prefill.token_parallelism`, `prefill.request_parallelism`,
`batching.prefill_real_batch`, `batching.prefill_executor_token_parallelism`,
`batching.prefill_executor_request_parallelism`, `batching.decode_real_batch`, and
`batching.decode_executor_request_parallelism`. These fields describe the executor's actual kernel
sharing, not the workload's requested concurrency. A logical batch row can therefore have
`workload.concurrent_requests > 1` while `batching.prefill_executor_request_parallelism == 1`.

## Memory Semantics

Memory must be recorded for throughput runs. On ROCm, use `rocm-smi --showmeminfo vram --json` or an equivalent runtime API. The preferred values are:

- `vram_baseline_bytes`: total used VRAM immediately before the engine command starts.
- `vram_peak_bytes`: maximum total used VRAM observed while the command runs.
- `vram_consumed_bytes`: `vram_peak_bytes - vram_baseline_bytes`, clamped at zero.
- `memory.*_by_card_bytes`: raw per-card values as reported by the monitoring backend.

The aggregate total is the comparison key. Per-card names may not match runtime device names exactly on every ROCm system, so they are diagnostic metadata unless a backend provides stable runtime-device mapping.

Tables derived from this schema should include:

- decode tokens/s
- consumed VRAM in GiB
- `decode tokens/s * consumed VRAM GiB`

The product is only a reference column. It is not a quality score by itself.

For uLLM pre-sq runtime runs, extend `memory` with KV cache accounting:

```json
{
  "kv_cache_bytes": null,
  "kv_cache_allocated_blocks": null,
  "kv_cache_free_blocks": null,
  "kv_cache_block_size": null
}
```

## Correctness

uLLM pre-sq runtime runs should include a top-level optional `correctness`
object. Long throughput runs do not need full reference comparison, but they
must record enough sanity data to detect broken runs.

```json
{
  "reference": "hf|golden_fixture|none",
  "reference_artifact": null,
  "logits_relative_mse": null,
  "logits_max_abs_diff": null,
  "top_k": 10,
  "top_k_agreement": null,
  "generated_prefix_matches_reference": null,
  "nan_count": 0,
  "inf_count": 0,
  "logit_min": null,
  "logit_max": null
}
```

For short correctness cases, prefer `hf` or `golden_fixture` and fill the
logits/top-k fields. For long TPS cases, `reference` may be `none`, but
`nan_count`, `inf_count`, and logit range should still be recorded.

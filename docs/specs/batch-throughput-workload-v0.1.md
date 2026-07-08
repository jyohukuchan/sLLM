# Batch throughput workload manifest v0.1

This document defines the input manifest consumed by
`tools/run-batch-throughput-workload.py`.

The manifest is an execution plan, not a benchmark result. Result rows are still
written as `inference-benchmark-result-v0.1` JSONL through
`tools/run-external-benchmark.py`.

## Required Fields

```json
{
  "schema_version": "ullm-batch-throughput-workload-v0.1",
  "run_id": "2026-07-07-aq4-r9700-grid",
  "package_dir": "/path/to/model.ullm.d",
  "engine": "target/release/ullm-engine",
  "engine_name": "uLLM",
  "model_name": "Qwen3.5-9B",
  "model_format": "ullm-package",
  "model_quantization": "AQ4",
  "gpu_card": "card2",
  "device_index": 2,
  "cases": [
    {
      "case_id": "aq4-r9700-b1-pp128-tg32",
      "concurrent_requests": 1,
      "prompt_tokens": 128,
      "generated_tokens": 32
    }
  ]
}
```

`prompt_token_ids_batch` defaults to `len:PROMPTxCONCURRENT`. `generated_tokens_batch` defaults to
the scalar `generated_tokens`, which applies the same fixed decode length to every request.

## Optional Defaults

The root object may provide defaults used by every case:

- `chunk_bytes`
- `layers`
- `top_k`
- `lm_head_chunk_rows`
- `rotary_dim`
- `rope_base`
- `position_offset`
- `lm_head_mode`
- `kv_cache_dtype`
- `sq_candidate`
- `candidate_artifact`
- `prefill_executor`
- `resolved_prefill_executor`
- `warmup_runs`
- `measured_runs`
- `timeout_seconds`
- `memory_sample_interval`
- `require_hip_kernels`
- `env`
- `notes`

Each case may override `context_length`, `warmup_runs`, `measured_runs`, `timeout_seconds`,
`prompt_token_ids_batch`, `generated_tokens_batch`, and `notes`.

For R9700 SQ candidate runs, use:

```json
{
  "model_quantization": "SQ-FP8-W8A16",
  "sq_candidate": "sq-fp8-w8a16-r9700-v0",
  "candidate_artifact": "artifacts/sq-fp8-w8a16-r9700-v0",
  "prefill_executor": "cached_prefix_rdna4_fp8_auto",
  "resolved_prefill_executor": null
}
```

`resolved_prefill_executor` may be `null` in the workload manifest. Result rows must preserve the
runtime-resolved executor when a component runner reports one.

## Output Layout

Given `--output-dir OUT`, the runner writes:

```text
OUT/
  workload.json
  execution-plan.json
  warmup.jsonl
  results.jsonl
  CASE_ID/
    warmup-0/
      raw.json
      stdout.log
      stderr.log
      memory.jsonl
    measured-0/
      raw.json
      stdout.log
      stderr.log
      memory.jsonl
```

`warmup.jsonl` is diagnostic. `results.jsonl` is the measured aggregate file used for tables.

## Semantics

The workload runner does not make logical batch into real batch. It records and preserves the
`batching.mode` reported by `ullm-engine package-batch-throughput-bench`.

For SQ comparison rows, converted `inference-benchmark-result-v0.1` JSONL must preserve:

- `workload.prefill_mode`
- `workload.prefill_executor`
- `workload.resolved_prefill_executor`
- `workload.cached_prefix_tokens_per_request`
- `workload.new_prefill_tokens_per_request`
- `workload.total_context_tokens_after_prefill_per_request`
- `workload.cached_prefix_total_tokens`
- `workload.total_context_tokens_after_prefill`
- `workload.estimated_prefill_attention_work_tokens`
- `batching.prefill_executor`
- `batching.resolved_prefill_executor`
- `batching.prefill_real_batch`
- `memory.kv_cache_bytes_total`

For R9700 AQ4 package measurements, set `require_hip_kernels: true`. To select the experimental
device-resident token-loop prefill path, set:

```json
{
  "env": {
    "ULLM_PREFILL_DEVICE_TOKEN_LOOP": "1"
  }
}
```

This path reduces host boundaries but is still not real batch prefill.

## Component Real-Batch Rows

`tools/run-external-benchmark.py --parse ullm-component-prefill` can convert uLLM component prefill
smoke output, such as `runtime-causal-attn-batch-smoke`, into the same
`inference-benchmark-result-v0.1` JSONL schema.

These rows may report:

- `batching.mode = "real"`
- `batching.prefill_real_batch = true`
- `batching.prefill_executor_request_parallelism`
- `batching.prefill_executor_token_parallelism`
- `metrics.prefill_total_input_tokens_per_second`
- `metrics.attention_pair_tps_mean`

Component real-batch rows are useful for validating kernel-level request/token parallelism and
schema preservation. They are not full package throughput rows until a package prefill/decode runner
uses the same real-batch executor path.

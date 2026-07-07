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
- `warmup_runs`
- `measured_runs`
- `timeout_seconds`
- `memory_sample_interval`
- `require_hip_kernels`
- `env`
- `notes`

Each case may override `context_length`, `warmup_runs`, `measured_runs`, `timeout_seconds`,
`prompt_token_ids_batch`, `generated_tokens_batch`, and `notes`.

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

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

`batch_size` is the number of requests in one scheduling step. `concurrent_requests` is the number
of live requests in the run. They are equal for fixed prompt/decode benchmark grids, but they may
diverge once dynamic scheduling is implemented.

uLLM may record `batching.mode` outside this generic schema. `logical` means requests are accounted
as a batch but executed through sequential single-request paths. `real` means prefill and/or decode
shares kernels, weight residency, or scheduler state across requests. Only `real` rows should be
used to compare production batch throughput against vLLM.

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

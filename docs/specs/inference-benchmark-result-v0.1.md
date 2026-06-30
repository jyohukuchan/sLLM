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
    "kv_cache_dtype": "f16"
  },
  "metrics": {
    "prefill_tokens_per_second": 0.0,
    "decode_tokens_per_second": 0.0,
    "total_tokens_per_second": 0.0,
    "latency_p50_ms": null,
    "latency_p95_ms": null,
    "vram_peak_bytes": null,
    "power_watts_avg": null
  },
  "artifacts": {
    "command": "llama-bench ...",
    "stdout_log": null,
    "stderr_log": null
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

## Metrics

At minimum:

- prefill tokens/s
- decode tokens/s
- total tokens/s
- peak VRAM if available
- unsupported/OOM reason if metrics are unavailable

Latency and power metrics are optional in v0.1.

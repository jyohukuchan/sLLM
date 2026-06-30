# aq experiment result schema v0.1

## Purpose

This document defines the initial JSONL row shape for `aq` quantization experiments.

The schema is intentionally experimental. It records candidate parameters and validation metrics without declaring any `aq` candidate stable.

## JSONL Row

Each row is one candidate run for one scope.

Required top-level fields:

- `schema_version`: `aq-experiment-result-v0.1`
- `run_id`
- `timestamp_utc`
- `status`: `ok`, `failed`, `skipped`
- `model`
- `scope`
- `candidate`
- `inputs`
- `metrics`
- `artifacts`
- `notes`

## Model

```json
{
  "name": "Qwen3-14B",
  "source": "huggingface",
  "path": "/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3-14B",
  "dtype_reference": "bf16",
  "revision": null
}
```

## Scope

`scope` describes what was quantized or sampled.

```json
{
  "type": "tensor_sample",
  "tensor_names": ["model.layers.0.self_attn.q_proj.weight"],
  "families": ["attn_q"],
  "sample_elements_per_tensor": 1048576,
  "seed": 0
}
```

Allowed `scope.type` values:

- `tensor_sample`
- `tensor_full`
- `layer_replay`
- `model_eval`
- `runtime_microbench`

## Candidate

```json
{
  "candidate_id": "aq4_e8m0_g32_zf15",
  "index_bits": 4,
  "codebook": {
    "mode": "zero_free15",
    "storage_dtype": "bf16",
    "granularity": "per_family",
    "entry_count": 16
  },
  "scale": {
    "format": "e8m0",
    "bits": 8,
    "group_size": 32,
    "granularity": "per_group",
    "tensor_scale": "none",
    "family_scale": "none"
  },
  "group_layout": {
    "axis": "contiguous",
    "tile_shape": null
  },
  "optimizer": {
    "objective": "mse",
    "weighted": false,
    "scale_search": "exhaustive_256",
    "codebook_update": "coordinate_descent"
  }
}
```

## Metrics

Tensor metrics:

```json
{
  "effective_bpp": 4.25,
  "mse": 0.0,
  "relative_mse": 0.0,
  "weighted_mse": null,
  "max_abs_error": 0.0,
  "cosine_similarity": 1.0,
  "saturation_rate": 0.0,
  "zero_preservation_rate": 1.0,
  "mean_group_error": 0.0,
  "p95_group_error": 0.0
}
```

Layer/model/runtime metrics are added as optional fields when the scope requires them:

- `layer_output_mse`
- `layer_output_cosine_similarity`
- `perplexity`
- `eval_accuracy`
- `dequant_gib_per_second`
- `gemm_tokens_per_second`
- `vram_consumed_bytes`

## Artifacts

```json
{
  "result_json": "benchmarks/results/2026-06-30/aq/example.result.json",
  "sample_manifest": "benchmarks/results/2026-06-30/aq/example.sample.json",
  "codebook_path": "benchmarks/results/2026-06-30/aq/example.codebook.json"
}
```

Artifacts should be small for sampling runs. Full model payloads must not be stored under `benchmarks/results/`.

## Failure Rows

Failure rows keep the same candidate metadata and include:

```json
{
  "status": "failed",
  "error": {
    "type": "unsupported_scale_format",
    "message": "scale format requires arbitrary LUT decode"
  }
}
```

This makes rejected candidates comparable to successful candidates.

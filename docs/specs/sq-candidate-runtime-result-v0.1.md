# SQ Candidate Runtime Result v0.1

## Purpose

This schema records one runtime comparison row for an `sq` format candidate.

The row is not a general inference benchmark. It is specifically for comparing an `sq` candidate
against the current AQ4 prototype gate before the `sq` format is accepted as useful.

## Result Format

Results are stored as JSON Lines under `benchmarks/results/`.

Each row is one candidate run on one target GPU. A complete candidate comparison normally has one
R9700/RDNA4 row and, if RDNA2 support is claimed, one V620/RDNA2 row.

## Required Fields

```json
{
  "schema_version": "sq-candidate-runtime-result-v0.1",
  "run_id": "2026-07-06-sq-v0-candidate-a",
  "case_id": "sqv0-a-qwen35-9b-r9700-v03-suite",
  "status": "ok",
  "candidate": {
    "id": "sqv0_a",
    "format_version": "sq-format-v0.1",
    "description": "compact resident AQ-derived layout with layer-window materialization",
    "package_or_runtime_artifact": "/tmp/example-sqv0-a.ullm.d",
    "source_aq_policy": "qwen35_9b_p4p46_hidden3994_v1",
    "row_scale_override_policy": "preserved"
  },
  "model": {
    "name": "Qwen3.5-9B",
    "format": "ullm-package",
    "tokenizer": "/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B"
  },
  "hardware": {
    "host": "WRX80",
    "device_index": 2,
    "gpu_name": "Radeon AI PRO R9700",
    "architecture": "RDNA4",
    "backend": "hip"
  },
  "workload": {
    "suite": "benchmarks/prompts/pre-sq-runtime-prompt-suite-v0.3.json",
    "batch_size": 1,
    "tensor_parallel": 1,
    "sampling": "greedy",
    "kv_cache_dtype": "f32"
  },
  "storage": {
    "compact_resident_bytes": 0,
    "materialized_working_set_bytes": 0,
    "materialization_granularity": "layer_window",
    "whole_model_f32_resident": false,
    "kv_cache_bytes": null
  },
  "timing": {
    "materialization_wall_ms": 0.0,
    "prefill_tps_mean": 0.0,
    "decode_tps_mean": 0.0,
    "decode_tps_min": 0.0,
    "decode_tps_max": 0.0,
    "decode_p50_ms_mean": null
  },
  "quality": {
    "output_ok_count": 0,
    "output_warn_count": 0,
    "output_not_evaluated_count": 0,
    "verified_all": false
  },
  "guards": {
    "golden_prefix": {
      "status": "ok",
      "artifact": "benchmarks/results/.../package-golden-prefix.jsonl",
      "verified": true
    },
    "prompt_guard_bundle": {
      "status": "ok",
      "artifact": "benchmarks/results/.../guard-bundle-summary.json",
      "passed": true
    },
    "external_logits": {
      "status": "deferred",
      "artifact": null,
      "passed": null
    }
  },
  "artifacts": {
    "suite_summary_json": "benchmarks/results/.../summary.json",
    "suite_summary_md": "benchmarks/results/.../summary.md",
    "guard_bundle_json": "benchmarks/results/.../guard-bundle-summary.json",
    "command_log": null
  },
  "baseline": {
    "id": "aq4-rdna-prototype-2026-07-06",
    "r9700_decode_tps_mean": 19.796,
    "v620_decode_tps_mean": 15.434,
    "guard_bundle_artifact": "benchmarks/results/2026-07-06/engine/prompt-suite-aq4-pagedattn-r9700-v620-v0.3-guard-bundle/guard-bundle-summary.json"
  },
  "decision": {
    "comparable_to_baseline": false,
    "accepted_for_next_iteration": false,
    "reason": null
  },
  "notes": []
}
```

## Status Values

- `ok`: the candidate ran and all required metrics are present.
- `failed`: the run failed.
- `oom`: the run ran out of memory.
- `unsupported`: the candidate is not expected to support this target.
- `skipped`: intentionally not run.

Rows with `failed`, `oom`, `unsupported`, or `skipped` status should still include `candidate`,
`model`, `hardware`, `workload`, `artifacts`, `decision`, and an `error` object.

## Required Gate Semantics

An `sq` candidate is comparable to the AQ4 prototype only when:

- `status` is `ok`;
- `storage.compact_resident_bytes` is present;
- `storage.materialized_working_set_bytes` is present;
- `storage.materialization_granularity` is present;
- `timing.materialization_wall_ms` is present;
- `timing.decode_tps_mean` is present;
- `quality.verified_all` is true;
- `guards.golden_prefix.verified` is true;
- `guards.prompt_guard_bundle.passed` is true.

If a row does not satisfy these, set `decision.comparable_to_baseline` to `false` and explain why in
`decision.reason`.

## Baseline Anchor Rows

The current AQ4 prototype baseline may be recorded in this schema as a baseline anchor row so that
future `sq` rows have a machine-readable comparison target.

For these rows:

- `candidate.format_version` may be `aq4-prototype-current-runtime`;
- `decision.reason` should state that the row is a baseline anchor;
- `storage.compact_resident_bytes`, `storage.materialized_working_set_bytes`, and
  `timing.materialization_wall_ms` may be `null` if they were not measured for the current runtime.

This exception applies only to AQ4 baseline anchor rows. An actual `sq` candidate must still satisfy
the required gate semantics above to be comparable.

## Memory Semantics

- `compact_resident_bytes`: bytes that must remain resident in compact `sq` form during the run.
- `materialized_working_set_bytes`: maximum bytes materialized into an execution dtype at one time.
- `materialization_granularity`: one of `tensor`, `projection_group`, `layer`, `layer_window`,
  `model`, or a more specific documented string.
- `whole_model_f32_resident`: must be `false` for a useful first `sq` candidate.

The first `sq` candidate may be useful even if decode TPS is similar to the AQ4 prototype, but only
if compact residency or materialized working-set bytes are materially better and the guard bundle
passes.

## Guard Semantics

`prompt_guard_bundle` should be produced by:

```text
tools/run-package-prompt-guard-bundle.py
```

The bundle currently covers:

- v0.3 prompt generated-token agreement;
- v0.3 prefill top-logits agreement;
- v0.3 final decode-step top-logits agreement;
- stop condition agreement;
- output status agreement;
- optional standalone logits guard.

`external_logits` remains optional in v0.1 because the current CPU full-model logits path is too
slow for routine use. If it is not run, set `status` to `deferred`.

## Decision Guidance

Do not accept a candidate solely because it is faster.

Accept a candidate for the next iteration only if it improves at least one of:

- compact resident bytes;
- materialized working-set bytes;
- materialization granularity;
- decode TPS without output or guard regression;
- RDNA2/RDNA4 portability under the same gate.

Reject or keep iterating if:

- the candidate fails the guard bundle;
- quality-scored v0.3 cases gain unexplained warnings;
- whole-model f32 residency remains required;
- memory fields are missing;
- speed improves only by dropping required correctness or output-health gates.

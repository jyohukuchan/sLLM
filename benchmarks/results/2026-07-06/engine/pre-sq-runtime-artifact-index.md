# Pre-SQ Runtime Artifact Index 2026-07-06

## Purpose

Freeze the starting state for `docs/plans/pre-sq-runtime-tps-plan-v0.1.md` T0.
The next implementation phase needs stable package paths, device IDs, output
paths, and a benchmark record shape before adding end-to-end token-ID runtime
work.

## Source Plan

- `docs/plans/pre-sq-runtime-tps-plan-v0.1.md`
- T0 goal: state freeze and benchmark contract
- T0 deliverables:
  - this artifact index
  - benchmark JSON example

## Git State

- commit: `1eda85ada3b967895cd49dd79d0ea7327bb77bd7`
- latest commit: `1eda85a Add pre-SQ runtime TPS plan`

## Accepted Package

Named-policy package:

```text
/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-qwen35-hidden3994-v1-row-scale-layer6-layer10.ullm.d
```

The package exists on the local filesystem at T0 freeze time.

Package evidence:

- package summary: `benchmarks/results/2026-07-05/engine/ullm-quant-direct-package-fullpkg-qwen35-9b-qwen35-hidden3994-v1-row-scale-layer6-layer10-jobs64.json`
- package verify log: `benchmarks/results/2026-07-05/engine/ullm-quant-direct-package-fullpkg-qwen35-9b-qwen35-hidden3994-v1-row-scale-layer6-layer10-jobs64-verify.log`
- policy: `qwen35_9b_p4p46_hidden3994_v1`
- selected tensors: `255`
- passthrough tensors: `520`
- codebooks: `14`
- total file bytes: `9127853385`

Baseline comparison package:

```text
/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer10.ullm.d
```

The baseline package also exists on the local filesystem at T0 freeze time.

## Correctness Evidence

Named-policy five-fixture gate:

- matrix: `benchmarks/results/2026-07-05/engine/qwen-prefix-smoke-matrix-qwen35-hidden3994-policy-cpu-five-fixture/summary.json`
- summary: `benchmarks/results/2026-07-05/engine/qwen-prefix-qwen35-hidden3994-policy-cpu-five-fixture-summary.json`
- gate: `benchmarks/results/2026-07-05/engine/qwen-prefix-qwen35-hidden3994-policy-cpu-five-fixture-gates.json`
- decision: `accept`
- fixture count: `5`
- mean improvement: `0.047989845275878906`
- median improvement: `0.02260303497314453`
- max regression: `0`

Backend checks already recorded before T0:

- R9700/RDNA4 five-fixture gate: `accept`
- V620/RDNA2 device index `1` five-fixture gate: `accept`
- V620/RDNA2 device index `3` five-fixture gate: `accept`

## Runtime Devices

`target/debug/ullm-engine inspect-devices` at T0 freeze time:

| engine device index | backend | runtime id | name | memory bytes | compute | role |
| ---: | --- | ---: | --- | ---: | ---: | --- |
| `0` | `cpu` | `0` | `host CPU fallback` | `0` | `0.0` | correctness/debug fallback |
| `1` | `hip` | `0` | `AMD Radeon Pro V620` | `32195477504` | `10.3` | RDNA2 primary compatibility target |
| `2` | `hip` | `1` | `AMD Radeon Graphics` | `34208743424` | `12.0` | RDNA4 primary speed target |
| `3` | `hip` | `2` | `AMD Radeon Pro V620` | `32195477504` | `10.3` | RDNA2 secondary compatibility target |

Preferred pre-sq benchmark devices:

- R9700/RDNA4: engine device index `2`
- V620/RDNA2 primary: engine device index `1`
- V620/RDNA2 secondary: engine device index `3`

`rocm-smi` notes:

- driver version: `6.16.13`
- reported cards:
  - `card0`: `AMD Radeon Pro V620`, `gfx1030`, `32195477504` bytes
  - `card1`: `AMD Radeon Pro V620`, `gfx1030`, `32195477504` bytes
  - `card2`: `AMD Radeon Graphics`, `gfx1201`, `34208743424` bytes

The `rocm-smi` card order differs from the uLLM runtime device order because
uLLM includes CPU device index `0` and maps HIP runtime IDs separately.

## Benchmark Contract

Pre-sq runtime benchmark records should use:

```text
schema_version = "inference-benchmark-result-v0.1"
engine.name = "uLLM"
model.name = "Qwen3.5-9B"
model.format = "ullm.d"
model.quantization = "qwen35_9b_p4p46_hidden3994_v1"
parallelism.tensor_parallel = 1
parallelism.pipeline_parallel = 1
workload.batch_size = 1
workload.concurrent_requests = 1
```

The schema remains compatible with
`docs/specs/inference-benchmark-result-v0.1.md`, with the added optional
pre-sq fields for wall times, KV cache accounting, and correctness sanity
checks.

Example record:

- `benchmarks/results/2026-07-06/engine/pre-sq-runtime-benchmark-example.json`

## Output Paths

Recommended output root for this phase:

```text
benchmarks/results/2026-07-06/engine/pre-sq-runtime/
```

Recommended files:

- `pre-sq-runtime-bench-r9700.jsonl`
- `pre-sq-runtime-bench-v620.jsonl`
- `pre-sq-runtime-bench-summary.json`
- `pre-sq-runtime-bench-summary.md`
- `logs/`

## Initial Grid

Required:

| prompt tokens | generated tokens | devices |
| ---: | ---: | --- |
| `128` | `32` | CPU, R9700, V620 |
| `512` | `256` | R9700, V620 |
| `2048` | `256` | R9700, V620 if memory allows |

Stretch:

| prompt tokens | generated tokens | devices |
| ---: | ---: | --- |
| `2048` | `512` | R9700 first |
| `4096` | `512` | R9700 only if memory allows |

## T0 Exit Decision

T0 is satisfied when:

- this artifact index exists,
- the benchmark example exists,
- the benchmark spec contains the needed pre-sq optional fields,
- package and device state are recorded from current command output.

After T0, start T1 by adding a token-ID logits smoke boundary that can load
embedding/final norm/lm_head and run from token IDs to final logits/top-k.

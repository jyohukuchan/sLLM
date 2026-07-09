# SQ8_0 Implementation Plan v0.1

## 前回の要点

- `FQ8_0` は会話上のtypoであり、正式なpublic format IDは `SQ8_0` とする。
- `SQ8_0` はFP8 E4M3を使う正式採用formatであり、FP8そのものの品質採否候補ではない。
- 旧 `sq-fp8-w8a16-r9700-v0` はpublic format IDではなく、`SQ8_0` のimplementation lineage / legacy aliasとして扱う。
- 既存実装では、artifact生成、manifest検証、row / row-block scale、SQ FP8 direct matvec、batch、pair、triple、selected-layer model-loop overlayまで進んでいる。
- 旧T2のstrict top1 / top-k / text guardは、今後は採用gateではなく実装回帰診断として扱う。

## 今回の変更点

- 旧FP8 SQ候補計画を、`SQ8_0` の正式実装計画として再定義する。
- 実装対象、非対象、runtime境界、backend dispatch方針、C++ kernel構成、回帰検証、完了条件を固定する。
- 直近の実装順を、既存部品を壊さずに進められる小さい単位へ分解する。

## 次の行動

1. 旧CLI名とartifact schemaを互換維持しながら、表示・結果・docs上のpublic format表記を `SQ8_0` へ寄せる。
2. CLI-localのSQ FP8 resident overlay処理をlibrary moduleへ移し、package/model-loop runtimeから再利用できる境界にする。
3. `backend_dispatch` を実runtime pathへ接続し、まずSQ8_0 matvec系とcached-prefix attention executor選択に使う。

## Objective

`SQ8_0` を、Qwen3/Qwen3.5系LM weightのFP8 E4M3実行formatとして実装する。

ここでの「実装完了」は、FP8の一般的な品質を証明することではない。次の条件を満たすことを意味する。

- `SQ8_0` artifactを仕様どおり生成できる。
- runtimeがartifactを仕様どおり読み、resident payload / scaleを保持できる。
- 対応GPUでは、F32 materialize fallbackではなくSQ8_0 direct pathで主要projectionを実行できる。
- model-loop / batch throughput / prompt-suite回帰診断で、実装ミスを検出できる。
- 結果schemaではpublic format ID `SQ8_0` とimplementation lineageを分けて記録できる。

## Naming And Compatibility

| item | rule |
| --- | --- |
| public format ID | `SQ8_0` |
| legacy aliases | `sq`, `sq-format-v0.1`, `sq-fp8*` |
| current implementation lineage | `sq-fp8-w8a16-r9700-v0` |
| initial payload dtype | `fp8_e4m3` |
| initial activation contract | W8A16-style: FP8 weights, BF16/F32 activations depending on runtime boundary |
| initial scale dtype | `f32` |
| initial scale layouts | `tensor`, `row`, `row_block`, and mixed per-tensor policy |

Compatibility rule:

- Readers must accept old `sq-fp8*` IDs.
- New artifacts and result rows should emit `candidate.id = "SQ8_0"` and `candidate.format_id = "SQ8_0"`.
- If a legacy implementation ID is used, it must be preserved as `candidate.implementation_id` or `legacy_*` metadata.

## Scope

Initial target:

- model family: Qwen3/Qwen3.5 decoder-only LM path;
- tensor kind: language-model 2D weight tensors;
- GPU target: R9700/RDNA4 first;
- runtime path: package-backed inference, selected-layer and model-loop smoke, then full-package throughput;
- storage: sidecar SQ8_0 artifact first, later integrated `.ullm.d` package format.

Initial tensor families:

- embedding and lm_head;
- self-attention `q/k/v/o`;
- Qwen3.5 self-attention output gate support when present;
- linear-attention projections when present;
- MLP `gate/up/down`.

Deferred:

- visual tower and MTP tensors;
- tensor parallel layout;
- full server API;
- tokenizer integration beyond existing package prompt tools;
- RDNA2/V620 native FP8 optimization. RDNA2 may use fallback/dequant paths, but it is not a blocker for R9700/RDNA4 `SQ8_0` implementation.

## Current Implementation State

Done:

- `AQ4_0` / `SQ8_0` format ID canonicalization and legacy alias handling.
- SQ FP8 artifact builder emits `SQ8_0` public metadata.
- Artifact manifest supports FP8 payload files, F32 scales, row-block scale, mixed policy metadata, and passthrough tensors.
- Runtime artifact parsing and selected tensor materialization smoke exist.
- SQ FP8 runtime APIs exist for:
  - single matvec;
  - batch matvec;
  - pair matvec;
  - triple matvec.
- `PackageAq4ResidentMatvec` has an `SqFp8` storage variant in the CLI smoke/runtime harness path.
- SQ8_0 selected-layer model-loop bridge exists through `sq-fp8-token-ids-model-loop-smoke`.
- R9700 cached-prefix FP8 KV cache and FlashAttention2-style executor experiments exist.
- RDNA4 FP8 WMMA probes exist.
- Qwen3-14B-FP8 now has a same-model connectivity path for the first layer:
  - `ullm-quant` can build a BF16-only thin `.ullm.d` shell with passthrough dtype/suffix filters;
  - `ullm-engine` can infer Qwen3 self-attention layers from thin package `q_norm/k_norm`
    passthrough tensors;
  - a layer0 SQ8_0 sidecar artifact reaches `sq-fp8-token-ids-logits-smoke` with `verified=true`.

Partial:

- SQ8_0 resident overlay logic is still too CLI-local.
- Runtime result labels still use many `sq-fp8-*` command names.
- Direct SQ8_0 execution is available at projection API boundaries, but not all higher-level fused layer boundaries are SQ8_0-aware.
- `backend_dispatch` exists as a typed selector, but is not yet wired into real runtime operations.
- C++ runtime still has broad `.inc` chunks rather than explicit model/GPU/format implementation modules.
- Qwen3-14B-FP8 same-model work has passed package/layer0 connectivity, but not the full 40-layer
  SQ8_0 throughput row.

Not done:

- Integrated SQ8_0 package format inside `.ullm.d`.
- Full-package SQ8_0 real-batch throughput row with artifact load/materialization accounting.
- Full 40-layer `Qwen3-14B-FP8` same-model uLLM row for comparison with vLLM.
- C++ implementation registry that can choose kernels by model architecture, GPU architecture, GPU name, phase, and format.
- SQ8_0 fused MLP / attention-specific direct kernels beyond matvec pair/triple boundaries.
- Stable regression suite named around `SQ8_0` rather than old SQ FP8 candidate language.

## Runtime Architecture

### Artifact Boundary

`sq_manifest.json` remains the source of truth for v0.1.

Required fields:

- public candidate ID and format ID;
- implementation ID;
- payload dtype;
- scale dtype and layout;
- tensor entries with shape, family, payload file, scale file, byte counts, and checksums;
- passthrough entries with explicit reasons;
- storage estimates;
- optional policy block.

Implementation rule:

- The runtime must not infer scale layout from file naming.
- Per-tensor manifest entries are authoritative.
- Row-block scale must validate `scale_block_cols > 0` and expected scale element count.

### Rust Loader Boundary

Create or extend a library-side SQ8_0 runtime module so CLI smokes stop owning the core behavior.

Target module options:

- `crates/ullm-engine/src/sq_runtime.rs`, or
- an expanded `crates/ullm-engine/src/qwen3_loader.rs` boundary if the behavior is Qwen-specific.

The preferred split is:

- `sq.rs`: artifact schema, manifest validation, FP8 decode helpers, row selection utilities.
- `sq_runtime.rs`: resident payload/scale loading, runtime buffer registration, SQ8_0 matvec storage references.
- `qwen3_loader.rs`: mapping model tensor names to Qwen3/Qwen3.5 layer runtime weights and SQ8_0 overlay application.

### Runtime Kernel Boundary

Keep these C ABI functions as the first stable SQ8_0 W8A16 surface:

- `ullm_runtime_sq_fp8_matvec_f32`
- `ullm_runtime_sq_fp8_matvec_batch_f32`
- `ullm_runtime_sq_fp8_matvec_pair_f32`
- `ullm_runtime_sq_fp8_matvec_triple_f32`

Next kernel boundaries:

- SQ8_0 gate/up fused MLP projection when both tensors are SQ8_0 and the chosen policy includes them.
- SQ8_0 Q/K/V fused projection for self-attention when all required tensors are SQ8_0.
- Optional mixed AQ4/SQ8_0 fused boundary only after all-SQ8_0 fused boundary is stable.

Fallback rule:

- CPU/HIP staging fallback is allowed for correctness smoke.
- Performance rows must mark fallback explicitly with `sq_execution_mode`.
- A result must not be counted as native SQ8_0 throughput if it used F32 materialized fallback for the measured projection path.

## Backend Dispatch Plan

Use the existing Rust `backend_dispatch` model as the high-level selector.

The request should include:

- operation: e.g. `matvec`, `matvec_batch`, `self_attn_qkv`, `mlp_gate_up`, `cached_prefix_attention`;
- phase: `prefill`, `decode`, `component`, `materialize`, or `smoke`;
- format ID: `SQ8_0`, `AQ4_0`, or unquantized;
- model architecture: e.g. `qwen3`, `qwen3.5`;
- GPU architecture: e.g. `RDNA4`, `RDNA2`, `Ampere`;
- GPU name: e.g. `Radeon AI PRO R9700`, `A100_80GB`;
- implementation priority.

Selection rules:

- Exact GPU-name implementation beats GPU-architecture default.
- GPU-architecture implementation beats generic implementation.
- Format-specific implementation beats generic implementation when other specificity is equal.
- Model-specific implementation beats model-agnostic implementation when other specificity is equal.
- Explicit priority resolves remaining ties.

First real connections:

1. cached-prefix attention executor resolution:
   - RDNA4 + SQ8_0/FP8 KV + prefill -> `cached_prefix_rdna4_fp8_auto`;
   - fallback -> existing chunked/F32 path.
2. SQ8_0 projection matvec resolution:
   - RDNA4 + SQ8_0 + batch prefill -> direct SQ8_0 batch/pair/triple kernel;
   - CPU or unsupported GPU -> staging/materialized fallback with explicit telemetry.
3. future C++ implementation registry:
   - C++ kernel families should expose implementation IDs that Rust can select or report.

## C++ Runtime Organization

Short term:

- Keep `runtime/src/ullm_runtime.cpp` as the translation unit entrypoint.
- Keep current `.inc` split while adding clearer kernel-family boundaries.
- Add descriptors in C++ or Rust-side metadata before splitting into multiple `.cpp` files.

Medium term:

```text
runtime/src/
  ullm_runtime.cpp
  ullm_runtime_api_*.inc
  kernels/
    sq8_0/
      sq8_0_matvec_rdna4.inc
      sq8_0_matvec_cpu.inc
    aq4_0/
      aq4_0_matvec_rdna4.inc
    attention/
      cached_prefix_rdna4_fp8.inc
      cached_prefix_generic.inc
```

Do not introduce model/GPU `if` chains inside hot API wrappers. Implementation choice should happen through descriptors or a registry-style resolver.

## Milestones

### M0: Naming Foundation

Status: done.

Deliverables:

- `format_id.rs`;
- `tools/ullm_format_ids.py`;
- `docs/specs/format-ids-v0.1.md`;
- SQ artifact builder emits `SQ8_0`.

### M1: SQ8_0 Plan And Spec Cleanup

Status: this plan starts it.

Deliverables:

- this plan;
- update old SQ FP8 plan language when touched;
- keep legacy CLI names documented as compatibility names;
- make new result summaries use `SQ8_0` public format wording.

### M2: Artifact Hardening

Status: partial.

Deliverables:

- manifest checksum validation required for generated payloads;
- deterministic artifact generation test for a tiny fixture;
- policy JSON validation errors with stable messages;
- artifact rebuild command recorded in result metadata;
- sidecar artifact path and future package-integrated path documented separately.

Verification:

```text
python3 -m unittest tests.test_build_sq_fp8_artifact_policy tests.test_ullm_format_ids
python3 -m py_compile tools/build-sq-fp8-w8a16-artifact.py tools/ullm_format_ids.py
```

### M3: Library-Side Resident SQ8_0 Loader

Status: done for the current v0.1 resident loader boundary.

Deliverables:

- move resident SQ FP8 payload/scale buffer loading out of CLI-only `main_parts`;
- expose a reusable SQ8_0 overlay loader for Qwen3 package runtime;
- preserve direct matvec storage references without requiring F32 materialization;
- keep telemetry fields for selected tensors, passthrough tensors, scale layout, and execution mode.

Verification:

```text
cargo test -p ullm-engine sq_runtime -- --test-threads=1
cargo check -p ullm-engine
```

### M4: Native SQ8_0 Projection Execution

Status: partial.

Done:

- single, batch, pair, and triple SQ FP8 matvec APIs exist.
- mixed request-state rows report the projection boundary counters and the selected SQ8_0 projection
  implementation IDs resolved through `backend_dispatch`.
- full mixed request-state smoke can run supported SQ8_0 tensors through the direct resident
  dequant matvec path without whole-tensor F32 materialization.
- selected-layer `sq-fp8-token-ids-model-loop-smoke` rows now report SQ8_0 projection telemetry
  (execution mode, boundary, implementation IDs, and per-kernel counters), including the
  fallback case where projection is materialized to F32.
- full-throughput/benchmark summary rows now reject `materialized_f32_fallback` unless the run is
  explicitly marked as fallback (CLI flag/report marker) or it is the selected-layer diagnostic
  path (`sq-fp8-token-ids-model-loop-smoke`).
- benchmark Markdown summaries also exclude unmarked materialized SQ8_0 fallback rows from the
  default success table and expose `SQ mode` as a comparison column.

Remaining:

- keep coverage for remaining throughput rows where projection telemetry is still pending;

Verification:

```text
cargo test -p ullm-runtime-sys cpu_sq_fp8_matvec -- --test-threads=1
cargo test -p ullm-runtime-sys first_hip_sq_fp8_matvec -- --test-threads=1
cargo check -p ullm-engine
```

For required HIP validation:

```text
ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_KERNEL=1 cargo test -p ullm-runtime-sys first_hip_sq_fp8_matvec_f32_computes_expected_values_when_available -- --test-threads=1
ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_BATCH_KERNEL=1 cargo test -p ullm-runtime-sys first_hip_sq_fp8_matvec_batch_f32_computes_expected_values_when_available -- --test-threads=1
ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_PAIR_KERNEL=1 cargo test -p ullm-runtime-sys first_hip_sq_fp8_matvec_pair_f32_computes_expected_values_when_available -- --test-threads=1
ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_TRIPLE_KERNEL=1 cargo test -p ullm-runtime-sys first_hip_sq_fp8_matvec_triple_f32_computes_expected_values_when_available -- --test-threads=1
```

### M5: Model-Loop And Batch Integration

Status: partial.

Deliverables:

- selected-layer SQ8_0 model-loop bridge for len4/case_a/case_b prompt bundle;
- full mixed request-state path with SQ8_0 overlay metadata;
- batch throughput rows with artifact load excluded/included fields explicitly separated;
- final logits guard and behavioral prompt-suite result recorded as regression diagnostics.

Minimum result fields:

- `format_id = SQ8_0`;
- `implementation_id`;
- `sq_artifact`;
- `sq_fp8_tensor_count`;
- `sq_passthrough_tensor_count`;
- `sq_execution_mode`;
- projection boundary counters;
- artifact load/materialization wall time:
  - `metrics.artifact_load_wall_time_seconds`;
  - `metrics.artifact_materialization_wall_time_seconds`;
  - `metrics.load_excluded_total_wall_time_seconds`;
  - `metrics.load_included_total_wall_time_seconds`;
- prefill/decode/end-to-end total tok/s;
- prompt-suite regression status.

Done:

- `--parse ullm-model-loop-throughput` preserves explicit artifact-load, optional
  materialization, load-excluded total, and load-included total wall-time fields without changing
  the legacy `layer_load_wall_time_seconds` / `outer_wall_time_seconds` fields.
- `build-sq-candidate-runtime-row.py` now preserves prompt-suite regression status and the
  prompt-suite guard metrics (`acceptance_mode`, strict/behavioral pass flags, compared case count,
  generated token/text match counts, and top-logit diff summaries) in candidate runtime rows.
- `run-external-benchmark.py` can attach an existing prompt-suite guard bundle to
  `inference-benchmark-result-v0.1` rows via `--prompt-guard-bundle-json`, preserving the same
  regression status and guard metric summary beside throughput rows.
- First guard-attached full mixed SQ8_0 row saved at
  `benchmarks/results/2026-07-09/package-batch-throughput/phase-m5-sq8-guard-attached-full-mixed-v1/results.schema.jsonl`.
  It records `format_id=SQ8_0`, direct `single+triple` projection execution, artifact load vs
  measured-total timing, and behavioral prompt-suite status `passed`.
- An AQ4-derived self-attention stack real-batch diagnostic was run with the layer3 full-projection
  SQ8_0 sidecar overlay. The scheduler path reports `batching_mode=real`,
  `prefill_real_batch=true`, `decode_real_batch=true`, and request parallelism `4`, but the SQ8_0
  overlay still enters `sq_execution_mode=materialized_f32_fallback` with no direct SQ8_0 projection
  counters. This proves the existing stack scheduler can carry SQ overlay metadata through a real
  batch-shaped run, but it is not yet a direct resident SQ8_0 stack execution path.
- A resident stack diagnostic command now exists:
  `sq-fp8-package-self-attn-stack-batch-smoke`. For the layer3 Qwen3.5 SQ8_0 artifact it reports
  `sq_execution_mode=direct_fp8_dequant_matvec` and
  `sq_projection_boundary=single+triple`, proving the mixed request-state stack path can avoid
  materialized F32 SQ8_0 weights. The first saved row is
  `benchmarks/results/2026-07-09/sq8-stack-resident-diagnostic/results.jsonl`.
  It is still `batching_mode=grouped` with `prefill_real_batch=false`,
  `decode_real_batch=false`, and `sq_fp8_batch_matvec_count=0/21`, so it is a resident
  stack-connectivity diagnostic rather than the final full-package real-batch row.
- The next resident stack diagnostic row is
  `benchmarks/results/2026-07-09/sq8-stack-resident-qkv-batch/results.jsonl`. It reports
  `batching_mode=real`, `prefill_real_batch=true`, `decode_real_batch=true`,
  `sq_projection_boundary=single+batch`, and `sq_fp8_batch_matvec_count=9/21`. This proves q/k/v
  projections use direct SQ8_0 batch matvec in both prefill and decode. The remaining o/gate/up/down
  projections still use direct single boundaries, so it is still a selected-layer resident
  diagnostic rather than the final all-projection/full-serving row.
- The current resident stack diagnostic row is
  `benchmarks/results/2026-07-09/sq8-stack-resident-all-batch/results.jsonl`. It reports
  `batching_mode=real`, `prefill_real_batch=true`, `decode_real_batch=true`,
  `sq_projection_boundary=batch`, `sq_fp8_single_matvec_count=0`, and
  `sq_fp8_batch_matvec_count=21/21`. This proves the selected layer3 self-attention q/k/v/o and
  MLP gate/up/down projections all use direct SQ8_0 batch matvec boundaries. It still uses
  diagnostic host staging and a selected-layer artifact, so the remaining comparison blocker is a
  full-package or server-style row with the same execution mode.

### M6: Dispatch Integration

Status: partial.

Done:

- cached-prefix attention smoke uses `backend_dispatch` when the CLI executor is not explicitly
  overridden.
- SQ8_0 projection execution records operation-level implementation IDs for single, batch, pair, and
  triple matvec boundaries.
- `PackageAq4ResidentMatvec` now carries SQ8_0 projection dispatch decisions to the single, batch,
  pair, and triple matvec execution boundaries before calling the existing runtime kernels.
- SQ8_0 single/batch/pair/triple direct-matvec entry points now validate that the selected descriptor
  family is `Direct` before calling the kernel; unresolved/non-direct selections return an explicit
  error instead of entering the runtime kernel path.
- Test-only non-direct SQ8_0 matvec descriptor fixtures now verify both dispatch selection and
  single/batch/pair/triple boundary rejection. This keeps future non-direct family additions from
  accidentally entering the current direct-only runtime path.
- runtime GPU-architecture detection maps `compute_major == 12` to `RDNA4`.
- SQ8_0 matvec projection dispatch now has active R9700-specific direct descriptor IDs
  (`sq8_0_matvec*_r9700_direct`). Dispatch-only GPU-name canonicalization maps ROCm's generic
  `AMD Radeon Graphics` report to `Radeon_AI_PRO_R9700` when the device is `gfx1201`,
  `compute_major == 12`, and the observed memory size matches the local R9700 range. Non-matching
  RDNA4 devices still resolve to `sq8_0_matvec*_rdna4_direct`.
- SQ8_0 projection matvec registry (`operation`, `phase`, and descriptor IDs) is now defined in
  `backend_dispatch.rs`, and `part_00.rs` consumes it through public APIs.
- higher-level SQ8_0 fused projection descriptor catalog entries (`self_attn_qkv`, `self_attn_o`,
  `mlp_gate_up`, `mlp_down`, `linear_attn_qkv`, `linear_attn_out`) have been added to
  `backend_dispatch.rs` as `SQ8_0_FUSED_PROJECTION_DESCRIPTOR_CATALOG` (not yet in active runtime
  selection).
- `tools/run-external-benchmark.py --parse ullm-component-prefill` now preserves cached-prefix
  dispatch metadata (`selected_implementation_id`, `executor_selection`, `dispatch_*`) and
  cached-prefix token breakdown fields in result rows.
- `tools/run-external-benchmark.py` now records SQ8_0 direct-kernel requirement environment
  variables (`ULLM_REQUIRE_HIP_SQ_FP8_MATVEC*`) in `artifacts.command`, matching the existing AQ4
  require-flag provenance.
- `tools/run-package-token-prompt-bench.py --require-hip-kernels` now also sets the SQ8_0 batch
  matvec requirement flag, so prompt-suite and future real-batch SQ8_0 runs fail closed if that
  kernel boundary is unavailable.
- `tools/summarize-benchmark-results.py` now exposes a compact `Impl` column from
  `sq_projection_implementation_ids`, `dispatch_selected_implementation_id`, or
  `selected_implementation_id`, and classifies `SQ8_0` rows as FP8 family rows.
- `backend_dispatch` now normalizes optional selector fields for model/GPU matching. It accepts
  case-insensitive punctuation-insensitive GPU names such as `Radeon_AI_PRO_R9700` vs
  `Radeon AI PRO R9700`, and implementation-side prefix selectors such as `model_arch=Qwen3*`.
  Exact normalized matches outrank prefix matches, which outrank broad `*` matches.
- Qwen3/Qwen3.5 SQ8_0 model-loop projection telemetry now passes `model_arch=Qwen3` into the
  projection dispatch request. Unknown or non-model-loop SQ projection paths still pass `None`, so
  existing default selection behavior is preserved.
- Active SQ8_0 matvec dispatch now distinguishes Generic, RDNA4, and R9700 direct descriptors while
  preserving the same current direct kernel family. Higher-level fused descriptors have R9700 naming
  support, but the active fused catalog remains Generic/RDNA4 until fused kernels are ready.
- SQ8_0 projection result rows now report both `sq_projection_implementation_ids` and
  `sq_projection_kernel_families`, so future non-direct/fused C++ kernel families can be separated
  from the current `direct` family without parsing implementation IDs.
- A layer0 Qwen3-14B-FP8 SQ8_0 mixed-request-state smoke on device index `2` now reports
  `sq_projection_implementation_ids=single=sq8_0_matvec_r9700_direct,triple=sq8_0_matvec_triple_r9700_direct`
  while preserving the human-facing runtime device name `AMD Radeon Graphics`.
- Mixed-request-state multi-request rows now report request grouping separately from real batching:
  `batching_mode=grouped`, `prefill_request_grouped=true`, and `decode_request_grouped=true`, while
  keeping `prefill_real_batch=false` / `decode_real_batch=false` until the layer path actually uses
  batched projection kernels. A layer0 `len:4x2` SQ8_0 smoke confirms grouped execution with
  `sq_fp8_batch_matvec_count=0`.
- Mixed-request-state reports now also carry phase-local SQ8_0 batch projection counters:
  `prefill_sq_fp8_batch_matvec_count`, `decode_sq_fp8_batch_matvec_count`, and
  `mixed_request_state_real_batch_projection_used`. Future real-batch promotion is therefore tied
  to the phase that actually used a batch projection.
- A first local SQ8_0 real-batch projection smoke exists:
  `sq-fp8-package-self-attn-layer-batch-smoke`. The R9700 layer3 Qwen3.5 run with a partial q/k/v
  SQ8_0 artifact reports `real_batch=true`, `sq_projection_boundary=batch`,
  `sq_fp8_batch_matvec_count=6`, and `sq_fp8_expected_all_batch_matvec_count=14`, proving the
  SQ8_0 batch matvec path is callable inside the self-attention layer batch smoke while still
  recording that the artifact is a partial overlay.
- A full layer3 projection SQ8_0 artifact for Qwen3.5 self-attention plus MLP now passes the same
  batch smoke with `real_batch=true`, `sq_projection_boundary=batch`,
  `sq_projection_implementation_ids=batch=sq8_0_matvec_batch_r9700_direct`,
  `sq_fp8_batch_matvec_count=14`, and `sq_fp8_expected_all_batch_matvec_count=14`. This proves the
  local layer-level batch smoke can run every supported projection in that layer through the direct
  SQ8_0 batch matvec boundary. It is still a selected-layer proof, not a full-package serving row.
- The same full layer3 projection smoke is now saved as an `inference-benchmark-result-v0.1`
  component row at `benchmarks/results/2026-07-09/sq8-layer-batch-component/results.jsonl`. The row
  preserves `batching.mode=real`, `prefill_real_batch=true`,
  `sq_projection_implementation_ids=batch=sq8_0_matvec_batch_r9700_direct`, and
  `sq_fp8_batch_matvec_count=14/14`. This makes the layer-level proof machine-readable while still
  keeping it separate from full-package serving throughput.

Remaining:

- add actual runtime C++ projection family switching once non-direct implementations exist;
- keep runtime dispatch switched off for higher-level fused projection kernels; only the catalog is
  published in `backend_dispatch.rs`;
- keep selected implementation IDs in all result rows that represent dispatch-selected execution.

Example intent:

```text
format_id == SQ8_0 && gpu_arch == RDNA4 && operation == self_attn_qkv
  -> sq8_0_self_attn_qkv_rdna4_v0

gpu_arch == RDNA4 && gpu_name == Radeon_AI_PRO_R9700 && operation == cached_prefix_attention
  -> cached_prefix_rdna4_fp8_auto
```

### M7: C++ Kernel Descriptor Split

Status: partial.

Deliverables:

- define implementation descriptor naming convention;
- done: split SQ8_0 API wrapper group (`ullm_runtime_sq_fp8_matvec_f32`, `*_batch_*`, `*_pair_*`,
  `*_triple_*`) into `runtime/src/ullm_runtime_api_sq8_0.inc`;
- done: split SQ8_0 matvec runtime helper group into
  `runtime/src/kernels/sq8_0/sq8_0_matvec_runtime.inc`;
- done: keep single C++ translation unit while kernel boundaries are represented in `.inc` includes;
- done: make build.rs track new include files;
- later split `.cpp` files only after compile/link behavior is stable.

Now:

- Done: `sq_fp8_matvec_kernel_source()` is fully moved to
  `runtime/src/kernels/sq8_0/sq8_0_matvec_hiprtc.inc` and still consumed by the same
  compile entry points.
- Done: descriptor naming template and descriptor IDs are now centralized as a registry API in
  `backend_dispatch.rs` (`sq8_0_<operation>_<target>_<family>`), including single/batch/pair/triple
  matvec entries and RDNA4/generic targets.
- Done: higher-level SQ8_0 fused projection descriptor catalog entries were added in
  `backend_dispatch.rs`, and descriptor naming coverage tests now include fused entries.
- Done: `backend_dispatch` now exposes `sq8_0_projection_descriptor_family()` and the Rust matvec
  execution path now carries the selected family metadata.
- Done: SQ8_0 stdout rows and external benchmark rows now preserve
  `sq_projection_kernel_families` next to `sq_projection_implementation_ids`.
- Done: test-only non-direct SQ8_0 matvec descriptors exercise `family=None` selection and the
  Rust-side direct-family guard before any runtime C++ kernel is called.
- Done: SQ8_0 matvec runtime helper group is moved to
  `runtime/src/kernels/sq8_0/sq8_0_matvec_runtime.inc`.
- Done: large runtime-sys Rust test body was split from
  `crates/ullm-runtime-sys/src/lib_parts/part_01.rs` into
  `crates/ullm-runtime-sys/src/test_parts/`; managed Rust/C++/`.inc` files are currently below
  10k lines.
- Remaining: fused kernel family implementations are still pending.

### M8: Regression Suite

Status: partial.

Regression categories:

- schema/format regression;
- artifact generation regression;
- runtime decode regression;
- direct-kernel unit regression;
- package model-loop regression;
- throughput regression;
- behavioral prompt-suite regression.

Important rule:

- Strict AQ4_0-vs-SQ8_0 token/logit equality is diagnostic, not a proof of FP8 quality and not a universal blocker.
- A failing diagnostic must say whether it indicates an implementation bug, an expected FP8 numerical difference, or an unsupported policy/layout.

### M9: SQ8_0 v0 Completion Criteria

`SQ8_0` v0 implementation is complete when:

- public format metadata is consistently emitted as `SQ8_0`;
- legacy aliases remain readable;
- a reproducible SQ8_0 artifact can be generated from policy JSON;
- runtime can load SQ8_0 payload/scale resident buffers without whole-model F32 expansion;
- R9700/RDNA4 uses direct SQ8_0 projection kernels for selected supported tensors;
- result rows clearly report native vs fallback SQ8_0 execution;
- backend dispatch records which implementation was selected;
- package/model-loop regression suite passes with documented tolerances;
- docs/specs describe the implementation as adopted SQ8_0, not as an FP8 quality experiment.

### M10: vLLM + FP8 External Baseline Comparison

Status: partial. The M10 harness now has same-model uLLM and vLLM rows for the smoke and
representative workloads, plus real-batch SQ8_0 diagnostics. The current uLLM rows are still
model-loop or selected-layer diagnostic measurements rather than server-style serving rows. This is
intentionally a later-phase comparison after SQ8_0 artifact loading, runtime dispatch, and
implementation-valid model-loop rows are in place.

Purpose:

- Compare uLLM `SQ8_0` against an established FP8 serving/runtime baseline.
- Keep this as a performance and systems comparison, not a proof that FP8 quantization is acceptable.
- Use `vLLM + Qwen3-14B-FP8` as the primary external baseline because it is the closest existing
  FP8 runtime target already represented in local plans and benchmark records.

Prerequisites:

- M5 model-loop / batch integration is stable enough to produce repeatable uLLM rows.
- M6 dispatch records the selected uLLM implementation ID.
- uLLM rows clearly distinguish native SQ8_0 direct execution from fallback execution.
- `tools/run-external-benchmark.py` can emit comparable `inference-benchmark-result-v0.1` rows for
  both uLLM and vLLM.

Baseline target:

```text
engine: vLLM
model: Qwen/Qwen3-14B-FP8
local model path: ~/datapool/ai_models/safetensors/Qwen/Qwen3-14B-FP8/
primary GPU: R9700/RDNA4
environment: build/envs/vllm-rocm-nightly or the current working vLLM ROCm env
```

Current local baseline state:

- `Qwen/Qwen3-14B-FP8` exists at the planned local model path.
- `build/envs/vllm-rocm-nightly` is the current known-good vLLM ROCm environment for this target.
- Existing 2026-06-30 rows show vLLM can run the smoke and representative workloads on R9700 when
  the GPU is selected with `ROCR_VISIBLE_DEVICES=1`.
- `HIP_VISIBLE_DEVICES=1` is not sufficient for this comparison on the current host because AITER
  can still observe a V620/gfx1030 path first; this failure mode should remain recorded as a raw
  failed or unsupported row when it is intentionally probed.
- `tools/run-external-benchmark.py --parse vllm-throughput` already preserves the comparable vLLM
  metrics, and its failure classifier now maps common ROCm no-binary / invalid-device-function
  messages to `unsupported` rows.
- uLLM now has a guard-attached SQ8_0 smoke-shape row at
  `benchmarks/results/2026-07-09/sq8-vllm-fp8-comparison/results.jsonl` for
  `prompt_tokens=16`, `generated_tokens=8`, and `concurrent_requests=1`. This row uses
  Qwen3.5-9B, so it fixes the uLLM SQ8_0 measurement path but is not yet a same-model comparison
  against `Qwen3-14B-FP8`.
- The same JSONL now also contains a successful `vLLM + Qwen3-14B-FP8` R9700 smoke row for the
  same `prompt_tokens=16`, `generated_tokens=8`, `concurrent_requests=1` shape. This proves the
  local vLLM FP8 smoke baseline can run, while preserving the model mismatch limitation.
- The same JSONL now also contains successful matching-shape vLLM FP8 rows for the uLLM real-batch
  no-final-logits diagnostics:
  - `prompt_tokens=16x2`, `generated_tokens=8x2`, `concurrent_requests=2`. It records prefill
    `34.41438620647337` tok/s, decode `17.21` tok/s, total `51.62` tok/s, and consumed VRAM
    `21007855616` bytes.
  - `prompt_tokens=16x4`, `generated_tokens=8x4`, `concurrent_requests=4`. It records prefill
    `135.04146895989985` tok/s, decode `67.52` tok/s, total `202.56` tok/s, and consumed VRAM
    `30121553920` bytes.
  - `prompt_tokens=16x8`, `generated_tokens=8x8`, `concurrent_requests=8`. It records prefill
    `236.01404374447745` tok/s, decode `118.01` tok/s, total `354.02` tok/s, and consumed VRAM
    `30121566208` bytes.
- It also contains a successful vLLM representative row for `prompt_tokens=512`,
  `generated_tokens=128`, and `concurrent_requests=1` with decode `22.54 tok/s` and consumed VRAM
  `30837428224` bytes.
- It now contains config-aligned same-model uLLM rows for the same local `Qwen3-14B-FP8` target:
  - smoke `pp16/tg8/b1`, `rotary_dim=128`, `rope_base=1000000`, decode `3.057004 tok/s`,
    consumed VRAM `13763940352` bytes;
  - representative `pp512/tg128/b1`, `rotary_dim=128`, `rope_base=1000000`, decode
    `2.774043 tok/s`, consumed VRAM `14242410496` bytes.
- Refreshed config-aligned same-model uLLM smoke and representative rows were added after R9700
  projection dispatch descriptors were enabled. They record
  `sq_projection_implementation_ids=single=sq8_0_matvec_r9700_direct,triple=sq8_0_matvec_triple_r9700_direct`,
  with smoke decode `2.702039 tok/s` / consumed VRAM `13763952640` bytes and representative decode
  `2.858786 tok/s` / consumed VRAM `14242406400` bytes.
- A short full 40-layer mixed-request-state real-batch row now exists at
  `benchmarks/results/2026-07-09/sq8-qwen3-14b-full-mixed-real-batch-smoke/results.jsonl`. It uses
  the Qwen3-14B-FP8 thin package plus full SQ8_0 sidecar, `prompt_tokens=1x2`,
  `generated_tokens=1x2`, `rotary_dim=128`, and `rope_base=1000000`. It records
  `batching_mode=real`, `prefill_real_batch=true`, `decode_real_batch=true`,
  `sq_projection_boundary=batch`, and `sq_fp8_batch_matvec_count=560/560`.
- Diagnostic host-staging telemetry is now emitted for SQ8_0 mixed request-state rows and preserved
  by the external benchmark parser. The short layer3 telemetry smoke at
  `benchmarks/results/2026-07-09/sq8-host-staging-telemetry-smoke/results.jsonl` records
  `sq_diagnostic_host_staging_read_count=39`,
  `sq_diagnostic_host_staging_write_count=48`,
  `sq_diagnostic_host_staging_read_bytes=1327104`, and
  `sq_diagnostic_host_staging_write_bytes=1130496`.
- The first host-staging reduction row at
  `benchmarks/results/2026-07-09/sq8-host-staging-reduced-smoke/results.jsonl` keeps
  `sq_fp8_batch_matvec_count=21/21` and reduces the same selected-layer shape to
  `sq_diagnostic_host_staging_read_count=33`,
  `sq_diagnostic_host_staging_write_count=42`,
  `sq_diagnostic_host_staging_read_bytes=1228800`, and
  `sq_diagnostic_host_staging_write_bytes=1032192`.
- The MLP device-side reduction row at
  `benchmarks/results/2026-07-09/sq8-host-staging-mlp-residual-device-smoke/results.jsonl` keeps
  the same selected-layer `sq_fp8_batch_matvec_count=21/21` and further reduces the shape to
  `sq_diagnostic_host_staging_read_count=24`,
  `sq_diagnostic_host_staging_write_count=39`,
  `sq_diagnostic_host_staging_read_bytes=540672`, and
  `sq_diagnostic_host_staging_write_bytes=737280` by keeping MLP activation and residual add on
  batch device buffers.
- The D2D pack row at
  `benchmarks/results/2026-07-09/sq8-host-staging-d2d-pack-smoke/results.jsonl` keeps the same
  selected-layer `sq_fp8_batch_matvec_count=21/21` and reduces the host-staging diagnostic to
  `sq_diagnostic_host_staging_read_count=0`,
  `sq_diagnostic_host_staging_write_count=9`,
  `sq_diagnostic_host_staging_read_bytes=0`, and
  `sq_diagnostic_host_staging_write_bytes=196608` by using runtime buffer-to-buffer copies for
  batch packing and per-request unpacking.
- A full 40-layer D2D-pack smoke row at
  `benchmarks/results/2026-07-09/sq8-qwen3-14b-full-mixed-real-batch-d2d-pack-smoke/results.jsonl`
  keeps `sq_fp8_batch_matvec_count=560/560` for `Qwen3-14B-FP8` with real prefill/decode batching.
  It records `sq_diagnostic_host_staging_read_count=156`,
  `sq_diagnostic_host_staging_write_count=240`,
  `sq_diagnostic_host_staging_read_bytes=3194880`, and
  `sq_diagnostic_host_staging_write_bytes=6553600`. This shows the remaining full-stack staging is
  dominated by layer-to-layer residual handoff through the host-driven wrapper, not by the
  selected-layer batch pack/unpack boundary.
- A follow-up full 40-layer device-handoff smoke row at
  `benchmarks/results/2026-07-09/sq8-qwen3-14b-full-mixed-real-batch-device-handoff-smoke/results.jsonl`
  keeps `sq_fp8_batch_matvec_count=560/560` for the same `Qwen3-14B-FP8` shape and records
  `sq_diagnostic_host_staging_read_count=0`,
  `sq_diagnostic_host_staging_write_count=6`,
  `sq_diagnostic_host_staging_read_bytes=0`, and
  `sq_diagnostic_host_staging_write_bytes=163840`. This removes the measured layer-to-layer
  residual read staging from the mixed request-state stack while preserving real prefill/decode
  batching. The remaining counted writes are initial host-side residual inputs for the smoke path,
  so this is the new full-stack diagnostic baseline before the later vLLM+FP8 comparison rows.
- Mixed request-state CLI rows now support `TOP_K=0` to skip the final lm_head guard and exclude
  final logits from measured total latency. The serving-nearer full 40-layer rows at
  `benchmarks/results/2026-07-09/sq8-qwen3-14b-full-mixed-real-batch-no-final-logits-smoke/results.jsonl`
  cover `concurrent_requests=2`, `4`, and `8` with `rotary_dim=128` and `rope_base=1000000`. The b2
  row records `final_logits_in_total=false`, `final_lm_head_guard=false`,
  `sq_fp8_batch_matvec_count=6720/6720`, `sq_diagnostic_host_staging_read_count=0`,
  `sq_diagnostic_host_staging_write_count=72`, `prefill_total_input_tps=15.417194`,
  `decode_total_generated_tps=15.709506`, and `end_to_end_total_tps=15.513415`. The b4 row keeps
  `sq_fp8_batch_matvec_count=6720/6720` and records `sq_diagnostic_host_staging_read_count=0`,
  `sq_diagnostic_host_staging_write_count=120`, `prefill_total_input_tps=16.220953`,
  `decode_total_generated_tps=16.766274`, and `end_to_end_total_tps=16.398742`. The b8 row keeps
  `sq_fp8_batch_matvec_count=6720/6720` and records `sq_diagnostic_host_staging_read_count=0`,
  `sq_diagnostic_host_staging_write_count=216`, `prefill_total_input_tps=16.477829`,
  `decode_total_generated_tps=16.747149`, and `end_to_end_total_tps=16.566635`. These are still
  model-loop rows rather than server parity, but they remove the earlier final-logits latency
  caveat from the real-batch SQ8_0 diagnostic class and advance the batch grid through b8.
- The config-aligned uLLM rows now have a self-behavioral prompt-suite smoke guard attached:
  `benchmarks/results/2026-07-09/sq8-vllm-fp8-comparison/qwen3-14b-sq8-prompt-suite-smoke-rope128-theta1e6/guard-self-behavioral/guard-bundle-summary.json`.
  It records `passed=true`, `acceptance_mode=behavioral`, `strict_passed=true`, and
  `behavioral_passed=true`. Because the same summary is used as both reference and candidate and the
  suite has `output_health=false`, this is a guard plumbing smoke, not an external quality guard.
- Earlier uLLM Qwen3-14B rows with `rotary_dim=32` and `rope_base=10000000` are retained as
  preliminary connectivity rows only; the config-aligned rows should be used for M10 same-model
  discussion.
- A 2026-07-10 refresh at
  `benchmarks/results/2026-07-10/sq8-qwen3-14b-normalized-kernel-family-refresh/results.jsonl`
  re-ran the b2/b4/b8 uLLM rows after `sq_projection_kernel_families` telemetry was added. The
  refreshed rows record `sq_projection_kernel_families=batch=direct`,
  `sq_fp8_batch_matvec_count=6720/6720`, and `final_logits_in_total=false`.

Same-model readiness audit:

- The comparison directory now includes
  `benchmarks/results/2026-07-09/sq8-vllm-fp8-comparison/same-model-readiness.md`.
- A local BF16-only thin uLLM package for `Qwen3-14B-FP8` now exists at
  `/tmp/ullm-qwen3-14b-fp8-bf16-thin.ullm.d`. It is paired with a SQ8_0 sidecar artifact rather
  than native SQ tensors inside `.ullm.d`.
- `Qwen3-14B-FP8` source tensor naming uses `model.embed_tokens.weight`, `model.norm.weight`, and
  `model.layers.*`. The current uLLM token-id model-loop runtime uses
  `model.language_model.embed_tokens.weight`, `model.language_model.norm.weight`, and
  `model.language_model.layers.*` constants. Runtime/package/SQ selector fallback now accepts both
  namespaces for supported Qwen3 tensor names, while preserving exact-name lookup priority for
  existing Qwen3.5 packages.
- `tools/build-sq-fp8-w8a16-artifact.py --dry-run` can inspect the local `Qwen3-14B-FP8` directory
  and selects `281` FP8 tensors, with `442` passthrough tensors and a compact resident byte estimate
  of `15557220864`. This confirms SQ8_0 artifact metadata generation is plausible, but it does not
  prove the current runtime can execute the package.
- `ullm-quant --dry-run` can inspect the same source directory without loading full payloads.
  After excluding Qwen FP8 auxiliary `*.weight_scale_inv` tensors from AQ quantization, it sees
  `723` total tensors, `0` supported AQ source tensors, and `723` passthrough tensors. This is the
  correct classification for the local FP8 artifact: the projection matrices are already
  `F8_E4M3`, while current AQ direct package conversion only quantizes BF16/F16/F32 source
  matrices.
- `ullm-quant` now has passthrough filters for direct package import. Using those filters, the local
  `Qwen3-14B-FP8` source can produce a BF16-only thin package shell with `163` passthrough tensors,
  `0` quantized tensors, and no copied `F8_E4M3` or `*.weight_scale_inv` tensors. The observed
  temporary package was `/tmp/ullm-qwen3-14b-fp8-bf16-thin.ullm.d` at about `2.9G`.
- `ullm-engine package-layer-kind-inventory-smoke` now detects the thin package's `40` Qwen3
  self-attention layers from passthrough `q_norm/k_norm` names.
- A layer0 SQ8_0 sidecar artifact with `7` FP8 tensors and `716` passthrough metadata entries was
  built from the same Qwen3-14B-FP8 source. Combined with the thin package, it passes
  `sq-fp8-token-ids-logits-smoke` for layer `0` with `verified=true`.
- A full 40-layer SQ8_0 sidecar artifact with `281` FP8 tensors and `442` passthrough metadata
  entries was built at `/tmp/ullm-qwen3-14b-fp8-full-sq8-artifact`.
- The tensor namespace issue is closed for runtime lookup, and the same-model connectivity path is
  proven through full 40-layer `manifest-all` uLLM rows. Building an AQ4 package directly from
  `Qwen3-14B-FP8` remains the wrong route.

Same-model prerequisites:

1. Choose the Qwen3 tensor namespace strategy: Done. Runtime lookup now accepts both `model.*` and
   `model.language_model.*` for Qwen3 layers, embeddings, and final norm.
2. Add an FP8/SQ8_0 package import path for `Qwen3-14B-FP8`: partial. The current minimum route is a
   BF16-only thin `.ullm.d` package plus SQ8_0 sidecar artifact overlay; native FP8 tensors are not
   yet integrated into `.ullm.d`.
3. Build or import the matching `SQ8_0` artifact and verify the selected FP8 tensor count against
   the imported package tensors: done for sidecar v0.1. The full artifact reports `281` FP8 tensors.
4. Run a 40-layer `manifest-all` uLLM smoke row with `prompt_tokens=16`, `generated_tokens=8`, and
   `concurrent_requests=1`: done for the config-aligned Qwen3-14B-FP8 row, and refreshed once after
   R9700 projection dispatch descriptor selection was enabled.
5. Attach the prompt guard bundle or an equivalent behavioral guard before final comparison against vLLM
   rows: partial. A self-behavioral prompt-suite smoke is attached to the config-aligned rows. A
   non-self behavioral guard or output-health-evaluated prompt suite is still needed before treating
   this as final quality-regression evidence.
6. Add the representative `prompt_tokens=512`, `generated_tokens=128`, `concurrent_requests=1`
   same-model row: done for the config-aligned Qwen3-14B-FP8 row, and refreshed once after R9700
   projection dispatch descriptor selection was enabled.
7. Add a real-batch or server-style uLLM measurement path before promoting this to a final serving
   throughput conclusion: partial. Full 40-layer mixed-request-state real-batch rows now prove the
   direct batch model-loop path, and the `TOP_K=0` rows remove final logits from total latency.
   Matching-shape vLLM `concurrent_requests=2`, `4`, and `8` rows now exist. The remaining gap is
   server-style uLLM measurement or explicit harness normalization.

Workload grid:

- smoke: prompt `16`, generated `8`, batch/concurrency `1`;
- representative decode: prompt `512`, generated `128`, batch/concurrency `1`;
- cached-prefix or long-prefill probe when uLLM has the matching SQ8_0 path;
- later batch grid: concurrent requests `1, 2, 4, 8` only after uLLM and vLLM rows are both stable.
  The `concurrent_requests=2`, `4`, and `8` smoke shapes are now recorded for both uLLM and vLLM,
  but still as different harness classes. The remaining `b1` same-shape real-batch/no-final uLLM
  row is optional because a separate b1 model-loop row exists with final logits included.

Metrics:

- prefill tok/s;
- decode tok/s;
- total tok/s;
- latency p50/p95 when available;
- VRAM baseline, peak, and consumed bytes;
- decode tok/s times consumed GiB;
- backend/executor name;
- model dtype, quantization, KV cache dtype, and failure reason if unsupported.

Comparison rules:

- vLLM comparison happens after uLLM SQ8_0 rows are implementation-valid; it must not block earlier
  artifact or direct-kernel work.
- If vLLM cannot run the exact FP8 model or local GPU target, record an `unsupported` or `failed`
  row rather than omitting it.
- Do not compare uLLM selected-layer rows against full vLLM serving rows as equivalent results.
  Selected-layer uLLM rows are path-connectivity diagnostics only.
- Full-package uLLM rows can be compared against vLLM serving rows when prompt/generation lengths,
  batch/concurrency, KV cache dtype, and model target are documented.
- Earlier Qwen3-14B uLLM rows are full 40-layer model-loop rows with direct SQ8_0 projection
  execution, but `prefill_real_batch=false`, `decode_real_batch=false`, and final logits are included
  in total latency. Treat them as implementation-valid model-loop comparison rows, not final serving
  parity rows.
- The Qwen3-14B mixed-request-state rows reach `prefill_real_batch=true`,
  `decode_real_batch=true`, and direct SQ8_0 batch projection coverage (`560/560` for the short
  row, `6720/6720` for the `pp16/tg8/b2`, `pp16/tg8/b4`, and `pp16/tg8/b8` no-final-logits rows). Treat them as full model-loop
  real-batch evidence, but not final serving parity because they are still CLI model-loop rows and
  do not yet match vLLM's server/throughput harness semantics. Matching-shape vLLM `pp16/tg8/b2`,
  `pp16/tg8/b4`, and `pp16/tg8/b8` rows now exist for comparison as a separate harness class.
- SQ8_0 mixed request-state rows may now report `sq_diagnostic_host_staging_*` counters. Nonzero
  values make the host-staging caveat machine-readable and should keep the row outside final serving
  parity comparisons until those copies are removed or the row is explicitly classified as
  diagnostic.
- AQ4-derived stack real-batch rows with SQ8_0 overlay are not comparable serving rows when they
  report `sq_execution_mode=materialized_f32_fallback`. They are scheduler-connectivity diagnostics
  until the stack/model-loop loader uses resident SQ8_0 matvec storage instead of materialized F32
  runtime weights.
- Mixed-request-state can group several ready requests at the same timestep, but this is now recorded
  as `batching_mode=grouped` with `*_request_grouped=true`; it must not be promoted to real-batch
  evidence unless `prefill_real_batch` / `decode_real_batch` are true and the relevant batched
  projection counters are nonzero.
- Differences in tokenizer/server overhead should be recorded. When possible, keep a raw model-loop
  uLLM row and a server-style vLLM row as separate comparison classes.

Expected outputs:

- JSONL rows in `benchmarks/results/YYYY-MM-DD/external/` or a later SQ8_0 comparison directory.
- Markdown summary with one table for successful comparable rows and one table for unsupported or
  failed rows.
- Compact SQ8_0/vLLM batch-grid tables can be regenerated from JSONL with
  `tools/summarize-sq8-vllm-batch-grid.py`, for example with `--workload-prefix pp16-tg8 --requests 2,4,8`.
  The helper also accepts `--harness-class`, so diagnostic model-loop rows and serving-throughput
  rows can be viewed or gated as separate classes.
- New external benchmark rows carry a machine-readable `harness` object. This distinguishes
  `cli_model_loop_diagnostic` uLLM rows from `serving_throughput_benchmark` vLLM rows and records
  whether the row is a serving-parity candidate without relying only on prose caveats.
- `tools/summarize-sq8-vllm-batch-grid.py --require-serving-parity` is the machine gate for final
  serving-comparison tables. It currently fails the b2/b4/b8 compact table by design because the
  selected uLLM rows are CLI model-loop diagnostics and the selected vLLM rows are serving
  throughput benchmark rows. Filtering with `--harness-class serving_throughput_benchmark` makes the
  current gate pass only for the vLLM slice, which is useful as a sanity check but not a comparative
  uLLM-vs-vLLM result. Final comparison commands should also use `--require-engines uLLM,vLLM`
  and `--require-engine-grid`, so a serving-only vLLM slice or a partially populated request-count
  grid cannot accidentally satisfy the comparison gate while no uLLM serving-parity row exists.
- M10 comparison is now defined as a same-shape normalized throughput comparison gate, not serving parity.
  The gate requires `--require-normalized-throughput-comparison` so `uLLM` (`cli_model_loop_diagnostic`)
  and `vLLM` (`serving_throughput_benchmark`) rows are validated with explicit shape-homogenizing checks.
- The refreshed 2026-07-10 uLLM rows plus the existing 2026-07-09 vLLM b2/b4/b8 rows now pass
  `--require-normalized-throughput-comparison --require-ullm-sq-batch-coverage --require-ullm-sq-kernel-families`.
  The same summary helper can add `--show-sq-details` to display `SQ boundary`, `SQ family`, and
  `SQ batch` columns in the comparison table.
- A note linking to `docs/plans/r9700-qwen3-14b-fp8-external-engine-plan-v0.1.md` and the exact
  vLLM environment used.
- The vLLM row should be produced through the derived command template in
  `docs/plans/r9700-qwen3-14b-fp8-external-engine-plan-v0.1.md` so memory and failure semantics
  match the uLLM rows.

## Immediate Work Queue

1. Add `sq_runtime.rs` or equivalent library boundary. Done.
   - Move SQ resident payload/scale loading and `SqFp8` storage refs out of CLI-local code.
2. Add an SQ8_0 implementation result label. Done.
   - Keep CLI command compatibility, but result rows should report public `format_id = SQ8_0`.
3. Wire `backend_dispatch` into cached-prefix executor resolution. Done.
   - This is the smallest real dispatch connection and exercises GPU arch/name selection.
4. Wire `backend_dispatch` into SQ8_0 projection execution. Partial.
   - Operation-level reporting is done.
   - Dispatch decisions now reach the direct matvec execution boundary.
   - Selected-layer model-loop rows now distinguish direct-path runs from
     `materialized_f32_fallback`.
   - Accidental `materialized_f32_fallback` throughput rows are now rejected unless explicitly
     allowed (or explicitly saved as selected-layer diagnostics).
   - Benchmark summary tables now exclude unmarked materialized SQ8_0 fallback rows by default and
     show `SQ mode` for retained rows.
   - Backend selector matching now tolerates GPU/model naming variations and supports
     implementation-side prefix selectors for future model-family entries.
   - Qwen3/Qwen3.5 model-loop SQ8_0 rows now feed `model_arch=Qwen3` into projection dispatch
     selection without changing stdout/JSON schema.
   - SQ8_0 matvec projection dispatch now has active R9700-specific direct descriptor IDs and a
     conservative dispatch-only canonical name for the local `gfx1201` R9700 device.
   - SQ8_0 projection rows now preserve `sq_projection_kernel_families`, which is currently
     `direct` for executed matvec boundaries and keeps future C++ kernel family switching
     machine-readable.
   - A short layer0 SQ8_0 mixed-request-state smoke confirms the local R9700 path selects
     `*_r9700_direct` descriptor IDs.
   - Non-direct descriptor fixtures now prove the current execution boundary fails closed before
     calling unimplemented C++ families.
   - C++ kernel-family switching remains a follow-up.
   - Selected-layer model-loop rows now carry projection boundary and counter telemetry for
     `sq-fp8-token-ids-model-loop-smoke`.
5. Generate one fresh SQ8_0 artifact from policy JSON and run. Done.
   - materialize smoke;
   - selected-layer model-loop smoke;
   - mixed request-state throughput row.
6. Update old docs opportunistically.
   - Do not rewrite all historical T2 logs.
   - New plans/results should use `SQ8_0` terminology.
7. After the first implementation-valid full-package SQ8_0 rows exist, run the M10 vLLM + FP8
   comparison grid and save unsupported/failure rows explicitly when vLLM cannot match the target.
   - Harness support for vLLM throughput parsing and common ROCm unsupported failure messages is
     prepared.
   - The config-aligned smoke and representative rows have been refreshed after R9700 descriptor
     selection and now record `*_r9700_direct`.
   - Multi-request mixed-state rows are now classified as grouped, not real-batch, so they cannot
     accidentally satisfy the final serving-comparison gate.
   - A local `sq-fp8-package-self-attn-layer-batch-smoke` now exercises the SQ8_0 batch matvec
     runtime path with `real_batch=true`. The latest full layer3 projection run reaches
     `sq_fp8_batch_matvec_count=14/14`, so the next comparison blocker is no longer layer-local
     batch projection coverage; it is connecting the same direct batch boundary to full-package
     real-batch or server-style rows.
   - Reusing the existing AQ4 self-attention stack real-batch runner with an SQ8_0 overlay is not
     sufficient for this blocker: the current `Qwen3PackageModelRuntime::load_with_sq_overlay`
     path materializes SQ8_0 projection tensors into F32 runtime weights. The next implementation
     step is to move the stack/model-loop layer runtime onto a resident projection abstraction such
     as `PackageAq4ResidentMatvec` or a new resident stack-batch layer that can call
     `matvec_batch` for q/k/v/o/gate/up/down.
   - The new `sq-fp8-package-self-attn-stack-batch-smoke` closes the materialized-F32 side of this
     blocker for a stack-shaped diagnostic by using the resident mixed request-state path. The
     latest row now reaches `batching_mode=real`, `sq_projection_boundary=batch`, and
     `sq_fp8_batch_matvec_count=21/21` for the selected layer3 self-attention projections. The
     remaining blocker is moving from selected-layer diagnostics with host staging to
     full-package/server rows.
   - Full 40-layer Qwen3-14B-FP8 mixed-request-state rows now reach
     `batching_mode=real`, `sq_projection_boundary=batch`, and direct SQ8_0 batch projection
     coverage. The `TOP_K=0` `pp16/tg8/b2`, `pp16/tg8/b4`, and `pp16/tg8/b8` rows record
     `final_logits_in_total=false`, `sq_fp8_batch_matvec_count=6720/6720`, and host staging read `0`,
     giving serving-nearer model-loop diagnostics. The matching vLLM rows now record decode
     `17.21` tok/s and total `51.62` tok/s for b2, and decode `67.52` tok/s and total `202.56`
     tok/s for b4, and decode `118.01` tok/s and total `354.02` tok/s for b8. The remaining blocker
     for final vLLM serving comparison is adding server-style uLLM measurement or explicitly
     normalizing the harness difference.
   - Comparison scripts should now fail M10 runs unless SQ8_0 uLLM rows pass
     `--require-ullm-sq-kernel-families`, so fallback/no-kernel SQ8_0 rows do not silently leak
     into direct-projection comparison cases.
   - Same M10 gates should also start requiring
     `--require-ullm-sq-batch-coverage` so SQ8_0 rows with non-batch projection boundaries or
     incomplete batch matvec counters are blocked before mixing against serving rows.
   - The 2026-07-10 b2/b4/b8 refresh now satisfies the normalized comparison gate together with
     `--require-ullm-sq-batch-coverage` and `--require-ullm-sq-kernel-families`.
   - Host staging is now annotated by `sq_diagnostic_host_staging_*` counters in SQ8_0 mixed
     request-state rows. A first reduction moved the selected-layer layer3 shape from `39/48`
     read/write operations to `33/42` by keeping the o residual add and post-RMSNorm on batch device
     buffers. A second reduction moved the same shape to `24/39` by keeping the MLP gate/up
     SiLU-mul activation and MLP down residual add on batch device buffers. A third reduction moves
     the shape to `0/9` by adding runtime buffer-to-buffer copy and using it for batch pack/unpack
     boundaries. A subsequent full-stack device handoff removes the layer-to-layer host reads,
     moving the full 40-layer short row from `156/240` read/write operations to `0/6`. The
     `TOP_K=0` rows record `0/72` for `pp16/tg8/b2`, `0/120` for `pp16/tg8/b4`, and `0/216` for
     `pp16/tg8/b8`; the remaining counted writes are host residual inputs in this smoke path.

## Risks

| risk | impact | handling |
| --- | --- | --- |
| old SQ FP8 wording remains in historical docs | confusion about whether FP8 is adopted | keep history unchanged but add current SQ8_0 plan and specs |
| CLI-local SQ overlay logic grows further | hard to reuse in package/model-loop runtime | move resident loader into library before adding more paths |
| direct SQ8_0 matvec is correct but not fast enough | format appears slow due to kernel maturity | report implementation ID and direct/fallback mode separately |
| backend dispatch is added too broadly | hard-to-debug selection bugs | connect one operation at a time with tests |
| C++ split breaks anonymous namespace/helper ownership | build instability | keep one translation unit until descriptors are stable |
| strict AQ4 equality blocks valid FP8 behavior | false negatives | keep exact equality as regression diagnostic, use implementation correctness and behavioral suite for gating |
| vLLM comparison is run before uLLM rows are comparable | misleading performance conclusions | only compare full-package SQ8_0 rows with documented workload and execution mode |

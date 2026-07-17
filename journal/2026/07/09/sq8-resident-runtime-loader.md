# SQ8 Resident Runtime Loader

## Summary

- Added `ullm_engine::sq_runtime` as the library-side boundary for SQ8_0 resident payload and scale loading.
- Moved SQ8_0 scale-kind mapping out of CLI-local `main_parts`.
- Rewired `PackageAq4ResidentMatvec::load_with_sq_overlay` to use `load_sq8_resident_tensor` while preserving direct SQ FP8 matvec storage.
- Removed now-unused CLI-local SQ FP8 raw-buffer helper code.

## Verification

```text
cargo fmt --all --check
cargo test -p ullm-engine sq_runtime -- --test-threads=1
cargo check -p ullm-engine
git diff --check -- ':!README.md'
```

## Follow-up: SQ8_0 Fused Projection Descriptor Catalog

- Added a higher-level SQ8_0 fused projection descriptor catalog in `backend_dispatch.rs`.
- Covered planned fused projection operations:
  - `self_attn_qkv`
  - `self_attn_o`
  - `mlp_gate_up`
  - `mlp_down`
  - `linear_attn_qkv`
  - `linear_attn_out`
- Kept fused descriptors separate from the active matvec projection dispatch list so runtime
  selection remains unchanged until fused kernels exist.
- Added tests for fused descriptor naming, catalog coverage, and unresolved active selection.
- Updated the SQ8_0 plan M6/M7 to record the catalog as done while keeping fused runtime switching
  and C++ kernel-family implementation as remaining work.

Verification:

```text
cargo fmt --all --check
cargo test -p ullm-engine backend_dispatch -- --test-threads=1
cargo check -p ullm-engine
git diff --check -- ':!README.md' ':!journal/**'
```

## Follow-up: SQ8_0 Direct Projection Family Guard

- Added `sq8_0_projection_descriptor_family()` in `backend_dispatch.rs`.
- Active SQ8_0 single/batch/pair/triple matvec descriptors now resolve to the `Direct` family.
- `SqFp8ProjectionDispatch` now carries optional family metadata alongside the implementation ID.
- SQ8_0 direct matvec entry points now require a direct family before calling the runtime kernel.
- Unknown, unresolved, or future non-direct descriptor IDs will fail before entering the direct
  kernel path.
- Updated the SQ8_0 plan M6/M7 to mark the direct-family guard done while keeping actual C++
  multi-family switching as remaining work.

Verification:

```text
cargo fmt --all --check
cargo test -p ullm-engine backend_dispatch -- --test-threads=1
cargo check -p ullm-engine
target/debug/ullm-engine sq-fp8-token-ids-mixed-request-state-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d /tmp/ullm-sq8_0-kup6-gate5-down5-policy-20260709-artifact 2 1048576 manifest-all len:2x2 1 1 1024 32 10000000 0
git diff --check -- ':!README.md' ':!journal/**'
```

All commands passed. The runtime C++ build still emits the existing anonymous-namespace linkage warnings.

## Notes

- `README.md` had pre-existing uncommitted changes and was not modified.
- The next SQ8_0 step is to add explicit result metadata labels, then connect `backend_dispatch` to runtime paths.

## Follow-up: Result Labels

- Added SQ8_0 public `format_id` / `sq_format_id` fields to SQ materialize, model-loop, and mixed request-state output rows.
- Preserved legacy SQ FP8 lineage as `candidate_legacy`, `sq_candidate_legacy`, and `sq_implementation_id`.
- Updated JSON `sq_overlay` reports for logits/generate smokes with public candidate and implementation metadata.
- Updated `tools/run-external-benchmark.py` so parsed model-loop rows retain `format_id`, `sq_format_id`, and `sq_implementation_id`.

Verification:

```text
cargo fmt --all --check
cargo check -p ullm-engine
python3 -m unittest tests.test_external_benchmark_batch_parser tests.test_build_sq_fp8_artifact_policy tests.test_ullm_format_ids
git diff --check -- ':!README.md'
```

## Follow-up: Cached-Prefix Backend Dispatch

- Connected `backend_dispatch` to `runtime-cached-prefix-attn-smoke` executor selection.
- Preserved explicit CLI executor overrides as `executor_selection=cli_override`.
- For unspecified executor, dispatch now selects:
  - generic fallback: `cached_prefix_chunked`;
  - RDNA4 + FP8/SQ8_0 request: `cached_prefix_rdna4_fp8_auto`.
- Added stdout metadata for dispatch source, selected implementation ID, operation, phase, request format ID, and GPU architecture.
- Made RDNA4 FP8 auto shape-aware: rocWMMA is used only when its head/value constraints are met; otherwise auto resolves to Flash2 FP8Q.

Verification:

```text
cargo fmt --all --check
cargo check -p ullm-engine
cargo run -p ullm-engine -- runtime-cached-prefix-attn-smoke 0 4 2 1 2 1 4 4 f32
cargo test -p ullm-engine backend_dispatch -- --test-threads=1
git diff --check -- ':!README.md'
```

## Follow-up: SQ8_0 Projection Dispatch Reporting

- Added `backend_dispatch` implementation descriptors for SQ8_0 single, batch, pair, and triple
  projection matvec boundaries.
- Mixed request-state rows now include `sq_projection_implementation_ids`, for example
  `pair=sq8_0_matvec_pair_rdna4_direct`.
- This is operation-level reporting only. Kernel selection semantics are still the existing direct
  SQ FP8 matvec calls; dispatch-selected kernel switching remains a follow-up.
- Updated external benchmark row enrichment so the new string field is preserved in workload
  metadata.
- Updated `docs/plans/sq8-implementation-plan-v0.1.md` to mark M3 done, M6 partial, and keep M10
  `vLLM + FP8` comparison in the later half of the SQ8_0 plan.

## Follow-up: M10 vLLM FP8 Harness Preparation

- Confirmed the local M10 baseline assets exist:
  - `~/datapool/ai_models/safetensors/Qwen/Qwen3-14B-FP8/`
  - `build/envs/vllm-rocm-nightly`
- Existing 2026-06-30 vLLM rows show the R9700 FP8 baseline runs with `ROCR_VISIBLE_DEVICES=1`.
- Added test coverage for `parse_vllm_metrics()` using vLLM throughput JSON plus stdout.
- Extended `classify_failure()` so common ROCm no-binary and invalid-device-function errors become
  `unsupported` rows instead of generic command failures.
- Updated the SQ8_0 implementation plan M10 section with the current local baseline state.
- Added a vLLM smoke and representative command template to
  `docs/plans/r9700-qwen3-14b-fp8-external-engine-plan-v0.1.md`, using
  `tools/run-external-benchmark.py --parse vllm-throughput`.

Verification:

```text
python3 -m unittest tests.test_external_benchmark_batch_parser
```

## Follow-up: Component Prefill Dispatch Metadata Rows

- Extended `tools/run-external-benchmark.py --parse ullm-component-prefill` so cached-prefix
  component rows keep:
  - requested `prefill_executor`
  - runtime `resolved_prefill_executor`
  - `selected_implementation_id`
  - `dispatch_selected_implementation_id`
  - `executor_selection`
  - `dispatch_operation`
  - `dispatch_phase`
  - `dispatch_format_id`
  - `dispatch_gpu_arch`
- Preserved cached-prefix workload token fields:
  - `cached_prefix_tokens_per_request`
  - `new_prefill_tokens_per_request`
  - `total_context_tokens_after_prefill_per_request`
  - `cached_prefix_total_tokens`
  - `total_context_tokens_after_prefill`
- Added a parser test using `runtime-cached-prefix-attn-smoke` style key-value stdout.
- Updated the SQ8_0 implementation plan M6 done list.

Verification:

```text
python3 -m unittest tests.test_external_benchmark_batch_parser
python3 -m py_compile tools/run-external-benchmark.py tests/test_external_benchmark_batch_parser.py
git diff --check -- ':!README.md' ':!journal/**'
```

## Follow-up: Summary Implementation ID Column

- Added an `Impl` column to `tools/summarize-benchmark-results.py` output.
- The column resolves implementation IDs in this order:
  - `workload.sq_projection_implementation_ids`
  - `workload.dispatch_selected_implementation_id`
  - `workload.selected_implementation_id`
- Added summary tests for SQ projection implementation IDs and cached-prefix dispatch
  implementation IDs.
- Classified `SQ8_0` as FP8 in summary family output.
- Updated the SQ8_0 implementation plan M6 done list.

Verification:

```text
python3 -m unittest tests.test_summarize_benchmark_results tests.test_external_benchmark_batch_parser
python3 -m py_compile tools/summarize-benchmark-results.py tests/test_summarize_benchmark_results.py
```

## Follow-up: Runtime-Sys Test Split

- Split `crates/ullm-runtime-sys/src/lib_parts/part_01.rs` from a 9237-line test module into a
  9-line include wrapper plus four files under `crates/ullm-runtime-sys/src/test_parts/`.
- New test part line counts:
  - `part_00.rs`: 4256
  - `part_01.rs`: 3338
  - `part_02.rs`: 916
  - `part_03.rs`: 721
- Current managed Rust/C++/`.inc` files are below 10k lines.
- Updated the SQ8_0 implementation plan M7 source-size note.

Verification:

```text
cargo fmt --all --check
cargo test -p ullm-runtime-sys cpu_sq_fp8_matvec -- --test-threads=1
cargo test -p ullm-runtime-sys --lib --no-run
git diff --check -- ':!README.md' ':!journal/**'
```

Verification:

```text
cargo fmt --all --check
cargo check -p ullm-engine
python3 -m unittest tests.test_external_benchmark_batch_parser
git diff --check -- ':!README.md'
```

## Follow-up: SQ8_0 Materialized Fallback Result Guard

- Added a benchmark-row guard for `sq_execution_mode=materialized_f32_fallback`.
- `tools/run-external-benchmark.py --parse ullm-model-loop-throughput` now turns unmarked
  materialized fallback rows into `status=failed` with `error.type=invalid_fallback`.
- Explicit fallback rows remain writable when one of these markers is present:
  - CLI: `--allow-materialized-fallback`
  - smoke output: `fallback_allowed=true`
  - smoke output: `diagnostic=true`
  - selected-layer diagnostic command: `sq-fp8-token-ids-model-loop-smoke`
- Fallback rows now preserve `workload.fallback_allowed` and `workload.diagnostic`.
- `tools/summarize-benchmark-results.py` now excludes unmarked materialized SQ8_0 fallback rows
  from the default success table even for older `status=ok` JSONL rows, and adds an `SQ mode`
  column so retained direct/fallback rows are visibly distinct.
- Updated the SQ8_0 plan M4 / Immediate Queue to mark accidental fallback rejection done.

Verification:

```text
python3 -m unittest tests.test_external_benchmark_batch_parser tests.test_summarize_benchmark_results
python3 -m py_compile tools/run-external-benchmark.py tools/summarize-benchmark-results.py
git diff --check -- ':!README.md'
```

End-to-end harness check:

```text
selected_rc 0 selected_status ok selected_error None
unmarked_rc 2 unmarked_status failed unmarked_error {'message': 'materialized_f32_fallback rows must be explicitly marked fallback or excluded from comparable throughput.', 'type': 'invalid_fallback'}
```

## Follow-up: SQ8_0 Projection Descriptor Registry

- Moved SQ8_0 projection dispatch descriptors from CLI-local `main_parts/part_00.rs` into the
  library-side `backend_dispatch.rs`.
- Added a public SQ8_0 projection descriptor registry covering:
  - operations: single, batch, pair, triple matvec;
  - targets: generic and RDNA4;
  - family: direct.
- Added the descriptor naming template `sq8_0_<operation>_<target>_<family>` while preserving the
  existing public IDs such as `sq8_0_matvec_rdna4_direct`.
- `part_00.rs` now resolves SQ8_0 projection implementation IDs through the backend dispatch
  registry API instead of owning the descriptor list locally.
- Kept C++ kernel-family switching as future work; this step only centralizes descriptor metadata
  and selection.
- Updated the SQ8_0 plan M6/M7 status.

Verification:

```text
cargo fmt --all --check
cargo test -p ullm-engine backend_dispatch -- --test-threads=1
cargo check -p ullm-engine
git diff --check -- ':!README.md' ':!journal/**'
```

## Follow-up: M7 runtime helper split

- `part_00.inc` の SQ8_0 matvec helper 群を `runtime/src/kernels/sq8_0/sq8_0_matvec_runtime.inc` へ移設。
- `part_00.inc` には `matvec_f32_host` 後に1か所のみ include を追加。
- `crates/ullm-runtime-sys/build.rs` の `rerun-if-changed` に新規 include を追加。
- 検証: `cargo fmt --all --check`, `cargo check -p ullm-runtime-sys`, `cargo test -p ullm-runtime-sys cpu_sq_fp8_matvec -- --test-threads=1` すべて合格。
- docs/plans は M7 を「SQ8_0 matvec runtime helper split done / C++ TU 1つ / fused kernel implementation pending」に更新済み。

Runtime smoke:

```text
target/debug/ullm-engine sq-fp8-token-ids-mixed-request-state-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d /tmp/ullm-sq8_0-kup6-gate5-down5-policy-20260709-artifact 2 1048576 manifest-all len:2x2 1 1 1024 32 10000000 0
```

Confirmed output included:

```text
sq_execution_mode=direct_fp8_dequant_matvec
sq_projection_boundary=single
sq_projection_implementation_ids=single=sq8_0_matvec_rdna4_direct
sq_fp8_single_matvec_count=132
```

## Follow-up: SQ8_0 API Wrapper Include Split

- Added `runtime/src/ullm_runtime_api_sq8_0.inc`.
- Moved the SQ8_0 API wrappers only (`ullm_runtime_sq_fp8_matvec_f32`,
  `ullm_runtime_sq_fp8_matvec_batch_f32`, `ullm_runtime_sq_fp8_matvec_pair_f32`,
  `ullm_runtime_sq_fp8_matvec_triple_f32`) from `ullm_runtime_api_primitives.inc` into
  the new include, preserving source order.
- Kept existing function bodies and behavior unchanged; only include partitioning changed.
- Updated `runtime/src/ullm_runtime_api.inc` to include the new file directly after
  `ullm_runtime_api_primitives.inc`.
- Updated `crates/ullm-runtime-sys/build.rs` `rerun-if-changed` list to track the new include.
- Updated `docs/plans/sq8-implementation-plan-v0.1.md` M7 to mark SQ8_0 API wrapper split done while
  keeping translation unit single and runtime helpers/fused families as remaining items.

Verification:

```text
cargo fmt --all --check
cargo check -p ullm-runtime-sys
cargo test -p ullm-runtime-sys cpu_sq_fp8_matvec -- --test-threads=1
git diff --check -- ':!README.md' ':!journal/**'
```

## Follow-up: Selected-Layer SQ8_0 Projection Telemetry

- Added SQ8_0 projection telemetry fields to `sq-fp8-token-ids-model-loop-smoke` and the shared
  selected-layer model-loop output path:
  - `sq_execution_mode`
  - `sq_projection_boundary`
  - `sq_projection_implementation_ids`
  - `sq_fp8_single_matvec_count`
  - `sq_fp8_batch_matvec_count`
  - `sq_fp8_pair_matvec_count`
  - `sq_fp8_triple_matvec_count`
- Reset SQ8_0 projection telemetry at the selected-layer model-loop run boundary.
- Verified that this selected-layer path currently materializes SQ8_0 tensors into F32 runtime
  buffers through `Qwen3PackageModelRuntime::load_with_sq_overlay`, so rows with no direct SQ8_0
  matvec counter now report `sq_execution_mode=materialized_f32_fallback` instead of claiming
  direct execution.
- Updated external benchmark parsing to preserve explicit `none` boundary/implementation values
  for SQ overlay rows without inserting `None` for older rows that lack those keys.
- Updated the SQ8_0 plan to mark selected-layer telemetry reporting done while keeping direct
  full-package throughput and M10 `vLLM + FP8` comparison as later work.

R9700 selected-layer smoke:

```text
cargo run -p ullm-engine -- sq-fp8-token-ids-model-loop-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d /tmp/ullm-sq8_0-kup6-gate5-down5-policy-20260709-artifact 2 1048576 3,7 len:2x2 1 2 1024 32 10000000 0
```

Confirmed output included:

```text
sq_execution_mode=materialized_f32_fallback
sq_projection_boundary=none
sq_projection_implementation_ids=none
sq_fp8_single_matvec_count=0
sq_fp8_batch_matvec_count=0
sq_fp8_pair_matvec_count=0
sq_fp8_triple_matvec_count=0
```

Verification:

```text
cargo fmt --all --check
cargo check -p ullm-engine
python3 -m unittest tests.test_external_benchmark_batch_parser
git diff --check -- ':!README.md'
```

## Follow-up: SQ8_0 HIPRTC Source Split

- Moved `sq_fp8_matvec_kernel_source()` out of `runtime/src/ullm_runtime_hiprtc_sources.inc`.
- Added `runtime/src/kernels/sq8_0/sq8_0_matvec_hiprtc.inc` as the SQ8_0-specific HIPRTC source
  holder for single, batch, pair, and triple SQ FP8 matvec kernels.
- Kept one C++ translation unit and the existing `compile_sq_fp8_matvec_*_kernel` entry points.
- Added the new include file to `crates/ullm-runtime-sys/build.rs` `rerun-if-changed` tracking.
- Updated `docs/plans/sq8-implementation-plan-v0.1.md` M7 status to partial.
- File size after split:
  - `runtime/src/kernels/sq8_0/sq8_0_matvec_hiprtc.inc`: 254 lines.
  - `runtime/src/ullm_runtime_hiprtc_sources.inc`: 5798 lines.

Verification:

```text
cargo check -p ullm-engine
cargo test -p ullm-runtime-sys cpu_sq_fp8_matvec -- --test-threads=1
cargo test -p ullm-runtime-sys first_hip_sq_fp8_matvec -- --test-threads=1
git diff --check -- ':!README.md'
```

## Follow-up: SQ8_0 Projection Dispatch At Matvec Boundary

- Added typed SQ8_0 projection dispatch decisions for single, batch, pair, and triple matvec
  operations.
- `PackageAq4ResidentMatvec` now stores the dispatch decisions created from runtime device info.
- The direct SQ FP8 single/batch/pair/triple matvec paths fetch the dispatch decision immediately
  before calling the existing runtime kernel and record projection telemetry through that decision.
- Runtime GPU-architecture detection now treats `compute_major == 12` as `RDNA4`; the R9700 reports
  an empty `gcn_arch_name` but `compute_major=12`, so this is required for RDNA4 dispatch.
- Kernel-family switching is still unchanged; this step moves dispatch from reporting-only toward
  the execution boundary.

Fresh SQ8_0 artifact and smoke:

- Generated `/tmp/ullm-sq8_0-kup6-gate5-down5-policy-20260709-artifact` from
  `benchmarks/results/2026-07-08/sq-fp8-kup6-gate5-down5-policy-v0.1.json`.
- Artifact summary: `fp8_tensor_count=22`, `passthrough_tensor_count=753`,
  `compact_resident_bytes_estimate=18579553248`.
- R9700 materialize smoke passed for
  `model.language_model.layers.3.self_attn.k_proj.weight`.
- R9700 selected-layer model-loop smoke passed for `layers=3,7`, `len:2x2`.
- R9700 full mixed request-state smoke passed for `manifest-all`, `len:2x2`, with
  `sq_execution_mode=direct_fp8_dequant_matvec`,
  `sq_projection_implementation_ids=single=sq8_0_matvec_rdna4_direct`, and
  `sq_fp8_single_matvec_count=132`.

Verification:

```text
python3 tools/build-sq-fp8-w8a16-artifact.py --source-model-dir /home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B --base-package /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d --policy-json benchmarks/results/2026-07-08/sq-fp8-kup6-gate5-down5-policy-v0.1.json --output-artifact /tmp/ullm-sq8_0-kup6-gate5-down5-policy-20260709-artifact --summary-json /tmp/ullm-sq8_0-kup6-gate5-down5-policy-20260709-summary.json --overwrite
target/debug/ullm-engine sq-fp8-materialize-smoke /tmp/ullm-sq8_0-kup6-gate5-down5-policy-20260709-artifact 2 model.language_model.layers.3.self_attn.k_proj.weight 2 0
target/debug/ullm-engine sq-fp8-token-ids-model-loop-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d /tmp/ullm-sq8_0-kup6-gate5-down5-policy-20260709-artifact 2 1048576 3,7 len:2x2 1 2 1024 32 10000000 0
cargo run -p ullm-engine -- sq-fp8-token-ids-mixed-request-state-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d /tmp/ullm-sq8_0-kup6-gate5-down5-policy-20260709-artifact 2 1048576 manifest-all len:2x2 1 1 1024 32 10000000 0
cargo fmt --all --check
cargo check -p ullm-engine
python3 -m unittest tests.test_external_benchmark_batch_parser
git diff --check -- ':!README.md'
```

## Follow-up: SQ8_0 Runtime Helper Split (C++)

- Moved SQ8_0 matvec runtime helper群 (`sq_fp8_*_host`, kernel caches, `sq_fp8_*_hip_kernel`,
  `sq_fp8_*_hip_staging`) from `runtime/src/ullm_runtime_parts/part_00.inc` into
  `runtime/src/kernels/sq8_0/sq8_0_matvec_runtime.inc`.
- Added one include in `part_00.inc` immediately after `matvec_f32_host`.
- Added the new include to `crates/ullm-runtime-sys/build.rs` `rerun-if-changed`.
- Updated `docs/plans/sq8-implementation-plan-v0.1.md` M7: helper split done / one TU, fused families still
  pending.

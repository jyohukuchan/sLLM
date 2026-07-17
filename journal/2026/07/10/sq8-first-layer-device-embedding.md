# SQ8 First-Layer Device Embedding

- Added a resident embedding path for SQ8_0 mixed request-state first-layer inputs.
- `PackageEmbeddingRuntime` can now gather a token row into a caller-provided `RuntimeBuffer`, so batch slots can use distinct device buffers without duplicating the resident embedding matrix.
- The mixed request-state runner loads resident embedding once, allocates one F32 output buffer per request, and feeds first-layer inputs through the existing device-to-device layer path when resident embedding is available.
- The old host residual path remains as fallback when resident embedding is not available.
- Added `first_layer_input_source={device_embedding|host_residual}` to the smoke output and preserved it in `tools/run-external-benchmark.py` workload rows.
- Verified:
  - `cargo build -p ullm-engine`
  - `cargo fmt --all --check`
  - `cargo test -p ullm-engine -- package_model_loop_cli_tail_tests::infer_mixed_request_state_real_batch_flags_enables_both_phases_when_batch_matvec_used`
  - `python3 -m unittest tests.test_external_benchmark_batch_parser`
  - `python3 -m unittest tests.test_external_benchmark_batch_parser tests.test_summarize_sq8_vllm_batch_grid`
  - `python3 -m py_compile tools/run-external-benchmark.py tests/test_external_benchmark_batch_parser.py`
  - `git diff --check -- ':!README.md'`
- Short layer3 smoke records `first_layer_input_source=device_embedding` and `sq_diagnostic_host_staging_read_count=0`, `sq_diagnostic_host_staging_write_count=0`.
- Full 40-layer Qwen3-14B-FP8 `pp16/tg8/b2` direct smoke records `first_layer_input_source=device_embedding`, `sq_fp8_batch_matvec_count=6720/6720`, and host staging `0/0`.
- Re-ran the normalized M10 comparison uLLM side for `pp16/tg8` b2/b4/b8 at commit `51d9f75`.
- The new rows are in `benchmarks/results/2026-07-10/sq8-qwen3-14b-no-host-staging-refresh/results.jsonl`.
- All three rows record `first_layer_input_source=device_embedding`, `sq_fp8_batch_matvec_count=6720/6720`, `sq_projection_kernel_families=batch=direct`, and host staging read/write `0/0`.
- The comparison helper passes with:
  - `--require-normalized-throughput-comparison`
  - `--require-ullm-sq-batch-coverage`
  - `--require-ullm-sq-kernel-families`
  - `--require-ullm-sq-no-host-staging`
- Tightened `--require-ullm-sq-no-host-staging` so it requires all four
  `workload.sq_diagnostic_host_staging_*` fields to be present and zero. Missing diagnostic fields
  now fail instead of being treated as implicitly clean.
- Tightened `--require-normalized-throughput-comparison` so it derives per-request
  prompt/generated token shape for each request count and fails when the selected uLLM/vLLM shape
  sets do not overlap.
- Tightened the same gate to require overlapping non-empty `model.name` values per request count.
  This blocks accidental Qwen3.5-vs-Qwen3-14B pairings without requiring `model.format`,
  `model.quantization`, or `workload.kv_cache_dtype` to match.
- Tightened the same gate to require each selected row's `workload.context_length` to be present
  and at least the derived per-request prompt plus generated token count. The gate does not require
  identical context limits across uLLM and vLLM.
- Added `tools/run-external-benchmark.py --parse ullm-serving-throughput` as a result-preserving
  inlet for future uLLM serving-style/offline-serving throughput rows. It uses
  `harness.class=ullm_serving_throughput_candidate` and `serving_parity_candidate=false` so candidate
  rows cannot accidentally satisfy the final serving parity gate.
- Clarified the M10 plan so `ullm-serving-throughput` rows should preserve a machine-readable
  serving candidate contract before any promotion to final parity. The contract should record loop
  kind, scheduler policy, request source/arrival pattern, tokenizer/HTTP-server inclusion,
  runtime/weight reuse, load-excluded timing semantics, final-logits inclusion, and conservative
  parity blockers.
- Added parser-side preservation for that contract: `tools/run-external-benchmark.py --parse
  ullm-serving-throughput` now writes `harness.ullm_serving_candidate` with serving metadata,
  runner-provided `parity_blockers`, parser-derived blockers for weak semantics, and
  `serving_parity_candidate=false`.
- Clarified that final serving-parity promotion must reject non-empty
  `harness.ullm_serving_candidate.parity_blockers`, even if a future row opts into
  `serving_parity_candidate=true`.

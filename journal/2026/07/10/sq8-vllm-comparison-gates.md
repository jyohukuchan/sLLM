# SQ8 vLLM comparison gates

## Summary

- Added `--require-ullm-sq-kernel-families` to the SQ8/vLLM batch-grid summarizer.
- The gate checks selected `uLLM` + `SQ8_0` rows for a non-empty, non-`none`
  `workload.sq_projection_kernel_families` field.
- Added `--require-ullm-sq-batch-coverage` to reject selected `uLLM` + `SQ8_0`
  rows without a `batch` projection boundary or with
  `sq_fp8_batch_matvec_count < sq_fp8_expected_all_batch_matvec_count`.
- Added `--require-normalized-throughput-comparison` to explicitly compare
  same-shape uLLM CLI model-loop diagnostic rows against vLLM serving-throughput
  rows without pretending they are strict serving parity rows.
- The gate is intended to be used with `--require-engines uLLM,vLLM` and
  `--require-engine-grid` for M10 comparison rows.

## Verification

- `python3 -m unittest tests.test_summarize_sq8_vllm_batch_grid`
- `python3 -m py_compile tools/summarize-sq8-vllm-batch-grid.py tests/test_summarize_sq8_vllm_batch_grid.py`
- `git diff --check -- ':!README.md'`
- Smoke: `sq8-stack-resident-all-batch` passes `--require-ullm-sq-batch-coverage`.
- Smoke: `sq8-stack-resident-qkv-batch` fails `--require-ullm-sq-batch-coverage`
  because it reports `sq_fp8_batch_matvec_count=9` and expected `21`.
- Smoke: existing b2/b4/b8 no-final uLLM rows plus vLLM rows pass
  `--require-normalized-throughput-comparison --require-ullm-sq-batch-coverage`.
- Smoke: the same rows still fail `--require-serving-parity`, preserving the
  distinction between normalized comparison and serving parity.
- Re-ran b2/b4/b8 uLLM rows on current `52b866b` into
  `benchmarks/results/2026-07-10/sq8-qwen3-14b-normalized-kernel-family-refresh/`.
  The refreshed rows record `sq_projection_kernel_families=batch=direct` and pass
  `--require-normalized-throughput-comparison --require-ullm-sq-batch-coverage --require-ullm-sq-kernel-families`
  together with the existing vLLM b2/b4/b8 rows.
- Added `--show-sq-details` to the batch-grid summarizer so M10 tables can display
  `SQ boundary`, `SQ family`, and `SQ batch` columns without changing default output.

# Package batch throughput JSONL preservation v1

Date: 2026-07-08

## Purpose

This records the T1 JSONL preservation guard for `package-batch-throughput-bench-v0.1` reports.

The goal is not performance measurement. The goal is to ensure that converted
`inference-benchmark-result-v0.1` rows preserve the fields needed for later AQ4/SQ/vLLM batch
throughput comparisons.

## 前回の要点

- `package-batch-throughput-bench` already emits raw total-throughput metrics and cold prefill
  accounting fields.
- T1 still needed confidence that the workload runner conversion path preserves those fields in
  JSONL rows.
- T1 real batch execution is still not implemented; the current runner remains logical batch.

## 今回の変更点

- `tools/run-external-benchmark.py` now falls back from raw `batching.prefill_executor` and
  `batching.resolved_prefill_executor` into JSONL `workload.*` executor fields when the runner CLI
  did not supply them.
- `tools/run-external-benchmark.py` now has a package-batch-specific memory enrichment helper that
  preserves `memory.kv_cache_bytes_total`.
- Added `tests/test_external_benchmark_batch_parser.py`.
- The test fixes preservation of:
  - `prefill_total_input_tokens_per_second`
  - `decode_total_generated_tokens_per_second`
  - `end_to_end_total_tokens_per_second`
  - `workload.prefill_mode`
  - `workload.cached_prefix_tokens_per_request`
  - `workload.new_prefill_tokens_per_request`
  - `workload.total_context_tokens_after_prefill_per_request`
  - `workload.cached_prefix_total_tokens`
  - `workload.total_context_tokens_after_prefill`
  - `workload.estimated_prefill_attention_work_tokens`
  - `workload.prefill_executor`
  - `workload.resolved_prefill_executor`
  - `memory.kv_cache_bytes_total`

## Synthetic main-path smoke

A synthetic `package-batch-throughput-bench-v0.1` report was passed through
`tools/run-external-benchmark.py --parse ullm-package-batch-throughput`.

Representative preserved row fields:

| field | value |
| --- | --- |
| `schema_version` | `inference-benchmark-result-v0.1` |
| `status` | `ok` |
| `workload.batch_size` | `2` |
| `workload.concurrent_requests` | `2` |
| `workload.prefill_mode` | `cold` |
| `workload.prefill_executor` | `cached_prefix_rdna4_fp8_auto` |
| `workload.resolved_prefill_executor` | `cached_prefix_flash2_fp8q` |
| `workload.cached_prefix_total_tokens` | `0` |
| `workload.total_context_tokens_after_prefill` | `8` |
| `workload.estimated_prefill_attention_work_tokens` | `20` |
| `metrics.prefill_total_input_tokens_per_second` | `100.0` |
| `metrics.decode_total_generated_tokens_per_second` | `100.0` |
| `metrics.end_to_end_total_tokens_per_second` | `80.0` |
| `memory.kv_cache_bytes_total` | `98304` |
| `batching.mode` | `logical` |
| `batching.prefill_real_batch` | `false` |

## Verification

- `python3 -m py_compile tools/run-external-benchmark.py tests/test_external_benchmark_batch_parser.py`
- `python3 -m unittest tests.test_external_benchmark_batch_parser tests.test_compare_package_guards tests.test_sq_candidate_runtime_row`
- Synthetic `run-external-benchmark.py --parse ullm-package-batch-throughput` main-path smoke.

## 次の行動

1. Treat T1 JSONL/schema preservation for package batch reports as done for v0.1.
2. Keep T1 real batch runner as not done.
3. Add real batch prefill/decode executors before using batch throughput rows for SQ format
   performance decisions.

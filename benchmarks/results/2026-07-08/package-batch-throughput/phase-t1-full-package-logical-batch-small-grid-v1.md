# T1 Full Package Logical Batch Small Grid v1

## Summary

This run records AQ4 full-package `package-batch-throughput-bench` rows for `batch=1/4/8`.

These are full package rows, but not real request-batch throughput rows. The engine report preserves
`batching.mode=logical`, `prefill_real_batch=false`, `decode_real_batch=false`,
`runtime_reused_across_requests=false`, and `weights_reloaded_per_request=true`.

Result directory:

- `benchmarks/results/2026-07-08/package-batch-throughput/phase-t1-full-package-logical-batch-small-grid-v1/`

Workload:

- `benchmarks/workloads/r9700-aq4-full-package-logical-batch-small-grid.json`

## Grid

| case | batch | prompt/request | generated/request | status | verified | batching mode | prefill real | decode real | prefill total tok/s | decode generated tok/s | end-to-end tok/s | KV cache bytes | VRAM consumed bytes |
| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `aq4-r9700-fullpkg-logical-b1-pp4-tg2` | 1 | 4 | 2 | ok | true | logical | false | false | 33.574674811 | 68.797769301 | 2.172326744 | 393216 | 4279500800 |
| `aq4-r9700-fullpkg-logical-b4-pp4-tg2` | 4 | 4 | 2 | ok | true | logical | false | false | 58.744142319 | 69.070493674 | 2.433498426 | 1572864 | 4206096384 |
| `aq4-r9700-fullpkg-logical-b8-pp4-tg2` | 8 | 4 | 2 | ok | true | logical | false | false | 67.008573726 | 69.071288010 | 2.533759132 | 3145728 | 4279500800 |

## Interpretation

- JSONL schema preservation is working for full package rows across `batch=1/4/8`.
- `prefill_total_input_tokens_per_second`, `decode_total_generated_tokens_per_second`,
  `end_to_end_total_tokens_per_second`, `memory.kv_cache_bytes_total`, VRAM peak/consumed, and
  `correctness.verified_all` are present in all measured rows.
- These rows are not final SQ throughput rows. The current implementation sequentially invokes the
  single-request package path for each request.
- The next T1 step is to connect real request-batch prefill/decode executors to the full package
  path and emit `batching.mode=real` rows for the same `batch=1/4/8` grid.

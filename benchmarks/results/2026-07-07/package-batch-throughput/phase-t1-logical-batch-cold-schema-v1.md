# Package batch throughput logical cold schema v1

Date: 2026-07-07

Runtime commit: `d262e10c9957cf05c34c53fa260927444323aebe`

Purpose:

- Verify that `package-batch-throughput-bench` emits the Phase T1 total
  throughput fields as one JSON report.
- Verify the added cold-prefill accounting fields:
  `workload.prefill_mode`, `cached_prefix_tokens_per_request`,
  `new_prefill_tokens_per_request`,
  `total_context_tokens_after_prefill_per_request`,
  `metrics.cached_prefix_total_tokens`,
  `metrics.total_context_tokens_after_prefill`, and
  `metrics.estimated_prefill_attention_work_tokens`.
- Exercise logical batch `B=1/2/4` on R9700.

This is a schema/control-plane smoke, not a performance representative run.
It uses one package layer, prompt length 4, generated length 2, and logical
batching. Each request still invokes the existing single-request path and
reloads weights.

Command shape:

```bash
ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL=1 \
ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1 \
ULLM_REQUIRE_HIP_DECODE_ATTN_KERNEL=1 \
  target/release/ullm-engine package-batch-throughput-bench \
  /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d \
  2 1048576 3 len:4xB 2 4 512 64 10000000 0 gpu_resident_f32 none none
```

## Results

| B | prefill mode | batching mode | prefill tokens | timed decode tokens | end-to-end tokens | estimated attention work | total context after prefill | prefill tok/s | decode tok/s | end-to-end tok/s | batch wall ms | verified |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | cold | logical | 4 | 1 | 6 | 10 | 4 | 53.808662 | 231.111649 | 8.270042 | 725.510248 | true |
| 2 | cold | logical | 8 | 2 | 12 | 20 | 8 | 98.478364 | 233.595603 | 9.792617 | 1225.413006 | true |
| 4 | cold | logical | 16 | 4 | 24 | 40 | 16 | 172.597125 | 233.697961 | 10.443645 | 2298.048298 | true |

Latency and memory:

| B | TTFT p50 ms | TTFT p95 ms | request latency p50 ms | request latency p95 ms | kv cache bytes total |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 74.337474 | 74.337474 | 725.431376 | 725.431376 | 49152 |
| 2 | 5.824871 | 75.411247 | 497.780487 | 727.454224 | 98304 |
| 4 | 5.925441 | 74.952944 | 536.057875 | 726.771644 | 196608 |

## Interpretation

- The T1 logical batch runner now emits separate
  `prefill_total_input_tps`, `decode_total_generated_tps`, and
  `end_to_end_total_tps` fields.
- The cold-prefill workload now records the same prefix/chunk/context fields
  needed by cached-prefix runs, with cached-prefix totals set to zero.
- `estimated_prefill_attention_work_tokens` follows the existing component
  convention: sum over requests of `N * (N + 1) / 2`.
- TTFT p95 is dominated by the first request because this logical smoke still
  reloads weights per request. This is expected and is recorded by
  `batching.weights_reloaded_per_request=true`.
- These rows validate schema and accounting only. They must not be used as
  SQ candidate batch performance rows.

## Verification

- `cargo fmt --all --check`
- `cargo test -p ullm-engine package_token_ids_logits_tests -- --test-threads=1`
- `cargo check -p ullm-engine`
- `cargo build -p ullm-engine --release`
- R9700 `package-batch-throughput-bench` schema smokes shown above

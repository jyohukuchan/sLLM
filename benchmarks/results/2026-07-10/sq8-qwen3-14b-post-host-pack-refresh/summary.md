# SQ8_0 Qwen3-14B Post Host-Pack Refresh

Date: 2026-07-10

## 前回の要点

- The previous normalized M10 refresh on `52b866b` proved the b2/b4/b8 same-shape uLLM rows
  could pass the stricter comparison gate with `sq_projection_kernel_families=batch=direct` and
  `sq_fp8_batch_matvec_count=6720/6720`.
- Those rows still reported host residual staging writes that scaled with request count:
  `0/72` for b2, `0/120` for b4, and `0/216` for b8.
- vLLM b2/b4/b8 FP8 rows remain in
  `benchmarks/results/2026-07-09/sq8-vllm-fp8-comparison/results.jsonl` and are reused as the
  external baseline slice.

## 今回の変更点

- Re-ran uLLM b2/b4/b8 on `7af8c3a` after packing first-layer residual inputs once per timestep
  and copying request slices device-to-device.
- The refreshed rows still pass the same-shape normalized M10 gate with direct SQ8_0 batch matvec
  coverage and kernel-family telemetry.
- Host staging writes now stay at `0/24` for b2, b4, and b8; write bytes still scale with batch
  width because each timestep packs the full batch residual buffer once.

```bash
python3 tools/summarize-sq8-vllm-batch-grid.py \
  benchmarks/results/2026-07-10/sq8-qwen3-14b-post-host-pack-refresh/results.jsonl \
  benchmarks/results/2026-07-09/sq8-vllm-fp8-comparison/results.jsonl \
  --workload-prefix pp16-tg8 \
  --requests 2,4,8 \
  --require-normalized-throughput-comparison \
  --require-ullm-sq-batch-coverage \
  --require-ullm-sq-kernel-families \
  --max-ullm-sq-host-staging-write-count 24 \
  --show-sq-details
```

## Result

| Engine | Case | Harness | Requests | Prompt tokens | Generated tokens | Prefill tok/s | Decode tok/s | Total tok/s | Consumed GiB | Decode x GiB | SQ boundary | SQ family | SQ batch | SQ staging ops | SQ staging MiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| uLLM | qwen3-14b-fp8-sq8-full-mixed-real-batch-no-final-pp16-tg8-b2-post-host-pack | cli_model_loop_diagnostic | 2 | 32 | 16 | 15.45 | 16.60 | 15.81 | 12.89 | 213.98 | batch | batch=direct | 6720/6720 | 0/24 | 0.00/0.94 |
| uLLM | qwen3-14b-fp8-sq8-full-mixed-real-batch-no-final-pp16-tg8-b4-post-host-pack | cli_model_loop_diagnostic | 4 | 64 | 32 | 16.24 | 16.78 | 16.41 | 13.06 | 219.15 | batch | batch=direct | 6720/6720 | 0/24 | 0.00/1.88 |
| uLLM | qwen3-14b-fp8-sq8-full-mixed-real-batch-no-final-pp16-tg8-b8-post-host-pack | cli_model_loop_diagnostic | 8 | 128 | 64 | 16.53 | 16.69 | 16.58 | 13.57 | 226.54 | batch | batch=direct | 6720/6720 | 0/24 | 0.00/3.75 |
| vLLM | vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b2-tp1-rocr | serving_throughput_benchmark | 2 | 32 | 16 | 34.41 | 17.21 | 51.62 | 19.57 | 336.72 | - | - | - | - | - |
| vLLM | vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b4-tp1-rocr | serving_throughput_benchmark | 4 | 64 | 32 | 135.04 | 67.52 | 202.56 | 28.05 | 1894.13 | - | - | - | - | - |
| vLLM | vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b8-tp1-rocr | serving_throughput_benchmark | 8 | 128 | 64 | 236.01 | 118.01 | 354.02 | 28.05 | 3310.52 | - | - | - | - | - |

Key uLLM fields:

| Case | SQ8 kernel families | SQ8 batch matvec | Host read/write | Host write MiB | Final logits in total |
| --- | --- | --- | --- | ---: | --- |
| b2 | `batch=direct` | `6720/6720` | `0/24` | 0.94 | `false` |
| b4 | `batch=direct` | `6720/6720` | `0/24` | 1.88 | `false` |
| b8 | `batch=direct` | `6720/6720` | `0/24` | 3.75 | `false` |

## Interpretation

The host residual staging count is now independent of request count for this TOP_K=0 model-loop
comparison shape. This does not yet make the row serving-parity with vLLM, but it removes the
previous request-count-scaled residual upload artifact from the normalized M10 diagnostic slice.

## 次の行動

- Carry `benchmarks/results/2026-07-10/sq8-qwen3-14b-post-host-pack-refresh/results.jsonl` forward
  as the current uLLM side of the interim normalized M10 comparison.
- The remaining comparison gap is still harness class: uLLM is `cli_model_loop_diagnostic`, while
  vLLM is `serving_throughput_benchmark`.

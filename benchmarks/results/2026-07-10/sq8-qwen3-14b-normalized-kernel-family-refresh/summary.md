# SQ8_0 Qwen3-14B Normalized Kernel-Family Refresh

Date: 2026-07-10

## 前回の要点

- 2026-07-09 の b2/b4/b8 uLLM rows already proved full 40-layer real-batch SQ8_0 coverage
  with `final_logits_in_total=false` and `sq_fp8_batch_matvec_count=6720/6720`.
- Those rows predated `sq_projection_kernel_families`, so the stricter M10 comparison gate
  `--require-ullm-sq-kernel-families` correctly rejected them.
- vLLM b2/b4/b8 FP8 rows already exist in
  `benchmarks/results/2026-07-09/sq8-vllm-fp8-comparison/results.jsonl`.

## 今回の変更点

- Re-ran uLLM b2/b4/b8 on current `52b866b` with the same Qwen3-14B-FP8 thin package and full
  SQ8_0 sidecar artifact.
- Each row now records `sq_projection_kernel_families=batch=direct` next to
  `sq_projection_implementation_ids=batch=sq8_0_matvec_batch_r9700_direct`.
- The refreshed rows pass the same-shape normalized M10 gate together with the existing vLLM rows:

```bash
python3 tools/summarize-sq8-vllm-batch-grid.py \
  benchmarks/results/2026-07-10/sq8-qwen3-14b-normalized-kernel-family-refresh/results.jsonl \
  benchmarks/results/2026-07-09/sq8-vllm-fp8-comparison/results.jsonl \
  --workload-prefix pp16-tg8 \
  --requests 2,4,8 \
  --require-normalized-throughput-comparison \
  --require-ullm-sq-batch-coverage \
  --require-ullm-sq-kernel-families \
  --show-sq-details
```

## Result

| Engine | Case | Harness | Requests | Prompt tokens | Generated tokens | Prefill tok/s | Decode tok/s | Total tok/s | Consumed GiB | Decode x GiB | SQ boundary | SQ family | SQ batch |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| uLLM | qwen3-14b-fp8-sq8-full-mixed-real-batch-no-final-pp16-tg8-b2-kernel-family | cli_model_loop_diagnostic | 2 | 32 | 16 | 15.45 | 16.62 | 15.82 | 12.81 | 212.79 | batch | batch=direct | 6720/6720 |
| uLLM | qwen3-14b-fp8-sq8-full-mixed-real-batch-no-final-pp16-tg8-b4-kernel-family | cli_model_loop_diagnostic | 4 | 64 | 32 | 16.23 | 16.79 | 16.41 | 13.06 | 219.32 | batch | batch=direct | 6720/6720 |
| uLLM | qwen3-14b-fp8-sq8-full-mixed-real-batch-no-final-pp16-tg8-b8-kernel-family | cli_model_loop_diagnostic | 8 | 128 | 64 | 16.52 | 16.69 | 16.57 | 13.57 | 226.55 | batch | batch=direct | 6720/6720 |
| vLLM | vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b2-tp1-rocr | serving_throughput_benchmark | 2 | 32 | 16 | 34.41 | 17.21 | 51.62 | 19.57 | 336.72 | - | - | - |
| vLLM | vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b4-tp1-rocr | serving_throughput_benchmark | 4 | 64 | 32 | 135.04 | 67.52 | 202.56 | 28.05 | 1894.13 | - | - | - |
| vLLM | vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b8-tp1-rocr | serving_throughput_benchmark | 8 | 128 | 64 | 236.01 | 118.01 | 354.02 | 28.05 | 3310.52 | - | - | - |

Key uLLM fields:

| Case | SQ8 kernel families | SQ8 batch matvec | Host read/write | Final logits in total |
| --- | --- | --- | --- | --- |
| b2 | `batch=direct` | `6720/6720` | `0/72` | `false` |
| b4 | `batch=direct` | `6720/6720` | `0/120` | `false` |
| b8 | `batch=direct` | `6720/6720` | `0/216` | `false` |

## Interpretation

These rows are the current machine-gated M10 normalized comparison input. They are still not strict
serving parity: uLLM is a CLI model-loop diagnostic row, while vLLM is a serving-throughput
benchmark row. The value of this refresh is that the comparison can now require explicit SQ8_0
kernel family telemetry, full batch projection coverage, and same-shape uLLM/vLLM request coverage
at the same time.

## 次の行動

- Use this refreshed uLLM JSONL plus the existing vLLM JSONL for interim normalized M10 comparison
  tables.
- Keep `--require-serving-parity` as the stricter future gate for an eventual server-style uLLM row.

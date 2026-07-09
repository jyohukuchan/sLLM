# SQ8_0 Qwen3-14B No-Host-Staging Refresh

> **Quarantined:** these uLLM rows used an invalid v0.1 sidecar that omitted source
> `weight_scale_inv`. Keep them as connection diagnostics only. See the
> [quarantine record](../sq8-qwen3-14b-invalid-sidecar-quarantine.md).

Date: 2026-07-10

## 前回の要点

- The post-host-pack refresh on `7af8c3a` reduced first-layer host residual staging to one write
  per timestep, so b2/b4/b8 all reported `0/24` host read/write operations.
- The remaining writes came from the first layer receiving packed host residuals.
- Existing vLLM b2/b4/b8 FP8 rows remain in
  `benchmarks/results/2026-07-09/sq8-vllm-fp8-comparison/results.jsonl`.

## 今回の変更点

- Re-ran uLLM b2/b4/b8 on `51d9f75` after the mixed request-state first layer began using
  resident device embedding buffers.
- All refreshed uLLM rows record `first_layer_input_source=device_embedding`,
  `sq_fp8_batch_matvec_count=6720/6720`, and host staging `0/0`.
- The comparison gate now includes `--require-ullm-sq-no-host-staging`.

```bash
python3 tools/summarize-sq8-vllm-batch-grid.py \
  benchmarks/results/2026-07-10/sq8-qwen3-14b-no-host-staging-refresh/results.jsonl \
  benchmarks/results/2026-07-09/sq8-vllm-fp8-comparison/results.jsonl \
  --workload-prefix pp16-tg8 \
  --requests 2,4,8 \
  --require-normalized-throughput-comparison \
  --require-ullm-sq-batch-coverage \
  --require-ullm-sq-kernel-families \
  --require-ullm-sq-no-host-staging \
  --show-sq-details
```

## Result

| Engine | Case | Harness | Requests | Prompt tokens | Generated tokens | Prefill tok/s | Decode tok/s | Total tok/s | Consumed GiB | Decode x GiB | SQ boundary | SQ family | SQ batch | SQ staging ops | SQ staging MiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| uLLM | qwen3-14b-fp8-sq8-full-mixed-real-batch-no-final-pp16-tg8-b2-no-host-staging | cli_model_loop_diagnostic | 2 | 32 | 16 | 13.96 | 16.47 | 14.70 | 14.34 | 236.28 | batch | batch=direct | 6720/6720 | 0/0 | 0.00/0.00 |
| uLLM | qwen3-14b-fp8-sq8-full-mixed-real-batch-no-final-pp16-tg8-b4-no-host-staging | cli_model_loop_diagnostic | 4 | 64 | 32 | 15.53 | 16.70 | 15.90 | 14.51 | 242.38 | batch | batch=direct | 6720/6720 | 0/0 | 0.00/0.00 |
| uLLM | qwen3-14b-fp8-sq8-full-mixed-real-batch-no-final-pp16-tg8-b8-no-host-staging | cli_model_loop_diagnostic | 8 | 128 | 64 | 16.14 | 16.66 | 16.31 | 15.02 | 250.27 | batch | batch=direct | 6720/6720 | 0/0 | 0.00/0.00 |
| vLLM | vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b2-tp1-rocr | serving_throughput_benchmark | 2 | 32 | 16 | 34.41 | 17.21 | 51.62 | 19.57 | 336.72 | - | - | - | - | - |
| vLLM | vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b4-tp1-rocr | serving_throughput_benchmark | 4 | 64 | 32 | 135.04 | 67.52 | 202.56 | 28.05 | 1894.13 | - | - | - | - | - |
| vLLM | vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b8-tp1-rocr | serving_throughput_benchmark | 8 | 128 | 64 | 236.01 | 118.01 | 354.02 | 28.05 | 3310.52 | - | - | - | - | - |

Key uLLM fields:

| Case | First-layer input | SQ8 kernel families | SQ8 batch matvec | Host read/write | Final logits in total |
| --- | --- | --- | --- | --- | --- |
| b2 | `device_embedding` | `batch=direct` | `6720/6720` | `0/0` | `false` |
| b4 | `device_embedding` | `batch=direct` | `6720/6720` | `0/0` | `false` |
| b8 | `device_embedding` | `batch=direct` | `6720/6720` | `0/0` | `false` |

## Interpretation

This removes the remaining diagnostic host staging copies from the current TOP_K=0 model-loop
comparison shape. The rows are still CLI model-loop diagnostics rather than server-parity rows, but
the SQ8_0 side of the normalized M10 comparison can now require no host staging, direct SQ8_0 batch
coverage, and explicit kernel-family telemetry together.

## 次の行動

- Carry this JSONL forward as the current uLLM side of the interim normalized M10 comparison.
- The remaining comparison gap is harness class: uLLM is still `cli_model_loop_diagnostic`, while
  vLLM is `serving_throughput_benchmark`.

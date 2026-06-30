# llama.cpp ROCm baseline summary - 2026-06-30

## Environment

- llama.cpp: `b9844` / `6c5de1cc83537bce5616ed08474f6fe119973a27`
- Backend: ROCm HIP `7.2.53211-e1a6bc5663`
- Host: `homelab1-WRX80-Creator`
- Devices visible to llama.cpp:
  - `ROCm0`: AMD Radeon Pro V620, `gfx1030`, 30704 MiB
  - `ROCm1`: AMD Radeon Graphics, `gfx1201`, 32624 MiB
  - `ROCm2`: AMD Radeon Pro V620, `gfx1030`, 30704 MiB

## Result Files

| File | Rows | Status | Scope |
| --- | ---: | --- | --- |
| `2026-06-30-llamacpp-qwen35-q4-rocm.jsonl` | 36 | 32 ok, 4 unsupported | Qwen3.5-27B Q4_K_M, V620/R9700 single GPU sweep; unsupported rows for vLLM, SGLang, ROCm/ATOM, TensorRT-LLM |
| `2026-06-30-llamacpp-qwen35-a3b-q4-single.jsonl` | 6 | 6 ok | Qwen3.5-35B-A3B Q4_K_M, V620/R9700 single GPU representative sweep |
| `2026-06-30-llamacpp-qwen35-q4-multigpu-representative.jsonl` | 8 | 8 ok | Qwen3.5-27B and 35B-A3B Q4_K_M, `ROCm0/ROCm2` and `ROCm0/ROCm1/ROCm2` layer split representative sweep |
| `2026-06-30-llamacpp-qwen35-27b-fp8-single.jsonl` | 8 | 8 failed | Qwen3.5-27B FP8_E4M3 / FP8_E5M2 load attempts on V620/R9700 |

## Representative Results

| Model | Target | Workload | Prefill tok/s | Decode tok/s | Total tok/s |
| --- | --- | --- | ---: | ---: | ---: |
| Qwen3.5-27B Q4_K_M | V620 single | pp8192/tg512/b2048 | 295.62 | 19.62 | 161.78 |
| Qwen3.5-27B Q4_K_M | R9700 single | pp8192/tg512/b2048 | 909.63 | 27.29 | 313.47 |
| Qwen3.5-35B-A3B Q4_K_M | V620 single | pp8192/tg128/b2048 | 1450.06 | 79.26 | 1145.31 |
| Qwen3.5-35B-A3B Q4_K_M | R9700 single | pp8192/tg128/b2048 | 2422.51 | 77.30 | 1651.60 |
| Qwen3.5-27B Q4_K_M | V620 x2 layer | pp2048/tg128/b2048 | 595.59 | 16.75 | 196.38 |
| Qwen3.5-27B Q4_K_M | V620/R9700/V620 layer | pp2048/tg128/b2048 | 833.73 | 17.24 | 220.22 |
| Qwen3.5-35B-A3B Q4_K_M | V620 x2 layer | pp2048/tg128/b2048 | 2310.23 | 48.24 | 614.75 |
| Qwen3.5-35B-A3B Q4_K_M | V620/R9700/V620 layer | pp2048/tg128/b2048 | 3050.13 | 50.44 | 678.10 |

## Notes

- `llama-bench` was rebuilt after updating llama.cpp from `86b94708f224` to `6c5de1cc8353`.
- `llama-bench` reports prompt-processing and token-generation rows separately. The uLLM JSONL `total_tokens_per_second` is computed from both averages.
- `context_length` in these rows is `prompt_tokens + generated_tokens`; this `llama-bench` mode does not expose an independent `n_ctx` sweep.
- Multi-device `llama-bench` targets must use slash separators such as `ROCm0/ROCm2`. Comma separators are interpreted as a device sweep.
- Qwen3.5-27B FP8_E4M3 and FP8_E5M2 failed to load with llama.cpp `b9844`; this is recorded as failed rows, not omitted.

# Quant family memory summary - 2026-06-30

## Scope

- Engine: llama.cpp `b9844` / `6c5de1cc83537bce5616ed08474f6fe119973a27`
- Device: R9700 target via llama.cpp `ROCm1`
- Model family: Qwen3.5-27B local GGUF artifacts
- Workload: `pp512/tg128/b2048` and `pp2048/tg128/b2048`
- Memory: `rocm-smi --showmeminfo vram --json`, peak total used VRAM minus pre-command total used VRAM

## Result Table

| Status | Engine | Model | Family | Quant | Target | Workload | Decode tok/s | Consumed GiB | Decode x GiB | Source |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| ok | llama.cpp | Qwen3.5-27B | I-Quant | IQ3_S | R9700 | pp512/tg128/b2048 | 26.95 | 12.28 | 330.93 | `2026-06-30-llamacpp-qwen35-27b-quantfamilies-r9700-memory.jsonl` |
| ok | llama.cpp | Qwen3.5-27B | I-Quant | IQ3_S | R9700 | pp2048/tg128/b2048 | 26.95 | 12.28 | 330.93 | `2026-06-30-llamacpp-qwen35-27b-quantfamilies-r9700-memory.jsonl` |
| ok | llama.cpp | Qwen3.5-27B | K-Quant | Q4_K_M | R9700 | pp512/tg128/b2048 | 27.82 | 16.05 | 446.52 | `2026-06-30-llamacpp-qwen35-27b-quantfamilies-r9700-memory.jsonl` |
| ok | llama.cpp | Qwen3.5-27B | K-Quant | Q4_K_M | R9700 | pp2048/tg128/b2048 | 27.82 | 16.05 | 446.52 | `2026-06-30-llamacpp-qwen35-27b-quantfamilies-r9700-memory.jsonl` |
| ok | llama.cpp | Qwen3.5-27B | UD | UD-Q5_K_XL | R9700 | pp512/tg128/b2048 | 24.97 | 19.16 | 478.29 | `2026-06-30-llamacpp-qwen35-27b-quantfamilies-r9700-memory.jsonl` |
| ok | llama.cpp | Qwen3.5-27B | UD | UD-Q5_K_XL | R9700 | pp2048/tg128/b2048 | 24.97 | 19.16 | 478.29 | `2026-06-30-llamacpp-qwen35-27b-quantfamilies-r9700-memory.jsonl` |

## Notes

- The `Decode x GiB` column is `decode tokens/s * consumed VRAM GiB`. It is a reference column, not a standalone quality score.
- Existing earlier rows from this date do not contain memory metrics; use this run or later runs for memory comparisons.
- The local Qwen3-14B-FP8 artifact is not present yet. External-engine R9700 runs should use `Qwen/Qwen3-14B-FP8` after downloading it.

# Memory and quant-family benchmarks

## Done

- Added memory metrics to the benchmark schema: baseline VRAM, peak VRAM, consumed VRAM, and decode token/s x consumed VRAM GiB.
- Updated the benchmark plan so vLLM, SGLang, and ROCm/ATOM are tried on R9700 with Qwen3-14B-FP8 instead of waiting only for MI300X.
- Updated the implementation plan to treat token/s and memory consumption as the baseline pair.
- Added memory sampling to `tools/run-llamacpp-benchmark.py` using `rocm-smi --showmeminfo vram --json`.
- Added `tools/summarize-benchmark-results.py` for Markdown tables with decode token/s, consumed GiB, and decode x GiB.
- Measured Qwen3.5-27B K-Quant, I-Quant, and UD on R9700 with memory.

## Outputs

- `uLLM-project/benchmarks/results/2026-06-30/llama.cpp/2026-06-30-llamacpp-qwen35-27b-quantfamilies-r9700-memory.jsonl`
- `uLLM-project/benchmarks/results/2026-06-30/llama.cpp/quant-family-memory-summary.md`
- `uLLM-project/docs/plans/r9700-qwen3-14b-fp8-external-engine-plan-v0.1.md`

## Notes

- Local Qwen3-14B-FP8 is not downloaded yet. Firecrawl search found the official Hugging Face candidate `Qwen/Qwen3-14B-FP8`.
- Firecrawl search feedback was attempted, but this Firecrawl deployment returned `DB_DISABLED`.

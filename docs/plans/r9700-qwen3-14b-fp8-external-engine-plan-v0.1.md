# R9700 Qwen3-14B-FP8 external engine benchmark plan v0.1

## Purpose

Measure whether vLLM, SGLang, and ROCm/ATOM can run a smaller FP8 workload on the local R9700 before waiting for MI300X.

## Target

- Device: R9700 only
- Model: `Qwen/Qwen3-14B-FP8`
- Engines: vLLM, SGLang, ROCm/ATOM
- First workload: representative single-GPU decode and prefill/decode mix
- Metrics: prefill token/s, decode token/s, total token/s, baseline VRAM, peak VRAM, consumed VRAM, decode token/s x consumed VRAM GiB

## Procedure

1. Download `Qwen/Qwen3-14B-FP8` with `hf download` to `~/datapool/ai_models/safetensors/Qwen/Qwen3-14B-FP8/`.
2. Keep V620 out of the process with `ROCR_VISIBLE_DEVICES=1` for ROCm Python engines. `HIP_VISIBLE_DEVICES=1` can leave AITER seeing a V620 first.
3. Record the exact engine commit, Python environment, PyTorch/ROCm version, and launch command.
4. Run a minimal load test first.
5. Run a representative benchmark: `prompt_tokens=512`, `generated_tokens=128`, `concurrent_requests=1`, `tp=1`, `pp=1`.
6. Record memory with the same baseline/peak/consumed semantics as `docs/specs/inference-benchmark-result-v0.1.md`.
7. Expand to `prompt_tokens=2048` and larger concurrency only after the minimal case is stable.
8. If an engine fails because of unsupported FP8, unsupported gfx target, missing kernel, or runtime assertion, write `failed` or `unsupported` JSONL rows instead of omitting it.

## Initial Commands To Derive

- vLLM: derive from its built-in benchmark or OpenAI-compatible server benchmark.
- SGLang: derive from its built-in benchmark or server benchmark.
- ROCm/ATOM: derive from its documented ROCm benchmark path after confirming the expected model format.

## Notes

- This is not a replacement for MI300X TP/PP testing. It is a local feasibility and baseline pass.
- Qwen3-14B-FP8 is a safetensors/Hugging Face target, not a GGUF comparison target.
- llama.cpp GGUF quant-family results remain separate from vLLM/SGLang/ATOM FP8 results.
- The 2026-06-30 run is recorded in `benchmarks/results/2026-06-30/external-r9700-qwen3-14b-fp8-summary.md`.

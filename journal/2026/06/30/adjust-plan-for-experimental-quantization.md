# Adjust plan for experimental quantization

## Done

- Updated the concept and implementation plan so `aq` and `sq` are not fixed ahead of experiments.
- Added an existing inference engine benchmark phase before Qwen3 implementation.
- Added `docs/specs/inference-benchmark-result-v0.1.md`.
- Added `docs/plans/existing-engine-benchmark-plan-v0.1.md`.
- Added `benchmarks/results/README.md`.
- Reframed `.ullm` `tensors/`, `codebooks/`, and `scales/` as logical storage areas interpreted through manifest metadata.

## Notes

- Stable `aq` / `sq` specs should only be written after candidate variants are benchmarked.
- V620 should run llama.cpp locally, while vLLM, SGLang, ROCm/ATOM, and TensorRT-LLM should be recorded as unsupported on V620 rather than forced.

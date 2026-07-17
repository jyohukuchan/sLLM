# SQ8 plan vLLM FP8 comparison

2026-07-09

- User requested that the later half of the SQ8_0 implementation plan include comparison against `vLLM + FP8`.
- Updated `uLLM-project/docs/plans/sq8-implementation-plan-v0.1.md`.
- Added milestone `M10: vLLM + FP8 External Baseline Comparison`.
- The comparison is intentionally after implementation-valid uLLM SQ8_0 rows exist.
- Primary target is `vLLM + Qwen/Qwen3-14B-FP8` on R9700/RDNA4 with the current working vLLM ROCm environment.
- The plan requires comparable `inference-benchmark-result-v0.1` rows and explicit `unsupported` / `failed` rows if vLLM cannot match the target.
- The plan warns not to compare uLLM selected-layer diagnostics directly against full vLLM serving rows.

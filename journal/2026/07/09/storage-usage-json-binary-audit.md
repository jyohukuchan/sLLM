# Storage usage JSON and binary audit

2026-07-09

- `uLLM-project` total size is about 41GB.
- Dominant directory is `uLLM-project/build/envs` at about 35GB. It is ignored by `.gitignore` through `build/`.
- `build/envs` contains four Python/ROCm environments:
  - `vllm-rocm-nightly`: about 11GB
  - `sglang-rocm`: about 11GB
  - `vllm-rocm`: about 7.4GB
  - `atom-rocm`: about 6.9GB
- Largest files are shared-library and compiled-kernel artifacts inside those envs, not project JSON:
  - `.so`: about 20GB total
  - `.aks2`: about 3.5GB total, mostly `torch/lib/aotriton.images`, about 890MB per env
  - `.pkl`: about 2.3GB total, mainly `aiter/jit/flydsl_cache` in `vllm-rocm-nightly` and `sglang-rocm`
  - `.cubin`: about 1.8GB total, mainly `flashinfer_cubin` in `vllm-rocm`
- The `aiter/jit` trees are large but cache/build-like:
  - `vllm-rocm-nightly/.../aiter/jit`: about 3.1GB
  - `sglang-rocm/.../aiter/jit`: about 3.1GB
  - each has `flydsl_cache` around 1.2GB
- `target/` is about 1.4GB and ignored.
- `build/reference` is about 320MB and ignored through `build/`.
- `reference-src/` is ignored and contains some build/generated or clone metadata:
  - `reference-src/sglang/rust/sglang-grpc/target`: about 897MB
  - `reference-src/sglang/sgl-kernel/build`: about 46MB
  - `reference-src/aiter/aiter/jit/build`: about 46MB
  - reference clone `.git` dirs: `aiter` about 404MB, `tensorrt-llm` about 93MB, `vllm` about 40MB, `llama.cpp` about 35MB
- `benchmarks/results` is about 672MB on disk and about 664MB is Git-tracked. This is mostly historical JSONL/result payloads and should be treated as repository data unless a deliberate archival policy is chosen.
- Excluding `build/envs` and `target`, JSON/JSONL/safetensors/bin/idx4/aq4/gguf style files total about 779MB, so JSON/binary project data is not the main 40GB driver.

Cleanup ranking:

1. Safest high-impact candidate: remove one or more ignored env directories under `build/envs` when not needed for immediate benchmark reproduction.
2. Safe routine cleanup: `cargo clean` or remove ignored `target/` to reclaim about 1.4GB.
3. Safe generated cleanup inside ignored references: remove `reference-src/sglang/rust/sglang-grpc/target` to reclaim about 897MB.
4. Medium caution: remove `aiter/jit/flydsl_cache` under envs to reclaim about 2.4GB, but first-run JIT cost may return.
5. Avoid casual deletion: `benchmarks/results`, because it is mostly tracked experimental history rather than ignored cache.

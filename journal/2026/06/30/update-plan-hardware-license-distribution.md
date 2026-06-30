# Update uLLM plan with hardware, license, and distribution notes

## Done

- Updated the concept document with JAX/TPU, HIP-first hardware rollout, C++20/HIP direct kernels, Rust control plane, speed prediction, and `.ullm` packaging direction.
- Updated the implementation plan with V620/R9700, MI300X, AVX-512 ordering and Qwen3-14B/Qwen3-30B-A3B initial targets.
- Added Qwen3.5 or Gemma4 as early advanced targets.
- Added reference source fetch tooling and ignored `reference-src/`.
- Downloaded llama.cpp, vLLM, SGLang, ATOM, and TensorRT-LLM into `reference-src/`.
- Added ADR 0001 for Apache-2.0 and reference-code policy.
- Added reference source inventory with commit and license notes.

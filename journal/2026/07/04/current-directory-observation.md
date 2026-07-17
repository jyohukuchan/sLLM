# Current directory observation

## Scope

- Current directory: `/home/homelab1/coding-local/ultimateLLM`
- Git-managed project directory: `uLLM-project/`
- Root wrapper contents: `AGENTS.md`, `memo-for-AGENT.md`, `docs/`, `journal/`, `uLLM-project/`

## Repository status

- `uLLM-project/` is on `main`.
- `main` is ahead of `origin/main` by 1 commit: `0221d4f Add 12h golden layer validation plan`.
- Working tree was clean before this observation note.
- The latest ahead commit adds `docs/plans/12h-golden-layer-validation-plan-v0.1.md`, `docs/words.txt`, and `journal/2026/07/04/12h-golden-layer-validation-plan.md`.

## Project shape

- Rust workspace members:
  - `crates/ullm-engine`
  - `crates/ullm-quant`
  - `crates/ullm-runtime-sys`
- C++ runtime boundary:
  - `runtime/include/ullm_runtime.h`
  - `runtime/src/ullm_runtime.cpp`
- Cargo config uses `clang` with `mold`.
- `ullm-engine` currently exposes modules for AQ helpers, package loading, runtime loading, Qwen3/Qwen3.5 decoder paths, scheduler state, and request decode runners.
- `crates/ullm-engine/src/main.rs` remains a large smoke/CLI surface.

## Current technical state

- The recent work moved Qwen3 package model runtime, decoder layer loading, weight construction, passthrough F32 loading, AQ4 materialization, host byte helpers, scheduler decode batching, and request/layer stack runners toward reusable library modules.
- Existing verified local package path: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d`
- That package is present and is about 8.5G.
- Current next plan is not full prompt generation. The latest plan proposes golden tensor fixture validation for one Qwen3.5 layer before moving to end-to-end generation.

## Environment observations

- `cargo fmt --all --check` passed.
- `cargo check --workspace` passed.
- `rustc`: 1.96.0
- `cargo`: 1.96.0
- `clang`: Ubuntu clang 18.1.3
- `mold`: 2.30.0
- `hipcc`: HIP 7.2.53211 / ROCm 7.2.1
- Runtime device discovery:
  - device 0: CPU fallback
  - device 1: AMD Radeon Pro V620
  - device 2: AMD Radeon Graphics, compute 12.0
  - device 3: AMD Radeon Pro V620
- Default `python3` has CPU-only PyTorch 2.12.0 and Transformers 5.12.1.
- Existing ROCm venvs under `build/envs/*-rocm` have PyTorch `2.11.0+gitd0c8b1f`, HIP `7.2.53211`, `torch.cuda.is_available() == True`, and Transformers installed.

## Smoke check

Command run:

```bash
./target/debug/ullm-engine package-self-attn-mlp-block-model-loop-smoke \
  /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d \
  0 1048576 3,7,11 3
```

Result:

- Passed on CPU fallback.
- `decode_batch_ready_counts=[2, 1]`
- `cached_tokens=[3, 3, 1]`
- `generated_tokens=[2, 1, 0]`
- all reported max diffs were `0`
- `verified=true`

## Practical next step

Follow `docs/plans/12h-golden-layer-validation-plan-v0.1.md`: create a small golden tensor fixture exporter and a `package-layer-golden-smoke` path that compares one package-loaded Qwen3.5 decoder layer against reference hidden states before attempting full prompt generation.

# R9700 Qwen3-14B-FP8 External Engine Summary - 2026-06-30

## Scope

- Device: R9700 / gfx1201, selected with `ROCR_VISIBLE_DEVICES=1`
- Model: `Qwen/Qwen3-14B-FP8`
- Local model path: `~/datapool/ai_models/safetensors/Qwen/Qwen3-14B-FP8/`
- Download command used: `hf download Qwen/Qwen3-14B-FP8 --local-dir ~/datapool/ai_models/safetensors/Qwen/Qwen3-14B-FP8`
- Engines: vLLM, SGLang, ROCm/ATOM
- Primary representative workload: `prompt_tokens=512`, `generated_tokens=128`, `tp=1`, `pp=1`, one request
- Memory: `rocm-smi --showmeminfo vram --json`, peak total used VRAM minus pre-command total used VRAM

## Result Table

| Status | Engine | Model | Family | Quant | Target | Workload | Decode tok/s | Consumed GiB | Decode x GiB | Source |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| ok | ATOM | Qwen3-14B-FP8 | FP8 | FP8 | R9700 | pp16/tg8/b1 | 11.86 | 24.11 | 285.98 | `2026-06-30-atom-r9700-qwen3-14b-fp8.jsonl` |
| ok | ATOM | Qwen3-14B-FP8 | FP8 | FP8 | R9700 | pp512/tg128/b1 | 18.27 | 24.30 | 444.07 | `2026-06-30-atom-r9700-qwen3-14b-fp8.jsonl` |
| ok | SGLang | Qwen3-14B-FP8 | FP8 | FP8 | R9700 | pp16/tg8/b1 | 19.12 | 16.75 | 320.37 | `2026-06-30-sglang-r9700-qwen3-14b-fp8.jsonl` |
| ok | SGLang | Qwen3-14B-FP8 | FP8 | FP8 | R9700 | pp512/tg128/b1 | 24.99 | 16.81 | 420.12 | `2026-06-30-sglang-r9700-qwen3-14b-fp8.jsonl` |
| ok | vLLM | Qwen3-14B-FP8 | FP8 | FP8 | R9700 | pp16/tg8/b1 | 5.98 | 28.72 | 171.75 | `2026-06-30-vllm-r9700-qwen3-14b-fp8.jsonl` |
| ok | vLLM | Qwen3-14B-FP8 | FP8 | FP8 | R9700 | pp512/tg128/b1 | 23.67 | 28.72 | 679.79 | `2026-06-30-vllm-r9700-qwen3-14b-fp8.jsonl` |

Earlier failed compatibility attempts remain in the raw JSONL files. This table lists the usable rows.

## Representative Rows

| Engine | Status | Workload | Prefill tok/s | Decode tok/s | Total tok/s | Consumed GiB | Decode x GiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| vLLM | ok | pp512/tg128/tp1/pp1 | 94.66 | 23.67 | 118.33 | 28.72 | 679.79 |
| SGLang | ok | pp512/tg128/tp1/pp1 | 49.50 | 24.99 | 74.49 | 16.81 | 420.12 |
| ATOM | ok | pp512/tg128/tp1/pp1 | 73.09 | 18.27 | 91.37 | 24.30 | 444.07 |

## Environment Notes

- vLLM success used `build/envs/vllm-rocm-nightly`, `vllm==0.23.1rc1.dev618+g8cf7c4d8a.rocm723`, torch `2.11.0+gitd0c8b1f`, ROCm runtime `7.2.1`.
- vLLM failed under `HIP_VISIBLE_DEVICES=1` because AITER still selected a V620/gfx1030 path. `ROCR_VISIBLE_DEVICES=1` fixed the runtime selection.
- SGLang used source commit `3add35e26dc0623d6647e226de7d17754bb61804` with a local ignored source patch:
  - `sgl-kernel/setup_rocm.py` was widened to allow experimental `gfx1201`.
  - `python/sglang/srt/layers/layernorm.py` was adjusted to call the installed vLLM ROCm `fused_add_rms_norm(input, residual, weight, eps)` ABI.
- ATOM used source commit `cce1a6e56dcd8cb300183f81901fdaed6090d951` in `build/envs/atom-rocm`.
- ATOM failed with wheel `amd-aiter==0.1.16.post2`; that wheel lacked `aiter.ops.shuffle.moe_shuffle_scale` and ModelRunner later exited during warmup.
- AITER was then cloned to `reference-src/aiter` and installed into `build/envs/atom-rocm` as editable source:
  - AITER commit `71829a74bc2600bfbce4c05f85ecbe0eeb994323`
  - version `amd-aiter==0.1.17.dev155+g71829a74b`
  - install mode: `AITER_USE_SYSTEM_TRITON=1 BUILD_TARGET=rocm GPU_ARCHS=gfx1201 PREBUILD_KERNELS=0`
  - `PREBUILD_KERNELS=1` was not used because the FlyDSL AOT path started compiling a large `gfx950` set despite `GPU_ARCHS=gfx1201`.
- The first ATOM source AITER representative row used `--enforce-eager`, `--block-size 64`, `--kv_cache_dtype bf16`, `--max-num-seqs 1`, and `--max-num-batched-tokens 640`. A follow-up row with the same pp512/tg128 workload removed `--enforce-eager` and improved the wrapper throughput from `9.15` to `18.27` tok/s.
- ATOM `benchmark_serving` requires percentile `99` when saving the PyTorch benchmark sidecar; `50,95` alone produced a post-benchmark `KeyError: 'p99_ttft_ms'` even though the request completed.
- ATOM official recipe comparisons should use TPOT-derived speed, `1000 / mean_tpot_ms`, not the wrapper `output_throughput`. The local Qwen3-8B-FP8 official-like CUDAGraph run reached `55.65` TPOT-derived tok/s, matching the official 52.9-class result. Details are in `atom-qwen3-fp8-cause-analysis.md`.

## Interpretation

- vLLM is currently the best R9700 FP8 baseline by `decode tok/s x consumed GiB` for the representative single-request row, but it consumes about 28.72 GiB.
- SGLang reaches the highest representative decode tok/s with much lower consumed VRAM than vLLM, but only after local compatibility patches.
- ATOM becomes runnable on R9700/gfx1201 after replacing the wheel AITER with source AITER HEAD. Without `--enforce-eager`, the single-request representative row is still slower than vLLM and SGLang, but the gap is smaller than the first eager row suggested.

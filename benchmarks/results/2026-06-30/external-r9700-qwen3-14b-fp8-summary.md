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
| failed | ATOM | Qwen3-14B-FP8 | FP8 | FP8 | R9700 | pp16/tg8/b1 | - | 16.13 | - | `2026-06-30-atom-r9700-qwen3-14b-fp8.jsonl` |
| failed | ATOM | Qwen3-14B-FP8 | FP8 | FP8 | R9700 | pp16/tg8/b1 | - | 16.13 | - | `2026-06-30-atom-r9700-qwen3-14b-fp8.jsonl` |
| failed | ATOM | Qwen3-14B-FP8 | FP8 | FP8 | R9700 | pp16/tg8/b1 | - | 15.66 | - | `2026-06-30-atom-r9700-qwen3-14b-fp8.jsonl` |
| failed | SGLang | Qwen3-14B-FP8 | FP8 | FP8 | R9700 | pp16/tg8/b1 | - | 16.58 | - | `2026-06-30-sglang-r9700-qwen3-14b-fp8.jsonl` |
| failed | SGLang | Qwen3-14B-FP8 | FP8 | FP8 | R9700 | pp16/tg8/b1 | - | 16.59 | - | `2026-06-30-sglang-r9700-qwen3-14b-fp8.jsonl` |
| ok | SGLang | Qwen3-14B-FP8 | FP8 | FP8 | R9700 | pp16/tg8/b1 | 19.12 | 16.75 | 320.37 | `2026-06-30-sglang-r9700-qwen3-14b-fp8.jsonl` |
| ok | SGLang | Qwen3-14B-FP8 | FP8 | FP8 | R9700 | pp512/tg128/b1 | 24.99 | 16.81 | 420.12 | `2026-06-30-sglang-r9700-qwen3-14b-fp8.jsonl` |
| unsupported | vLLM | Qwen3-14B-FP8 | FP8 | FP8 | R9700 | pp16/tg8/b1 | - | 0.00 | - | `2026-06-30-vllm-r9700-qwen3-14b-fp8.jsonl` |
| failed | vLLM | Qwen3-14B-FP8 | FP8 | FP8 | R9700 | pp16/tg8/b1 | - | 0.00 | - | `2026-06-30-vllm-r9700-qwen3-14b-fp8.jsonl` |
| ok | vLLM | Qwen3-14B-FP8 | FP8 | FP8 | R9700 | pp16/tg8/b1 | 5.98 | 28.72 | 171.75 | `2026-06-30-vllm-r9700-qwen3-14b-fp8.jsonl` |
| ok | vLLM | Qwen3-14B-FP8 | FP8 | FP8 | R9700 | pp512/tg128/b1 | 23.67 | 28.72 | 679.79 | `2026-06-30-vllm-r9700-qwen3-14b-fp8.jsonl` |

## Representative Rows

| Engine | Status | Workload | Prefill tok/s | Decode tok/s | Total tok/s | Consumed GiB | Decode x GiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| vLLM | ok | pp512/tg128/tp1/pp1 | 94.66 | 23.67 | 118.33 | 28.72 | 679.79 |
| SGLang | ok | pp512/tg128/tp1/pp1 | 49.50 | 24.99 | 74.49 | 16.81 | 420.12 |
| ATOM | failed | smoke only | - | - | - | 15.66-16.13 | - |

## Environment Notes

- vLLM success used `build/envs/vllm-rocm-nightly`, `vllm==0.23.1rc1.dev618+g8cf7c4d8a.rocm723`, torch `2.11.0+gitd0c8b1f`, ROCm runtime `7.2.1`.
- vLLM failed under `HIP_VISIBLE_DEVICES=1` because AITER still selected a V620/gfx1030 path. `ROCR_VISIBLE_DEVICES=1` fixed the runtime selection.
- SGLang used source commit `3add35e26dc0623d6647e226de7d17754bb61804` with a local ignored source patch:
  - `sgl-kernel/setup_rocm.py` was widened to allow experimental `gfx1201`.
  - `python/sglang/srt/layers/layernorm.py` was adjusted to call the installed vLLM ROCm `fused_add_rms_norm(input, residual, weight, eps)` ABI.
- ATOM used source commit `cce1a6e56dcd8cb300183f81901fdaed6090d951` in `build/envs/atom-rocm`.
- ATOM required a local ignored fallback import because the installed `amd-aiter==0.1.16.post2` wheel lacks `aiter.ops.shuffle.moe_shuffle_scale`; `shuffle_scale` was aliased for this dense Qwen3 smoke attempt.
- ATOM still failed before readiness with ModelRunner `exitcode=-11` during warmup, even with `--enforce-eager`, `--max-model-len 256`, `--max-num-batched-tokens 256`, and `--max-num-seqs 1`.
- ATOM's Dockerfile builds AITER from GitHub HEAD. A proper ATOM rerun should build AITER from source for `gfx1201` before treating this as an engine limitation.

## Interpretation

- vLLM is currently the best R9700 FP8 baseline by `decode tok/s x consumed GiB` for the representative single-request row, but it consumes about 28.72 GiB.
- SGLang reaches slightly higher representative decode tok/s with much lower consumed VRAM, but only after local compatibility patches.
- ATOM is not runnable for this exact local 14B FP8 setup with the wheel-based AITER environment. The failure is recorded rather than omitted.

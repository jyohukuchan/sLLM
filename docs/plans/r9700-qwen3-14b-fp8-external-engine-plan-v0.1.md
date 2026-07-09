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

## Derived vLLM Command Template

Use `ROCR_VISIBLE_DEVICES=1` for the current WRX80 host. `HIP_VISIBLE_DEVICES=1` can leave AITER
seeing a V620/gfx1030 path first and should only be used as an intentional failure probe.

Set shared paths once:

```text
RESULT_ROOT=benchmarks/results/$(date +%F)/sq8-vllm-fp8-comparison
MODEL=/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3-14B-FP8
VLLM_BIN=build/envs/vllm-rocm-nightly/bin/vllm
mkdir -p "${RESULT_ROOT}/logs"
```

Smoke row:

```text
ROCR_VISIBLE_DEVICES=1 VLLM_LOGGING_LEVEL=INFO python3 tools/run-external-benchmark.py \
  --run-id sq8-vllm-fp8-r9700 \
  --case-id vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-tp1-rocr \
  --output-jsonl "${RESULT_ROOT}/results.jsonl" \
  --stdout-log "${RESULT_ROOT}/logs/vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-tp1-rocr.stdout.log" \
  --stderr-log "${RESULT_ROOT}/logs/vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-tp1-rocr.stderr.log" \
  --memory-log "${RESULT_ROOT}/logs/vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-tp1-rocr.memory.jsonl" \
  --engine-name vLLM \
  --engine-version 0.23.1rc1.dev618+g8cf7c4d8a.rocm723 \
  --engine-commit 8cf7c4d8ad602d73ff2ec72a101420d47163c136 \
  --model-name Qwen3-14B-FP8 \
  --model-format safetensors \
  --model-quantization FP8 \
  --gpu-card card2 \
  --context-length 256 \
  --prompt-tokens 16 \
  --generated-tokens 8 \
  --batch-size 1 \
  --concurrent-requests 1 \
  --kv-cache-dtype auto \
  --parse vllm-throughput \
  --result-json "${RESULT_ROOT}/logs/vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-tp1-rocr.result.json" \
  --note "M10 vLLM FP8 smoke baseline for SQ8_0 comparison" \
  -- \
  "${VLLM_BIN}" bench throughput \
    --backend vllm \
    --model "${MODEL}" \
    --tokenizer "${MODEL}" \
    --num-prompts 1 \
    --random-input-len 16 \
    --random-output-len 8 \
    --max-model-len 256 \
    --dtype auto \
    --kv-cache-dtype auto \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --output-json "${RESULT_ROOT}/logs/vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-tp1-rocr.result.json"
```

Representative row:

```text
ROCR_VISIBLE_DEVICES=1 VLLM_LOGGING_LEVEL=INFO python3 tools/run-external-benchmark.py \
  --run-id sq8-vllm-fp8-r9700 \
  --case-id vllm-r9700-qwen3-14b-fp8-rep-pp512-tg128-tp1-rocr \
  --output-jsonl "${RESULT_ROOT}/results.jsonl" \
  --stdout-log "${RESULT_ROOT}/logs/vllm-r9700-qwen3-14b-fp8-rep-pp512-tg128-tp1-rocr.stdout.log" \
  --stderr-log "${RESULT_ROOT}/logs/vllm-r9700-qwen3-14b-fp8-rep-pp512-tg128-tp1-rocr.stderr.log" \
  --memory-log "${RESULT_ROOT}/logs/vllm-r9700-qwen3-14b-fp8-rep-pp512-tg128-tp1-rocr.memory.jsonl" \
  --engine-name vLLM \
  --engine-version 0.23.1rc1.dev618+g8cf7c4d8a.rocm723 \
  --engine-commit 8cf7c4d8ad602d73ff2ec72a101420d47163c136 \
  --model-name Qwen3-14B-FP8 \
  --model-format safetensors \
  --model-quantization FP8 \
  --gpu-card card2 \
  --context-length 4096 \
  --prompt-tokens 512 \
  --generated-tokens 128 \
  --batch-size 1 \
  --concurrent-requests 1 \
  --kv-cache-dtype auto \
  --parse vllm-throughput \
  --result-json "${RESULT_ROOT}/logs/vllm-r9700-qwen3-14b-fp8-rep-pp512-tg128-tp1-rocr.result.json" \
  --note "M10 vLLM FP8 representative baseline for SQ8_0 comparison" \
  -- \
  "${VLLM_BIN}" bench throughput \
    --backend vllm \
    --model "${MODEL}" \
    --tokenizer "${MODEL}" \
    --num-prompts 1 \
    --num-warmups 1 \
    --random-input-len 512 \
    --random-output-len 128 \
    --max-model-len 4096 \
    --dtype auto \
    --kv-cache-dtype auto \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --output-json "${RESULT_ROOT}/logs/vllm-r9700-qwen3-14b-fp8-rep-pp512-tg128-tp1-rocr.result.json"
```

These rows are not comparable with selected-layer uLLM diagnostics. Use them only after uLLM has a
full-package SQ8_0 row with matching prompt length, generated length, concurrency, KV dtype, and
execution-mode metadata.

## Current uLLM SQ8_0 Smoke-Shape Row

A uLLM SQ8_0 row with the smoke workload shape is available at:

```text
benchmarks/results/2026-07-09/sq8-vllm-fp8-comparison/results.jsonl
```

It uses `prompt_tokens=16`, `generated_tokens=8`, `concurrent_requests=1`, `kv_cache_dtype=f32`,
and records direct SQ8_0 `single+triple` projection execution plus the attached behavioral
prompt-suite guard bundle. This is a measurement-path row for Qwen3.5-9B, not a same-model row for
the Qwen3-14B-FP8 vLLM target.

A matching-shape vLLM row has also been recorded in the same JSONL:

```text
case_id: vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-tp1-rocr
status: ok
prefill tok/s: 31.185949962897716
decode tok/s: 15.59
total tok/s: 46.78
consumed VRAM: 30830026752 bytes
```

This confirms local vLLM FP8 smoke execution on R9700. It still should not be interpreted as a
same-model uLLM-vs-vLLM performance conclusion.

Matching b2, b4, and b8 vLLM rows have also been recorded for the uLLM real-batch no-final-logits
diagnostics:

```text
case_id: vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b2-tp1-rocr
status: ok
prefill tok/s: 34.41438620647337
decode tok/s: 17.21
total tok/s: 51.62
consumed VRAM: 21007855616 bytes

case_id: vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b4-tp1-rocr
status: ok
prefill tok/s: 135.04146895989985
decode tok/s: 67.52
total tok/s: 202.56
consumed VRAM: 30121553920 bytes

case_id: vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b8-tp1-rocr
status: ok
prefill tok/s: 236.01404374447745
decode tok/s: 118.01
total tok/s: 354.02
consumed VRAM: 30121566208 bytes
```

These match the `prompt_tokens=16x2` / `generated_tokens=8x2` / `concurrent_requests=2` and
`prompt_tokens=16x4` / `generated_tokens=8x4` / `concurrent_requests=4` and
`prompt_tokens=16x8` / `generated_tokens=8x8` / `concurrent_requests=8` shapes used by the uLLM
SQ8_0 no-final-logits real-batch diagnostics. The harness still differs: vLLM uses
`vllm bench throughput`, while uLLM currently uses CLI model-loop diagnostics.

A representative vLLM row has also been recorded:

```text
case_id: vllm-r9700-qwen3-14b-fp8-rep-pp512-tg128-tp1-rocr
status: ok
prefill tok/s: 90.17614034497254
decode tok/s: 22.54
total tok/s: 112.72
consumed VRAM: 30837428224 bytes
```

## Notes

- This is not a replacement for MI300X TP/PP testing. It is a local feasibility and baseline pass.
- Qwen3-14B-FP8 is a safetensors/Hugging Face target, not a GGUF comparison target.
- llama.cpp GGUF quant-family results remain separate from vLLM/SGLang/ATOM FP8 results.
- The 2026-06-30 run is recorded in `benchmarks/results/2026-06-30/external-r9700-qwen3-14b-fp8-summary.md`.

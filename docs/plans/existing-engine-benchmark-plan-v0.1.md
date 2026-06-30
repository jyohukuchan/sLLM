# Existing engine benchmark plan v0.1

## Purpose

Before implementing Qwen3 in uLLM or stabilizing `aq` / `sq`, measure existing inference engines under controlled conditions. This phase defines the target baseline for uLLM.

## Engines

| Engine | V620/R9700 status | Initial role |
| --- | --- | --- |
| llama.cpp | run locally | primary V620/R9700 baseline |
| vLLM | do not force on V620; try R9700 with Qwen3-14B-FP8 | R9700 smoke/representative baseline, MI300X later |
| SGLang | do not force on V620; try R9700 with Qwen3-14B-FP8 | R9700 smoke/representative baseline, MI300X later |
| ROCm/ATOM | do not force on V620; try R9700 with Qwen3-14B-FP8 | R9700 smoke/representative baseline, MI300X later |
| TensorRT-LLM | not for AMD GPUs | harness and unsupported records now, NVIDIA later |

## Measurement Grid

Start small and expand only after scripts are stable.

Initial llama.cpp grid:

- context length: `2048`, `4096`, `8192`, `16384`
- prompt tokens: `128`, `512`, `2048`
- generated tokens: `128`, `512`
- batch size: `1`, `4`, `8` where supported
- GPU count: `1`, `2`, `3` where supported
- KV cache dtype: `f16`

Quantization-family sweep:

- K-Quant: run separately, for example `Q4_K_M`, `Q5_K_M`, `Q6_K`.
- I-Quant: run separately, for example `IQ3_S` or later chosen IQ variants.
- UD: run separately, using Unsloth Dynamic GGUF artifacts.
- FP8: run separately where the engine can load the model.

Do not merge I-Quant, K-Quant, and UD rows into the same summary bucket. They must be separate rows and separate comparison groups.

R9700 early external-engine grid:

- engine: vLLM, SGLang, ROCm/ATOM
- device: R9700 only
- model: Qwen3-14B-FP8, official Hugging Face artifact if possible
- tensor parallelism: `1`
- pipeline parallelism: `1`
- prompt/generated tokens: representative first, then expand
- status: `ok`, `unsupported`, `failed`, or `oom`; do not omit failed setup attempts

Future MI300X grid:

- tensor parallelism: `1`, `2`, `4`, `8`
- pipeline parallelism: `1`, `2`
- concurrent requests: `1`, `4`, `16`, `64`
- context length: up to hardware limit

## Output

Write JSONL records matching `docs/specs/inference-benchmark-result-v0.1.md`.

Recommended paths:

```text
benchmarks/results/YYYY-MM-DD/<engine>/<run_id>.jsonl
benchmarks/results/YYYY-MM-DD/<engine>/logs/
```

## Procedure

1. Record hardware and compiler environment.
2. Record engine commit and build flags.
3. Select model artifact and quantization.
4. Run one warmup case.
5. Run the grid.
6. Store each case as one JSONL row.
7. Store unsupported cases explicitly.
8. Record memory baseline, peak, and consumed VRAM for every throughput run.
9. Summarize prefill, decode, total token/s, consumed VRAM, `decode token/s * consumed VRAM GiB`, and failure reason.

## V620 Rule

On V620, do not spend time forcing vLLM, SGLang, ROCm/ATOM, or TensorRT-LLM to run. Record them as unsupported for this hardware generation and proceed with llama.cpp plus uLLM HIP experiments.

## Done Criteria

- llama.cpp produces valid JSONL benchmark rows.
- Unsupported rows exist for vLLM, SGLang, ROCm/ATOM, and TensorRT-LLM on V620.
- At least one context-length sweep exists.
- At least one generated-token sweep exists.
- Memory consumption is recorded as baseline, peak, and consumed VRAM.
- Summary tables include decode token/s, consumed VRAM GiB, and decode token/s x consumed VRAM GiB.
- I-Quant, K-Quant, and UD results are split into separate comparison groups.
- Results are sufficient to define the first uLLM throughput target.

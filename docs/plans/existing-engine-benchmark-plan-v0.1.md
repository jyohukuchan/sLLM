# Existing engine benchmark plan v0.1

## Purpose

Before implementing Qwen3 in uLLM or stabilizing `aq` / `sq`, measure existing inference engines under controlled conditions. This phase defines the target baseline for uLLM.

## Engines

| Engine | V620/R9700 status | Initial role |
| --- | --- | --- |
| llama.cpp | run locally | primary V620/R9700 baseline |
| vLLM | do not force on V620; try R9700 with Qwen3-14B-FP8 | R9700 smoke/representative baseline, MI300X later |
| SGLang | do not force on V620; try R9700 with Qwen3-14B-FP8 | R9700 smoke/representative baseline, MI300X later |
| ROCm/ATOM | do not force on V620; try R9700 with Qwen3-14B-FP8 | R9700 smoke/representative baseline, MI300X later; see [AMD ATOM MI300X feasibility research (2026-07-26)](../research/amd-atom-mi300x-feasibility-2026-07-26.md) before allocating rental time |
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

For the MI300X×1 command-level extension—uLLM A′ ordering, same-source FP8 vLLM/SGLang rows, Q8_0 GGUF llama.cpp qualification, and the `1..128` concurrency sweep—use [mi300x-external-engine-benchmark-plan-v0.1.md](mi300x-external-engine-benchmark-plan-v0.1.md). That document is authoritative for the single-GPU rental run; this parent plan retains the broader R9700 and future multi-GPU grid.

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

## R9700 controlled external baseline (2026-07-26)

The first controlled R9700 decode position check is complete.  It used only
the gfx1201 R9700, one stream, five unprofiled repetitions, 16 M=1 decode
steps per repetition, and cache depth 1028 -> 1044 (midpoint 1036).

| engine | weight format | KV | timing method | decode tok/s |
| --- | --- | --- | --- | ---: |
| uLLM Phase 0 reference | SQ8_0 (~8.0 bpp) | F32 | selected decode region | 15.294956 |
| llama.cpp 68a5592 | GGUF Q8_0 (~8.5 bpp) | F32 | llama-bench decode-only | 30.468075 |
| llama.cpp 68a5592 | GGUF Q8_0 (~8.5 bpp) | F16 | llama-bench decode-only | 34.885347 |
| vLLM 0.21.0+rocm722 | Qwen3-14B-FP8 | auto (resolved dtype unconfirmed) | client SSE steady output | 15.455471 |
| SGLang v0.5.15.post1-rocm720-mi30x | Qwen3-14B-FP8 | BF16 requested | startup failed | n/a |

The llama.cpp source and documentation were inspected before selecting its
method: -p is prompt-only, -n is generation-only, -pg adds a combined
prompt-plus-generation row, -r repeats a selected row, and -d prefills/restores
the KV state before the timed region.  Therefore -p 0 -n 16 -d 1028 -r 5
matches the target decode shape; -pg would have included prompt work.  Both
llama.cpp rows had all layers requested on the R9700 and flash attention on.

The official fixed GGUF is
Qwen/Qwen3-14B-GGUF@530227a7d994db8eca5ab5ced2fb692b614357fd,
Qwen3-14B-Q8_0.gguf, SHA-256
a0dfe649137410b7d82f06a209240508e218f32f5b6fd81b69d6932160cfcd9d.
Its Hub metadata and the direct FP8 source both declare
base_model:Qwen/Qwen3-14B.  Q8_0 and SQ8_0 are different formats, so this is
speed positioning only, not a quality comparison.

vLLM did start on gfx1201 and five requests each reported 1028 prompt plus 16
completion tokens.  Its rate includes server scheduling and local HTTP/SSE
overhead, unlike the uLLM/llama.cpp decode-loop timers.  SGLang loaded the FP8
checkpoint and allocated BF16 KV, then segfaulted in
sgl_kernel.elementwise.rotary_embedding during decode CUDA-graph capture; the
standard bounded attempt stopped there rather than forcing a fallback.

The complete raw data, commands, per-repeat statistics, R9700-only validation,
thermal histories, image identities, and failure log are in
[r9700-external-engine-baseline](../../benchmarks/results/2026-07-26/r9700-external-engine-baseline/).
ullm-openai.service was stopped and restored in one 22 min 47 s isolation
window; llama-qwen35-udq4.service stayed inactive/disabled.

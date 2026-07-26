# R9700 external-engine decode baseline

Date: 2026-07-26
Device: AMD SMI GPU 2, 0000:47:00.0, Sapphire Radeon AI PRO R9700 32 GB,
gfx1201, 64 CU, ROCm 7.2 / amdgpu 6.16.13.

This is a **speed-only** comparison. Q8_0 and SQ8_0 are different weight
encodings, so no quality claim follows from these measurements:

- llama.cpp GGUF Q8_0: int8 values plus one FP16 scale per 32 values,
  approximately 8.5 bits/weight.
- uLLM SQ8_0: OCP E4M3FN with [128,128] BF16 block scales,
  approximately 8.0 bits/weight.

Both are approximately 8-bit and memory-bound, making this a useful throughput
position check, but it is not an accuracy comparison.

## Result

All successful rows are one stream, five unprofiled repetitions, 16 generated
tokens per repetition, and logical decode context 1028 -> 1044 (midpoint 1036).
"Mean decode" is total generated tokens divided by total timed duration; p50 and
variance are from five per-repeat rates.

| engine | weights / format | KV | timer | mean decode tok/s | p50 tok/s | sample variance (tok/s)^2 | flash attention | thermal process envelope (hotspot C / GFX MHz / socket W) |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| uLLM Phase 0 reference | SQ8_0, ~8.0 bpp | F32 | selected M=1 decode region | **15.294956** | 15.308831 | 0.002970 | handwritten F32 attention path | reference before/load/after: 37/1015/16, **73/3298/250**, 66/3439/123 |
| llama.cpp 68a5592 | GGUF Q8_0, ~8.5 bpp | F32 K + F32 V | llama-bench decode-only | **30.468075** | 31.089355 | 1.264832 | on (flash_attn=1) | 36–61 / 4–3391 / 8–288; start 37/41/13, end 42/41/16 |
| llama.cpp 68a5592 | GGUF Q8_0, ~8.5 bpp | F16 K + F16 V | llama-bench decode-only | **34.885347** | 35.053291 | 0.250263 | on (flash_attn=1) | 39–61 / 41–3460 / 12–263; start 40/41/15, end 46/41/13 |
| vLLM 0.21.0+rocm722 | source Qwen3-14B-FP8 / E4M3, ~8.0 bpp | auto (resolved dtype **unconfirmed**) | client SSE inter-token intervals | **15.455471** | 15.443856 | 0.035319 | ROCM_ATTN; per-op flash status unconfirmed | 40–59 / 6–3404 / 11–329; status samples include UNTHROTTLED and THROTTLED |
| SGLang v0.5.15.post1-rocm720-mi30x | source Qwen3-14B-FP8 | requested BF16 | no decode result | — | — | — | AITER default backend | failed during CUDA-graph capture |

The llama.cpp F32 condition-aligned row is 1.992034x uLLM's throughput (uLLM
is 50.20% of that row). Its F16 practical row is 2.280840x (uLLM is 43.84%).
vLLM's client-visible SSE rate is 1.010495x uLLM's rate, but that is **not** a
kernel-only equality: it includes server scheduling and local HTTP/SSE overhead,
while uLLM and llama.cpp report decode-loop timing.

The external process envelopes are cooler than the uLLM sampled-load reading.
They include model load, warm-up, and idle samples and are not synchronized to
each roughly 0.46–0.56 s llama.cpp timed sample. They record the thermal
difference, but do not establish a temperature-normalized causal comparison.
The external start-hotspot readings were 37 C (llama F32), 40 C (llama F16),
and 40 C (vLLM), so their starts were cooled and close but not identical.
An AMD SMI THROTTLED string is retained as sampled status only; no cause is
inferred from it.

## Artifact identity and base-model check

The downloaded llama.cpp artifact is the official Qwen/Qwen3-14B-GGUF file:

| field | value |
| --- | --- |
| fixed repository revision | 530227a7d994db8eca5ab5ced2fb692b614357fd |
| file | /home/homelab1/datapool/ai_models/gguf/Qwen/Qwen3-14B-GGUF-530227a7/Qwen3-14B-Q8_0.gguf |
| SHA-256 | a0dfe649137410b7d82f06a209240508e218f32f5b6fd81b69d6932160cfcd9d |
| size | 15,698,533,728 B |
| GGUF identity | qwen3, 40 layers, 14B, Q8_0 |
| uLLM/vLLM source artifact revision | Qwen/Qwen3-14B-FP8@9a283b4a5efbc09ce247e0ae5b02b744739e525a |

Both Hub records declare base_model:Qwen/Qwen3-14B; the GGUF metadata also
identifies qwen3 / 14B. This verifies the same base model, not the same
quantized weights. Raw Hub metadata, GGUF dump, download revision, file size,
and SHA-256 are in [environment/](environment/).

## Why llama-bench was used

The README and source at the measured commit were inspected and copied into
[environment/llama-bench-readme-excerpt.txt](environment/llama-bench-readme-excerpt.txt)
and [environment/llama-bench-source-excerpt.cpp](environment/llama-bench-source-excerpt.cpp).

- -p makes a prompt-processing-only test; -n makes a text-generation-only test;
  -pg pp,tg adds one combined prompt-plus-generation test; and -r repeats every
  selected test.
- -d 1028 builds/restores a 1,028-token KV state outside the timer. The source
  starts t_start only after depth handling. test_gen calls llama_decode with
  batch=1 and llama_synchronize once per token.
- The adopted -p 0 -n 16 -d 1028 -r 5 therefore times exactly 16 synchronized
  M=1 decode steps with the target cache window. It excludes depth prefill,
  tokenization, sampling, and model load; the built-in warm-up is outside the
  five samples. -pg was intentionally not used because it includes prompt
  processing in the timed row.

Complete commands are in raw/*/command.txt. llama.cpp was built for gfx1201
from 68a5592c10666d4d89b8480b5b9e8f8068b2f64c, with -ngl 999 and -fa on.
Its benchmark parser did not accept f32 for -ctk/-ctv, despite runtime support.
A minimal local parser mapping (f32 -> GGML_TYPE_F32) was applied only to the
external /home/homelab1/llama.cpp-src checkout and rebuilt; its diff/build output
is in environment/llama-bench-f32-parser-*. It is not a change to this
repository or to uLLM.

llama-bench depth prefill uses synthetic random token IDs, while the vLLM
fallback used real plain text tokenized to exactly 1,028 Qwen tokens. Both
match cache length and batch shape, but token contents are not identical.

## vLLM and SGLang bounded attempts

Containers received only /dev/kfd and /dev/dri/renderD129 (the R9700), with
container HIP_VISIBLE_DEVICES=0. Preflight in both images reported one gfx1201
device; no V620 node was passed. The host llama.cpp runner used
HIP_VISIBLE_DEVICES=1 and validates gfx1201 with no gfx1030 before every model
run.

vLLM used official image
vllm/vllm-openai-rocm@sha256:98a77b20df03adeb1cfc0ced009b4df6dd52b0a994ab99a32421f30876a9ae0c.
The direct FP8 checkpoint started with TP=1, context 1044, max sequences 1,
and max batched tokens 1044. It selected ROCM_ATTN and warned that R9700 W8A8
block-FP8 tuning JSON was absent, so default configs may be sub-optimal. One
warm-up plus five streaming requests succeeded; each reported prompt_tokens=1028,
completion_tokens=16, total_tokens=1044. vLLM did not emit the resolved auto KV
allocation dtype, so it remains unconfirmed. Prefix caching was enabled by
default. The metric excludes TTFT but remains client-visible rather than a
direct kernel timer.

SGLang used official image
lmsysorg/sglang-rocm:v0.5.15.post1-rocm720-mi30x-20260724@sha256:598cc3417792a9e182516bdd835181e3e53254e505747d664fd24ee57f527204.
It loaded the same FP8 checkpoint and allocated BF16 KV for 1,044 tokens. Its
default AITER path rebuilt modules for gfx1201, then segfaulted during decode
CUDA-graph capture in sgl_kernel.elementwise.rotary_embedding; the launcher
reported RuntimeError: Rank 0 scheduler died during initialization (exit code -11).
The standard configuration was bounded; the second launch retained a
complete final log after the first --rm container exited. No AITER/graph
fallback or further forcing was attempted. See
[attempts/sglang-r9700-start-default-logcapture-rerun/logs-final.txt](attempts/sglang-r9700-start-default-logcapture-rerun/logs-final.txt).

## Isolation, service lifecycle, and evidence layout

ullm-openai.service was stopped once at 2026-07-26T13:47:11+09:00 for the
single isolated window and restored successfully at 2026-07-26T14:09:58+09:00
(22 min 47 s). Final state is active, enabled, NRestarts=0;
llama-qwen35-udq4.service remained inactive, disabled, and was never started.
Service command records redact the credential but retain the non-secret
invocation. No active manifest, systemd unit content, or /opt/ullm file changed.

- raw/: benchmark stdout/stderr, sample arrays, commands, request/SSE events,
  and AMD SMI JSONL histories.
- attempts/: image identities, R9700-only preflight, and full standard vLLM /
  SGLang command and log evidence.
- environment/: driver/device data, GGUF identity, llama.cpp source/build
  evidence, and service stop/restore records.
- [r9700-external-engine-baseline.jsonl](r9700-external-engine-baseline.jsonl):
  normalized success/failure rows.
- [summary.json](summary.json): machine-readable comparison summary.

Two early command-construction probes were rejected by the llama.cpp argument
parser before model loading or GPU dispatch. They are not benchmark rows. Every
model-load and timed measurement above passed R9700-only validation; no V620
inference workload was executed.

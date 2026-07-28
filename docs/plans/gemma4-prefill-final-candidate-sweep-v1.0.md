# Gemma4 prefill final candidate sweep v1.0

## Decision

**No candidate clears the approximately 1.10x realistic whole-prefill bar.**
Gemma4 prefill optimisation is therefore complete for this effort. No runtime
or kernel source was changed in this sweep.

The remaining large *free-component* bounds are not implementation forecasts:
they require a new device-resident activation graph and a different attention
topology, not a safe few-hour kernel adjustment. The directly relevant,
already-implemented M=16 matmul and batched sliding-reader trials measured
1.075x / 1.024x and 0.916x respectively, so they do not justify a further
sub-threshold implementation.

## Method

This is a final current-default trace at `a2e39e6d` (M=16 BF16 matmul, full
split-KV reader enabled, ordinary M=1 sliding reader retained). One cold N=512
and one cold N=2048 prefill were run on HIP ordinal 1, `gfx1201` / amd-smi GPU
2 (R9700) with all required HIP primitives forced. `ullm-openai` was stopped,
and the work was serialised by `flock /run/ullm/r9700.lock`; it was restored
active with `NRestarts=0` afterward. The unprofiled `attention-profile` runs
provide wall-clock component attribution; independent `rocprofv3
--kernel-trace` runs provide dispatch durations and launch geometry. Neither
profiler duration nor this one-cold-run attribution is a throughput result.

Raw local evidence is retained at
`benchmarks/results/2026-07-28/gemma4-prefill-final-sweep-ef-v1.0/raw/`.
The unprofiled prefill walls were 11.758131 s (N=512) and 59.467007 s
(N=2048). They differ from the clean five-median promotion throughput window
and are used only as same-run denominators below.

PMC mode was not usable: it creates an R9700 process before the target and
trips the resident driver's process guard. Thus the byte figures and GFLOPS
below are source-accounted arithmetic/bulk traffic, not physical HBM counter
measurements.

## Current dispatch evidence

| kernel / N | summed GPU time | launch grid | CTA/CU | achieved work rate |
| --- | ---: | --- | ---: | --- |
| sliding `ullm_paged_decode_attn_f32_kernel`, 512 | 2.778497 s | 2,048 / 256 = 8 CTAs | 0.125 | 10.842 GFLOPS (QK+AV) |
| sliding `ullm_paged_decode_attn_f32_kernel`, 2048 | 19.177905 s | 2,048 / 256 = 8 CTAs | 0.125 | 10.977 GFLOPS (QK+AV) |
| full split partial, 512 | 0.136645 s | 16,384 / 256 = 64 CTAs | 1.000 | 110.223 GFLOPS (QK+AV) |
| full split partial, 2048 | 1.836836 s | 16,384 / 256 = 64 CTAs | 1.000 | 131.007 GFLOPS (QK+AV) |
| full split merge, 512 / 2048 | 0.007359 / 0.028094 s | 2,048 / 256 = 8 CTAs | 0.125 | no QK+AV FLOP denominator |
| M=16 BF16 matmul | 1.795590 / 7.127388 s | 2,048--98,304 CTAs | 32--1,536 | 1.024 TFLOPS on the representative 12,288 x 1,536 x 128 MLP projection |

The M=16 matmul grid range above is taken from the exact current trace; it is
not underfilled. The representative matmul rate is the direct `2*M*N*K` count
from the promoted M=16 measurement. The full-reader partial rates count only
the required QK and AV arithmetic; merge performs softmax state combination
and has no equivalent QK+AV denominator.

For the sliding reader, the QK+AV totals are 30.123491 GFLOP (N=512) and
210.512118 GFLOP (N=2048). Its 8-CTA grid can occupy at most eight of the 64
CUs. This continues to explain the extremely low rate; it is neither a
measured HBM roof nor a compute roof.

## Candidate table

Shares use the current unprofiled cold-prefill wall above. Parenthesised
values are N=512 / N=2048. Bounds overlap and must not be added.

| candidate | measured share of prefill | Amdahl ceiling if free | realistic ceiling | rough effort | risk |
| --- | ---: | ---: | --- | --- | --- |
| Further Gemma-only BF16 matmul redesign | 25.62% / 21.51% end-to-end (15.27% / 11.98% kernel) | 1.344x / 1.274x end-to-end | **<=1.075x / <=1.024x** from the current M=8 to M=16 clean result; no small next tile change supports >=1.10x | high | high: F32 activation precision and occupancy; MFMA/WMMA is outside the acceptance error budget |
| New sliding attention topology that spreads the M=1 grid | 32.66% / 39.13% reader envelope (23.63% / 32.25% kernel) | 1.485x / 1.643x envelope (1.309x / 1.476x kernel) | no validated near-term ceiling; the directly relevant snapshot/direct-ring batching achieved **0.916x**, not a gain | architectural | high: causal/ring/shared-KV and F32 softmax correctness; not landable in hours |
| Remove residual reader transport while retaining the kernels | 9.33% / 7.16% | 1.103x / 1.077x | <1.10x across the target contexts; requires the same device-resident graph that the primitive-port trial showed is necessary | high | medium: host/device graph redesign, not a copy-call deletion |
| Tune full split partial reader | 1.16% / 3.09% GPU (1.53% / 3.41% full reader envelope) | 1.012x / 1.032x GPU (1.016x / 1.035x envelope) | <1.04x | high | high: full causal split-softmax validation and scratch/resource uncertainty |
| Make the split merge free | 0.063% / 0.047% | 1.0006x / 1.0005x | <1.001x | medium | low--medium; correctly below the already rejected 1.013x historical merge bound |
| Q/K/V/O, norms, RoPE, residual, or K/V-write micro-tuning | individually at most 1.66% / 1.53% in the current attention components | at most 1.017x / 1.016x | <1.02x | low--medium | low--medium |

The first two free-component bounds are retained as future architectural work
only. They do not clear the project bar as realistic, validated candidates.
No source change was made and no GPU acceptance suite is applicable.

## Rejected and gated work

- RMSNorm as an isolated device edge was slower: 12.297 tok/s versus the
  15.733 tok/s primitive baseline. It left the producing and consuming
  activations host-visible.
- Sliding batching was tested through snapshot gather and direct-ring forms;
  despite roughly 100x fewer logical K/V reads, its realised N=512/N=2048
  throughput was 0.916x.
- The `ULLM_GEMMA4_PREFILL_SLIDING_RING_BATCHED` route remains off by default.
  `ULLM_GEMMA4_PREFILL_LAYER_MAJOR=0`,
  `ULLM_GEMMA4_FULL_ATTN_SPLIT_KV=0`, and
  `ULLM_GEMMA4_SLIDING_ATTN_SPLIT_KV=0` remain explicit rollback controls.

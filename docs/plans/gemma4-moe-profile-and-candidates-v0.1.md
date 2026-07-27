# Gemma4 / Qwen3.5-35B-A3B profile and candidate ranking v0.1

## Decision first

**The Gemma4 uLLM-vs-llama.cpp comparison is like-for-like on numeric weight
format.**  It is not a BF16-vs-quantized-GGUF comparison.  uLLM ran the
source BF16 safetensors resident with F32 activations and F32 K/V.  The
llama.cpp `68a5592` result used
`gemma-4-E2B-BF16.gguf` (SHA-256 `e6c475…fce6a`), reported its model type as
`gemma4 E2B BF16`, used F32 K/V (`-ctk f32 -ctv f32`), and disabled FA
(`-fa off`).  Its command and raw result are in
`benchmarks/results/2026-07-26/gemma4-e2b-resident-v0.1/llama-cpp-benchmark.json`.

There is one scope difference, but it does not reverse the format answer:
the GGUF is the text-model export (9.295 GB reported model payload), while
uLLM deliberately residents the complete source checkpoint, including
vision/audio tensors.  The text decoder executed by both is BF16.  Therefore
the historical 218.956/69.960 llama.cpp tok/s versus uLLM figures are an
implementation/path comparison, **not predominantly a quantization gap**.

The proposed Gemma 512-wide full-attention split kernel is **NOT WORTH
DOING** as the next optimisation.  On the measured decode trace it is 3.993%
of wall time; even a free implementation has only a **1.0416x** Amdahl
ceiling.  It would require new 512-wide GPU math and a new finite-precision
contract, so the cost/risk is radically out of proportion to the bound.

## Measurement record

One exclusive R9700 (`gfx1201`, physical amd-smi index 2) window was used.
`ullm-openai.service` was stopped first, R9700 ownership/process emptiness was
recorded, each profile ran serially under `/run/ullm/r9700.lock`, then the
lock was released before the service restart.  The service returned
`ActiveState=active`, `NRestarts=0`.  The 18080 readiness endpoint did not
answer within the 1/3/6 s probe, but the systemd service itself was active;
this does not affect the isolated profile.

Raw files and restoration evidence are under
`benchmarks/results/2026-07-27/gemma4-moe-profile-v0.1/`.

| model | profile command workload | profiled prefill | profiled decode |
| --- | --- | ---: | ---: |
| Gemma4 E2B BF16 | 6-token prompt, 3 timed M=N repeats; 3 x 4-token M=1 decode repeats | 18 tokens / 1,239.220 ms = 14.525 tok/s | 12 tokens / 921.000 ms = 13.029 tok/s |
| Qwen3.5-35B-A3B AQ4_0 MoE | 23-token prompt then 4 generated tokens; 262,144 context, F16 K/V | 2,800.989 ms = 8.211 tok/s | 453.830 ms = 8.814 tok/s |

These profiler-instrumented wall rates are deliberately not substituted for
the normal throughput claims.  The unprofiled reference MoE result was
10.407 prefill / 11.039 decode tok/s; the closest unprofiled Gemma baseline
was 15.565 decode tok/s.

### Attribution method and limits

`rocprofv3 --kernel-trace --runtime-trace` was used, matching the existing
AQ4 e2e trace style.  Phase attribution is reconstructed from invariants in
the trace: one Gemma token has 35 direct-attention dispatches; one MoE token
has 40 route dispatches.  Gemma warm-up and decode setup groups are excluded.
The MoE trace is cut at its final `ullm_top1_f32_kernel`, excluding the
post-generation independent route-readback copies which are outside
`generation.{prompt,decode}_wall_ms`.

The phase-boundary convention is the first marker of the next token.  It can
misassign at most the prefix before one first-layer marker at a phase boundary;
it does not affect the all-token totals or the 35/40 marker counts.  The
reproducer is `analysis/analyze_trace.py` and its JSON output is retained.

Bytes below are **logical traffic lower bounds calculated from the recorded
model geometry and package layout**, not a fabricated hardware counter.
ROCprof's CSV records duration but no byte count for the internal ROCclr copy
kernel, so no per-dispatch byte value exists for `__amd_rocclr_copyBuffer`.
Those entries remain in the complete attribution tables rather than being
silently dropped.  Where a kernel's byte value is unavailable, it is marked
`not exposed` rather than estimated.

## Gemma4 roofline and attention split

The decode driver's recorded lower-bound traffic is `18,241,229,080 B / 4`
= **4,560,307,270 B/token**: BF16 projection/row reads plus F32 K/V read and
write.  At 640 GB/s this is a **140.341 tok/s** bandwidth roofline.  The
unprofiled 15.565 tok/s baseline reaches **11.09%** of it (the profiled
13.029 tok/s run reaches 9.28%).  This is a whole-token upper bound, not a
claim that every byte reaches HBM exactly once.

Gemma decode attention itself was 39.283 ms / 921.000 ms = 4.265% of wall.
The required full-vs-sliding split is:

| kind | time | launches | share of attention | share of decode wall | free-kernel Amdahl ceiling |
| --- | ---: | ---: | ---: | ---: | ---: |
| full, layers 4/9/14/19/24/29/34 | 36.780 ms | 84 | 93.628% | **3.993%** | **1.0416x** |
| sliding, remaining 28 layers | 2.503 ms | 336 | 6.372% | 0.272% | 1.0027x |

The full layers dominate attention because their 512-wide cache reads cost
more, but attention does not dominate decode.  This is precisely the
Qwen3.5-9B-style Amdahl trap that the measurement was meant to avoid.

### Top semantic kernels: logical bytes and bandwidth

| phase | kernel | logical bytes/token | achieved GB/s | % of 640 GB/s | basis |
| --- | --- | ---: | ---: | ---: | --- |
| prefill | `ullm_matvec_bf16_f32_kernel` | 3.889 GB | 490.9 | 76.7% | driver logical lower bound / traced matvec time |
| prefill | `ullm_paged_decode_attn_f32_kernel` | 0.100 MB | 0.054 | 0.008% | F32 K+V geometry / traced attention time |
| prefill | `ullm_bf16_row_f32_kernel` | not exposed | not exposed | — | trace has no operation byte field |
| decode | `ullm_matvec_bf16_f32_kernel` | 4.560 GB* | 537.7 | 84.0% | whole BF16 projection/row lower bound; rows are small and included |
| decode | `ullm_paged_decode_attn_f32_kernel` | 0.731 MB | 0.00089 | 0.00014% | F32 K+V geometry at lengths 7–10 |
| decode | `ullm_bf16_row_f32_kernel` | not exposed | not exposed | — | trace has no operation byte field |

`*` The decode accounting combines matvec and row reads.  It overstates the
matvec-only number by the unseparated BF16 row traffic, but is a conservative
upper bound and shows that the matrix path already sits near the 640 GB/s
logical roofline.  It is not evidence for a large standalone matvec gain.

### Full Gemma kernel attribution

| phase | kernel symbol | total ms | kernel share | launches |
| --- | --- | ---: | ---: | ---: |
| prefill | `ullm_matvec_bf16_f32_kernel` | 142.609 | 63.027% | 4,971 |
| prefill | `__amd_rocclr_copyBuffer` | 35.538 | 15.706% | 14,781 |
| prefill | `ullm_paged_decode_attn_f32_kernel` | 33.669 | 14.880% | 630 |
| prefill | `ullm_bf16_row_f32_kernel` | 13.864 | 6.127% | 4,752 |
| prefill | `ullm_paged_kv_write_f32_kernel` | 0.586 | 0.259% | 270 |
| decode | `ullm_matvec_bf16_f32_kernel` | 101.778 | 58.352% | 3,315 |
| decode | `ullm_paged_decode_attn_f32_kernel` | 39.283 | 22.522% | 420 |
| decode | `__amd_rocclr_copyBuffer` | 23.722 | 13.600% | 9,843 |
| decode | `ullm_bf16_row_f32_kernel` | 9.252 | 5.304% | 3,162 |
| decode | `ullm_paged_kv_write_f32_kernel` | 0.387 | 0.222% | 179 |

## Qwen3.5-35B-A3B MoE findings

This trace is the current `bad1b58…ac25b` MoE binary with the successful
262,144-token F16-K/V resident configuration.  It executed 23 prompt tokens
and 4 decode tokens; 40 route launches/token proves the complete 40-layer MoE
path was measured.

The first three semantic kernels are not all equally promising:

| phase | kernel | logical bytes/token | achieved GB/s | % of 640 GB/s | interpretation |
| --- | --- | ---: | ---: | --- |
| prefill | `ullm_moe_route_f32_kernel` | 42.312 MB | 1.835 | 0.287% | 40 BF16 router matrices plus F32 input/output; severe under-use |
| prefill | `ullm_moe_decode_gemm_f32_kernel` | 4.027 GB | 280.8 | 43.9% | selected 8-expert F32 gate/up+down slabs |
| prefill | `ullm_aq4_dequant_f32_kernel` | 4.656 GB | 473.5 | 74.0% | 0.629 GB AQ4 index+scale read plus 4.027 GB F32 write |
| decode | `ullm_moe_route_f32_kernel` | 42.312 MB | 1.880 | 0.294% | same 40-layer route geometry |
| decode | `ullm_moe_decode_gemm_f32_kernel` | 4.027 GB | 285.2 | 44.6% | selected 8-expert F32 slabs |
| decode | `ullm_aq4_dequant_f32_kernel` | 4.656 GB | 483.9 | 75.6% | AQ4 read + F32 materialization |

The AQ4 byte calculation is exact for the package's group-8 layout: per
layer, 8 selected gate/up experts plus 8 down experts read 15.729 MB of
index+scale and materialize 100.663 MB F32.  Across 40 layers that is 0.629
GB read and 4.027 GB written.  The route calculation is the 40 x
`[256,2048]` raw-BF16 router weights plus its small F32 input/output.  These
are lower bounds: activation/output traffic would only reduce the quoted
effective bandwidth.

### Full MoE kernel attribution

| phase | kernel symbol | total ms | kernel share | launches |
| --- | --- | ---: | ---: | ---: |
| prefill | `ullm_moe_route_f32_kernel` | 530.135 | 35.989% | 920 |
| prefill | `ullm_moe_decode_gemm_f32_kernel` | 329.779 | 22.388% | 1,840 |
| prefill | `ullm_aq4_dequant_f32_kernel` | 226.126 | 15.351% | 1,840 |
| prefill | `ullm_matvec_bf16_f32_kernel` | 207.007 | 14.053% | 8,073 |
| prefill | `__amd_rocclr_copyBuffer` | 135.568 | 9.203% | 48,346 |
| prefill | `ullm_linear_attn_recurrent_f32_kernel` | 12.282 | 0.834% | 690 |
| prefill | `ullm_rmsnorm_f32_kernel` | 10.733 | 0.729% | 1,863 |
| prefill | `ullm_moe_scatter_weighted_f32_kernel` | 3.479 | 0.236% | 920 |
| prefill | `ullm_add_f32_kernel` | 3.270 | 0.222% | 1,840 |
| prefill | `ullm_paged_decode_attn_typed_kf16_vf16_kernel` | 2.938 | 0.199% | 230 |
| prefill | `ullm_linear_attn_qkv_prepare_f32_kernel` | 2.788 | 0.189% | 690 |
| prefill | all other 8 symbols | 8.937 | 0.607% | 4,376 |
| decode | `ullm_moe_route_f32_kernel` | 90.008 | 35.731% | 160 |
| decode | `ullm_moe_decode_gemm_f32_kernel` | 56.471 | 22.417% | 320 |
| decode | `ullm_aq4_dequant_f32_kernel` | 38.485 | 15.277% | 320 |
| decode | `ullm_matvec_bf16_f32_kernel` | 36.018 | 14.298% | 1,399 |
| decode | `__amd_rocclr_copyBuffer` | 22.777 | 9.042% | 8,397 |
| decode | `ullm_linear_attn_recurrent_f32_kernel` | 2.133 | 0.847% | 119 |
| decode | `ullm_rmsnorm_f32_kernel` | 1.799 | 0.714% | 322 |
| decode | `ullm_paged_decode_attn_typed_kf16_vf16_kernel` | 0.993 | 0.394% | 40 |
| decode | `ullm_moe_scatter_weighted_f32_kernel` | 0.600 | 0.238% | 160 |
| decode | `ullm_add_f32_kernel` | 0.556 | 0.221% | 320 |
| decode | `ullm_linear_attn_qkv_prepare_f32_kernel` | 0.542 | 0.215% | 119 |
| decode | all other 7 symbols | 1.570 | 0.623% | 446 |

The raw JSON contains every remaining symbol individually; the final rows are
only compacted in this document to keep the decision record readable.

## Wall gaps: launch overhead is not the observed explanation

| model / phase | driver wall | summed kernel time | wall outside kernels | gap time after next launch API had returned | result |
| --- | ---: | ---: | ---: | ---: | --- |
| Gemma prefill | 1,239.220 ms | 226.265 ms | 1,012.955 ms (81.741%) | 2.445% | next launch generally had **not** returned |
| Gemma decode | 921.000 ms | 174.422 ms | 746.578 ms (81.062%) | 2.488% | next launch generally had **not** returned |
| MoE prefill | 2,800.989 ms | 1,473.042 ms | 1,327.947 ms (47.410%) | 7.188% | next launch generally had **not** returned |
| MoE decode | 453.830 ms | 251.907 ms | 201.923 ms (44.493%) | 8.028% | next launch generally had **not** returned |

This does **not** reproduce BU's 88.53% post-launch-return observation.
Here, only 2.5–8.0% of positive gap time occurs after the next launch API
returned (counts are 29.1–29.2% for MoE; the trace's gap classification is
time-weighted above).  The observed waiting is therefore predominantly before
the following kernel launch is submitted, not GPU queue idle after a completed
launch call.  It points to executor-side preparation, host/device copies,
allocation/synchronisation, or per-operation dependency handling—not the
bare HIP launch API cost.  This is an attribution, not yet a root cause.

## Ranked, Amdahl-bounded candidates

The realistic column assumes either removing half of a measured gap or a 2x
speedup of a kernel; it is explicitly a scenario, not a forecast.  `share`
is share of profiled driver wall time, so ceilings correctly include the
large non-kernel intervals.

| candidate | model | measured share of decode (or prefill) | Amdahl ceiling if the kernel became free | realistic ceiling | rough effort | risk |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Trace and remove executor pre-launch gaps (device-resident activation/workspace and submission dependencies) | Gemma decode | 81.062% gap | 5.280x | 1.682x if half removed | 12–20 h to localise, then variable | high: semantic ordering / finite precision |
| Same gap work, but batch M=N rather than issuing token-shaped prefill work | Gemma prefill | 81.741% gap | 5.477x | 1.691x if half removed | >20 h | high |
| Route 40 BF16 router matrices efficiently | MoE decode | 19.833% | 1.247x | 1.110x at 2x | 12–20 h | medium: routing/tie contract |
| Same MoE router work | MoE prefill | 18.927% | 1.234x | 1.105x at 2x | 12–20 h | medium |
| Reduce MoE pre-launch gaps | MoE prefill | 47.410% gap | 1.902x | 1.311x if half removed | 12–20 h to localise | medium-high |
| Reduce MoE pre-launch gaps | MoE decode | 44.493% gap | 1.802x | 1.286x if half removed | 12–20 h to localise | medium-high |
| MoE selected-expert F32 GEMM | MoE decode | 12.443% | 1.142x | 1.066x at 2x | 16–30 h | medium |
| Gemma generic BF16 matvec | Gemma decode | 11.051% | 1.124x | 1.059x at 2x | 20+ h | medium; already ~84% logical roof |
| MoE AQ4 dequant | MoE decode | 8.480% | **1.093x — NOT WORTH DOING** | 1.044x at 2x | 12–20 h | medium; already ~76% logical roof |
| Gemma 512-wide full-attention split kernel | Gemma decode | 3.993% | **1.0416x — NOT WORTH DOING** | 1.020x at 2x | 40+ h | high: new GPU math and new finite-precision contract |
| MoE full-attention kernel | MoE decode | 0.219% | **1.0022x — NOT WORTH DOING** | ~1.001x | 20+ h | high |

## Next 20 hours

**Do next:** instrument the Gemma executor's host-side work between the end
of a kernel and submission of the next one—separating buffer allocation/copy,
host conversion, stream synchronisation, and descriptor preparation—then
remove the largest measured contributor while preserving the resident BF16 /
F32 contract.  This follows the 81.1% decode gap and is the only Gemma target
with material Amdahl room.  Do not call it “launch-overhead optimisation”;
the trace specifically refutes that explanation here.

**Do not do:** implement the 512-wide Gemma full-attention split kernel.  It
looks attractive because full layers are 93.6% of Gemma attention time, but
they are only 3.99% of end-to-end decode wall time and cannot exceed 1.0416x.

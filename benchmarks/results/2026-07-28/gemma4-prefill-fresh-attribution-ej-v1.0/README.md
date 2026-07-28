# Gemma4 promoted-prefill fresh attribution EJ v1.0

## Decision

No new candidate clears the approximately **1.10x realistic whole-prefill
bar** on the current promoted build.  In particular, the former dominant
sliding reader is no longer a plausible target: the default direct-ring,
split-16 path is active and its partial-reader grid is 128 CTAs, or 2.0
CTA/CU.  The largest remaining measured component is the already-promoted
Gemma-only M=16 BF16 batched matmul.  Its immediately preceding M=8-to-M=16
step realized 1.075x at N=512 and 1.024x at N=2048; a further small tile
adjustment has no evidence for a 1.10x whole-prefill result.  WMMA/MFMA is
not an alternative because it violates the established F32-input numerical
acceptance argument.

Therefore this attribution commit intentionally makes **no runtime or kernel
source change**.  It is a fresh measurement of the current promoted path,
not a re-use of EF's pre-sliding-split table.

## Method and scope

The reusable EH root wrapper stopped `ullm-openai`, the user runner acquired
`/run/ullm/r9700.lock`, and the wrapper restored the service only after the
runner exited.  The target was HIP ordinal 1 / amd-smi GPU 2 (`gfx1201`), not
either gfx1030 V620.  All promoted endpoint guards were present:

```
ULLM_REQUIRE_HIP_ADD_KERNEL=1
ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL=1
ULLM_REQUIRE_HIP_BF16_ROW_KERNEL=1
ULLM_REQUIRE_HIP_GEMMA_PROPORTIONAL_ROPE_KERNEL=1
ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL=1
ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL=1
ULLM_REQUIRE_HIP_RMSNORM_KERNEL=1
ULLM_REQUIRE_HIP_ROPE_KERNEL=1
```

For each context, `attention-profile` performed one cold M=N prefill with no
warm-up or decode; an independent `rocprofv3 --kernel-trace` pass collected
dispatch duration and launch geometry.  These are attribution runs, not a
replacement for EI's clean five-median throughput receipt.  The unprofiled
runs were 54.298 tok/s (N=512) and 52.903 tok/s (N=2048), consistent enough
for a same-run denominator with EI's 56.094 / 53.829 tok/s medians; no
profiler duration is used as throughput.  PMCs were deliberately not used,
because their pre-target process trips the R9700 process guard.  GFLOPS below
are consequently source-accounted arithmetic divided by dispatch time, never
physical-counter figures.

`telemetry/` recorded a 38--60 C hotspot range; the maximum junction/hotspot
temperature observed was **60 C**, below the 110 C critical limit.

The runner's `preflight/runtime-source-sentinels.sha256` and the repository
baseline retain the required runtime TU guard
`475204184566b0798883a931c1f1528b86dec79b6b1aeb8310a1637d2153f699`.

## Fresh dispatch attribution

There are 64 CUs.  CTA/CU is the per-dispatch grid, not a residency claim.
`n/a` GFLOPS means the kernel is copying, writing K/V, converting/reading a
row, or merging softmax state and has no QK+AV or GEMM FMA denominator.  The
BF16-matmul representative is the 12,288 x 1,536 x 128 MLP projection;
the final head is 262,144 x 1,536 x 1.

| kernel | N=512 GPU time | N=512 grid / CTA/CU | N=512 derived rate | N=2048 GPU time | N=2048 grid / CTA/CU | N=2048 derived rate |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| `ullm_gemma_bf16_matmul_f32_kernel` | 1.876669 s | 2,048--98,304 / 32--1,536 | 1.047 TFLOPS representative MLP | 7.504403 s | 2,048--98,304 / 32--1,536 | 1.052 TFLOPS representative MLP |
| sliding split partial | 0.259572 s | 128 / 2.000 | 116.05 GFLOPS QK+AV | 1.814281 s | 128 / 2.000 | 116.03 GFLOPS QK+AV |
| full split partial | 0.129230 s | 64 / 1.000 | 116.55 GFLOPS QK+AV | 1.788164 s | 64 / 1.000 | 134.56 GFLOPS QK+AV |
| sliding split merge | 0.054886 s | 8 / 0.125 | n/a (softmax-state merge) | 0.217097 s | 8 / 0.125 | n/a (softmax-state merge) |
| full split merge | 0.007180 s | 8 / 0.125 | n/a (softmax-state merge) | 0.028400 s | 8 / 0.125 | n/a (softmax-state merge) |
| K/V write | 0.022680 s | 2--4 / 0.031--0.063 | n/a (write) | 0.097329 s | 2--4 / 0.031--0.063 | n/a (write) |
| BF16 row | 0.009512 s | 1 / 0.016 | n/a (row read/conversion) | 0.037765 s | 1 / 0.016 | n/a (row read/conversion) |
| final BF16 matvec head | 0.001434 s | 262,144 / 4,096 | 561.6 GFLOPS | 0.001432 s | 262,144 / 4,096 | 562.4 GFLOPS |
| runtime copies | 0.083362 s | 1--2 / 0.016--0.031 | n/a (copy) | 0.329674 s | 1--2 / 0.016--0.031 | n/a (copy) |
| runtime fills | 0.003790 s | 32 / 0.500 | n/a (fill) | 0.003650 s | 32 / 0.500 | n/a (fill) |

Every traced kernel is listed above.  The genuinely underfilled merge,
row/write, copy, and fill grids are visible, but their summed time is too
small to make grid supply a whole-prefill target.  The two reader partial
grids are no longer the EF 8-CTA failure mode; the full reader is 1.0 CTA/CU
and the sliding reader is 2.0 CTA/CU.

## Candidate table

Shares use the matching unprofiled cold prefill walls, 9.429495 s / 38.712405
s.  Host-boundary values are used where a candidate affects the complete
synchronous operation; GPU-only values are explicitly labelled.  Several
rows overlap (for example, matmul calls occur inside attention regions), so
the bounds cannot be added.

| candidate | measured share of prefill (512 / 2048) | Amdahl ceiling if free | realistic ceiling | effort | risk |
| --- | --- | --- | --- | --- | --- |
| Further F32-input M=16 BF16 batched-matmul redesign | 35.53% / 33.90% host round trip; 19.90% / 19.38% GPU | 1.551x / 1.513x host | **<1.10x**: the already-landed adjacent M=8->M=16 step realized only 1.075x / 1.024x; no numerical route beyond it | high | high: F32-order/occupancy; WMMA ruled out |
| Sliding reader partial/transport | 4.68% / 6.40% complete reader; 2.75% / 4.69% GPU partial | 1.049x / 1.068x | **NOT WORTH DOING**, below 1.10 even if free | high | high: ring, causal, shared-KV and split-softmax |
| Full reader partial/transport | 1.94% / 5.17% complete reader; 1.37% / 4.62% GPU partial | 1.020x / 1.055x | **NOT WORTH DOING** | high | high: causal split-softmax/scratch |
| Sliding + full split merge | 0.66% / 0.63% GPU | 1.0066x / 1.0063x | **NOT WORTH DOING**; confirms the pre-EG rejection remains stronger after split-16 | medium | medium |
| Copies, K/V write, BF16 row, fill, final head combined | 1.25% / 1.21% GPU | 1.013x / 1.012x | **NOT WORTH DOING** | low--medium | low--medium |
| Q/K/V/O, norms, RoPE and residual micro-tuning | each is an overlapping sub-component; no isolated item reaches 1.10x | <1.10x after its complete host boundary is charged | **NOT WORTH DOING** | medium | medium |

## What changed from EF's stale table

EF's N=2048 trace had the unbatched sliding reader at 19.177905 s,
8 CTAs / 0.125 CTA/CU, 10.977 GFLOPS, and 32.25% GPU (39.13% complete reader)
of its cold prefill.  This fresh current-default trace records 1.814281 s,
128 CTAs / 2.0 CTA/CU, and 116.03 derived GFLOPS: a 10.57x lower partial
duration and 10.57x higher rate.  Its complete-reader share is now 6.40%,
not 39.13%.  This both refutes EF's ranking and validates EG's diagnosis that
the previously-existing split reader had been unreachable.

The new largest measured component is M=16 batched matmul, but it is not a
new unexamined win: it is already the promoted implementation and its direct
adjacent experiment did not meet the project bar.  No code is therefore
changed after this attribution commit, and no new GPU acceptance suite is
claimed or needed.

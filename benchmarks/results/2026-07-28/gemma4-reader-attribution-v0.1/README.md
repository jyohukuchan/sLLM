# Gemma4 attention reader attribution v0.1

## Decision

The layer-count explanation is **refuted**.  DW's region timer is a complete
attention-side timer, but the new split shows that the reader itself remains
the majority of the sliding region, especially at realistic context lengths.
That does **not** revive the K/V-traffic premise: rocprof shows that the
reader's GPU kernels, rather than K/V byte volume, account for most of that
time at N=2048.

| N | sliding region | sliding reader host round trip | sliding reader GPU kernel | GPU kernel / sliding region | full region | full reader host round trip | full reader GPU kernel |
| --: | --: | --: | --: | --: | --: | --: | --: |
| 128 | 0.712366 s | 0.466087 s (65.43%) | 0.186538 s (26.19%) | 26.19% | 0.158460 s | 0.078719 s (49.68%) | 0.070212 s |
| 512 | 4.674272 s | 3.825110 s (81.83%) | 2.778986 s (59.46%) | 59.46% | 1.270083 s | 0.976126 s (76.86%) | 0.941778 s |
| 2048 | 27.022086 s | 23.554878 s (87.17%) | 19.054416 s (70.51%) | 70.51% | 15.496220 s | 14.287705 s (92.20%) | 14.238781 s |

`reader host round trip` starts after K/V write and ends after the reader's
output copy and final stream synchronization.  It is deliberately not called
kernel time.  The GPU-kernel column is the summed duration of exact rocprof
kernel traces: `ullm_paged_decode_attn_f32_kernel` for sliding and
`ullm_gemma_full_attn_batched_512_f32_kernel` for full.

At N=2048, the host-only reader envelope above the two GPU kernels is 4.550 s
(6.55% of cold prefill).  The reader bottleneck is therefore principally
kernel execution at this context, not the old logical K/V-read count.

## Attention-side component attribution

All values below are seconds accumulated over the given layer kind.  `other`
is the exact unaccounted region remainder (ordinary host bookkeeping and any
small synchronization not inside a timed component), not a hidden reader
bucket.

| N / kind | input RMS | Q proj | K proj | V proj | RoPE + head norms | KV write | reader round trip | O proj | post norm | residual | other | region |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 sliding | .019890 | .047013 | .007761 | .007276 | .049875 | .044044 | .466087 | .047517 | .017572 | .005291 | .000042 | .712366 |
| 128 full | .004912 | .020130 | .002288 | .002308 | .019736 | .003894 | .078719 | .020580 | .004328 | .001555 | .000008 | .158460 |
| 512 sliding | .071413 | .180282 | .028778 | .026765 | .188520 | .096824 | 3.825110 | .175240 | .062867 | .018323 | .000151 | 4.674272 |
| 512 full | .017623 | .076241 | .008745 | .008379 | .072639 | .013403 | .976126 | .076373 | .015660 | .004864 | .000030 | 1.270083 |
| 2048 sliding | .295010 | .726089 | .117269 | .108833 | .830141 | .328636 | 23.554878 | .714867 | .264372 | .081249 | .000741 | 27.022086 |
| 2048 full | .073004 | .306848 | .037182 | .034323 | .304821 | .053064 | 14.287705 | .312942 | .065789 | .020352 | .000190 | 15.496220 |

The new global counter also measures every current batched BF16 matmul,
including PLE, attention projections, MLP projections, and the final head:
0.854060 s / 3.115722 s / 12.568436 s at N=128 / 512 / 2048 (276 / 1,104 /
4,416 calls).  This was not present in DS's earlier profile, so that older
75.7%-attention conclusion is not used as a current candidate ranking.

## Candidate table (N=2048 cold prefill: 69.463637 s)

| candidate | measured share of prefill | Amdahl ceiling if it became free | realistic ceiling | rough effort | risk |
| --- | ---: | ---: | ---: | --- | --- |
| Improve batched BF16 matmul (WMMA/MFMA structural redesign, not AQ4) | 18.09% | 1.221x | 1.10–1.18x | high | high: numerical/occupancy regressions; keep AQ4 WMMA untouched |
| Tune full batched reader kernel | 20.50% GPU kernel | 1.258x | unknown, likely below ceiling | high | high: causal correctness and LDS/occupancy |
| Batch sliding reader (DX/DY/DZ) | 27.43% GPU kernel; 33.91% full synchronous envelope | 1.378x kernel-only; 1.513x envelope | **0.916x measured** (DZ: 27.37 vs 29.86 tok/s) — **NOT WORTH DOING** | already attempted | proven regression despite ~100x lower logical K/V reads |
| Remove reader host transport alone | 6.55% | 1.070x | <1.07x — **NOT WORTH DOING** | medium | medium |
| Attention Q/K/V/O projections alone | 3.39% | 1.035x | <1.04x — **NOT WORTH DOING** | medium | medium |
| RoPE, RMSNorm, residual, and K/V-write micro-optimizations | each <=1.64% | <=1.017x | <1.02x — **NOT WORTH DOING** | low–medium | low–medium |

The single next implementation target is the **batched BF16 matmul**, gated
by a new isolated correctness/performance campaign.  It is the only
non-reader candidate with a measured whole-prefill ceiling above 1.10x and a
plausible path to that bar on gfx1201 BF16 matrix hardware.  The existing AQ4
WMMA GEMMs are reference-only and must not be modified.

## Method

The committed instrumentation uses only `std::time::Instant` around existing
host boundaries; it adds no HIP calls and does not alter the execution graph.
It extends DW's layer-major complete-region timer in `77e9c673`.  Each result
is one cold prefill, no warmup or decode, on HIP ordinal 1 (`gfx1201`) /
amd-smi GPU 2 with `ullm-openai` stopped and `/run/ullm/r9700.lock` held.

`raw/attention-profile-v2-n*.json` contains the non-profiled attribution.
`raw/rocprof-n*/` contains the independent exact kernel traces.  The `v2`
files include the global batched-matmul counter; preceding raw profiles are
retained as the initial component-timer check and are not used for the
candidate table.

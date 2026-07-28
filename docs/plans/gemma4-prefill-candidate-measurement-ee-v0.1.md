# Gemma4 prefill candidate measurement EE v0.1

## Decision

Candidate A (split-reader merge grids) is **not worth doing** at promoted
`626d09bc`. Candidate B (Gemma BF16 batched matmul) is the only candidate
above the 1.10x whole-prefill ceiling, but its grid is already amply
populated; this refutes the underfill premise.

## Method

One cold N=2048 resident prefill was traced on HIP ordinal 1 / gfx1201 (the
R9700, amd-smi GPU 2) with promoted split factors (full=8, sliding=32) and
the batched sliding reader. `ullm-openai` was stopped first; the trace ran
under an exclusive `flock /run/ullm/r9700.lock`. This is dispatch-level timing
from `rocprofv3 --kernel-trace`; PMC mode was not used.

Raw evidence remains outside the commit at
`benchmarks/results/2026-07-28/gemma4-prefill-ee-measurement/raw/`:
`attention-profile-promoted-n2048.json` and the rocprof SQLite database.

The unprofiled execution reported 38.941971 s / 52.591 tok/s. The host timer
reported 13.527764 s for 4,416 Gemma batched BF16 matmul calls.

## Candidate A — merge grids

| reader | partial GPU time | merge GPU time | merge / reader | merge / full prefill | merge grid | CTA/CU | whole-prefill ceiling |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| full 512, split=8 | 1.790359 s | 0.027952 s | 1.54% | 0.072% | 2,048 threads / 256 = 8 CTAs | 0.125 | 1.0007x |
| sliding 256, split=32 | 0.985930 s | 0.470846 s | 32.32% | 1.209% | 2,048 threads / 256 = 8 CTAs | 0.125 | 1.0122x |
| both merges | — | 0.498798 s | — | 1.281% | — | — | **1.0130x** |

The merge has one CTA per Q head, but most reader work remains in the split
partial phase. Even a free merge cannot meet the 1.10x threshold, so no
split-of-splits merge implementation is justified.

## Candidate B — batched BF16 matmul

| metric | measurement |
| --- | ---: |
| host-boundary time / whole prefill | 13.527764 s / 38.941971 s = **34.74%** |
| Amdahl ceiling if made free | **1.532x** |
| dispatch-only GPU time / all traced kernels | 8.167429 s / 11.879719 s = **68.75%** |
| launches | 4,416 |
| block / LDS / registers | 256 threads / 512 B / 32 VGPR, 0 AGPR, 128 SGPR |
| minimum grid population | 65,536 threads × 16 y / 256 = 4,096 CTAs = **64 CTA/CU** |
| maximum grid population | 3,145,728 threads × 16 y / 256 = 196,608 CTAs = **3,072 CTA/CU** |

The M=8 LDS-weight kernel is not grid-starved. It reuses each staged BF16
weight strip across M but repeatedly consumes the same F32 activation tiles
for many output rows. Candidate B therefore remains worth a Gemma-specific,
F32-input memory-reuse experiment; WMMA/MFMA remains excluded because input
rounding alone exceeds the `4.8e-5` acceptance bar.


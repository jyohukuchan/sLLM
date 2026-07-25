# SQ8_1 V620 (gfx1030) kernel optimization — 2026-07-26

## Verdict

For the decode-like M=1 Qwen3-14B FP8 `self_attn.q_proj` shape (5,120 ×
5,120), optimized `SQ8_1` wins the matched `SQ8_0` comparison on the allowed
V620 card0:

| format / path | median of three run medians (ms) | modeled GB/s | % of 512 GB/s | ns / logical FMA | elapsed-time result vs `SQ8_0` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `SQ8_0` E4M3 + F32 block-scale fallback | 0.639007 | 41.034 | 8.014% | 0.0243762 | baseline |
| `SQ8_1` W8A16, wave32 × 8 rows | 0.237362 | 117.343 | 22.919% | 0.00905464 | **2.692× faster** (62.855% less time) |
| `SQ8_1` explicit W8A8, tiled wave32 × 8 rows | 0.249762 | 111.517 | 21.781% | 0.00952766 | **2.558× faster** (60.914% less time) |

This is an elapsed-time result, not a release or dispatch admission. In
particular, the full-model `SQ8_1` W8A8 logits gate remains unverified, so the
explicit-only W8A8 API policy is unchanged.

`SQ8_1` streams 27,852,800 resident weight bytes in this benchmark, 6.224% more
than the 26,220,800-byte `SQ8_0` fallback representation. Its result therefore
does not come from a smaller resident representation. The GB/s and percentage
columns are modeled weight streams using each benchmark's declared 512 GB/s
reference; they are not profiler-derived DRAM transaction counts.

## Matched protocol and raw observations

Each `SQ8_1` run performed 32 warmups and 31 single-launch HIP-event trials.
The result is the median of the three per-run medians, matching the pre-existing
`SQ8_0` protocol. `SQ8_0` values are taken from the committed
`../sq9-v620-viability/raw/final-m1-r{1,2,3}-card0-v4.jsonl` records; `SQ8_1`
records are local `raw/final-m1-r{1,2,3}-card0.jsonl`. Both use Qwen3-14B
q_proj, M=1, physical `0000:03:00.0` / card0, a <=42 C cooldown target, and
the same 512 GB/s efficiency convention. They are separate benchmark
processes, not a co-dispatched A/B trace.

| run | `SQ8_0` ms | `SQ8_1` W8A16 ms | `SQ8_1` W8A8 ms |
| --- | ---: | ---: | ---: |
| r1 | 0.6390069723 | 0.2373619974 | 0.2504029870 |
| r2 | 0.6388069987 | 0.2370820045 | 0.2497619987 |
| r3 | 0.6399270296 | 0.2373629957 | 0.2497619987 |
| median of run medians | 0.6390069723 | 0.2373619974 | 0.2497619987 |

The first W8A8 run contains scheduler/DPM outliers in its 31 samples (up to
1.010089 ms), but its per-run median is retained as prescribed; the outer
three-run median rejects that noise. No sample selection was applied.

## Kernel changes and static evidence

The optimized HIPRTC source is compiled dynamically by the runtime; the
offline audit compiles that exact source device-only with ROCm 7.2.1. The
reference comparison is the committed source artifact from `8b15545e`
(`881511…a2df` source SHA-256). The optimized source hash is
`88489aa38dbddd297f59f91f27a22bbfa81e7b5677d6d64282ed78af75ce15a3`.

- W8A16 maps eight logical wave32 rows onto one 256-thread block. It replaces
  the 256-element LDS reduction tree with wave shuffles. A complete K=32
  payload block still uses exactly two aligned `uint4` loads: two 16-byte
  loads / 32 elements = 1/16 128-bit payload-load instructions per element.
- Explicit W8A8 quantizes the K=32 activation plane once for all eight output
  rows, stores it in dynamic LDS, and then uses the same plane for eight
  `v_dot4*`-based row computations. At the measured 5,120 columns that is
  5,760 B dynamic LDS (5,120 code bytes + 160 F32 scales), below the 48 KiB
  guarded launch limit. A shape-preserving fallback keeps arbitrary valid
  shapes usable when that reservation would exceed the limit.
- The full K=32 W8A8 hot path retains eight dot4 operations / 32 elements =
  0.25 dot4 instructions per element. The improvement is not a reduced dot
  count: activation scale/divide/round work is amortized across eight rows,
  from 32 activation quantizations per output row/K=32 to four amortized per
  output row/K=32.
- Architecture selection remains explicit: gfx1030 emits signed
  `v_dot4c_i32_i8`; RDNA3/RDNA4 emits `v_dot4_i32_iu8` with the signed-control
  path; the CDNA targets retain their dot4 selection. The all-whitelist
  compile is in `static-optimized/`.

gfx1030 resource and static-body deltas are in
[`static-optimized/isa-comparison.md`](static-optimized/isa-comparison.md).
In short: W8A16 fixed LDS goes 1,024 B -> 0 B and barriers 2 -> 0; W8A8 fixed
LDS goes 1,024 B -> 0 B (with the explicit 5,760 B dynamic tile at the measured
shape), barriers 2 -> 1, VGPR 53 -> 39, SGPR 59 -> 32, and all spill counts stay
zero. Achieved occupancy was not profiler-measured; no numerical occupancy
claim is made beyond the code-object resource evidence.

The static instruction counts describe emitted code bodies, including control
and tail paths, rather than an exact dynamic instruction count for every
element. They are paired with the source-level per-K=32 accounting above; the
static body alone must not be interpreted as a literal per-element ratio.

## Numerical gates and non-interference

All gates passed before timing in each final benchmark run.

| gate | result |
| --- | --- |
| CPU `SQ8_1` reference tests | 3/3 pass, including columns 1/15/16/17/31/32/33/65; signed -127/127, zero row, and physical-tail cases |
| V620 full-shape pre-timing gate | W8A16: max abs 0, relative L2 0; W8A8: max abs 0.1875, relative L2 2.331406575e-07; both pass declared <=1.0 / <=1e-6 limits |
| V620 runtime tail differential | 7 × 65, stride 80, eight launches/path: W8A16 relative L2 6.076546605e-08 and W8A8 4.333164297e-08; max abs 0.0078125 each; pass |
| `SQ8_0` CPU regression | `cpu_sq_fp8_matvec_f32_computes_expected_row_block_values`: pass |
| format / artifact separation | `tests/test_sq8_1_artifact.py` + `tests/test_ullm_format_ids.py`: 9/9 pass, including the `SQ8_0` non-interference assertion |
| `AQ4_0` offline oracle | `tests/test_aq4_layer0_matvec_oracle.py`: 18/18 pass |

Only the `SQ8_1` kernels, their runtime selection, an `SQ8_1` test, and
standalone benchmark/evidence files changed in this work. No `SQ8_0` or
`AQ4_0` production implementation, candidate, release, campaign,
authorization, `/opt/ullm` content, service, or active manifest was changed.

## Safety record and scope limits

The benchmark accepted only the V620 reported by
`hipDeviceGetPCIBusId`: AMD Radeon Pro V620 / `gfx1030`, BDF `0000:03:00.0`.
It resolved that BDF to DRM `card0` and its own junction sensor
`/sys/class/drm/card0/device/hwmon/hwmon5/temp2_input` (`temp2_label=junction`).
Thermal samples were taken before and after every warmup and timed launch.

| run | junction samples | min–max | guard result |
| --- | ---: | ---: | --- |
| r1 | 272 | 41–43 C | completed; cooldown restored 41 C before W8A8 timing |
| r2 | 271 | 41–42 C | completed |
| r3 | 271 | 41–42 C | completed |
| tail differential | 36 | 41–41 C | completed |

The hard stop was 85 C. No `thermal_guard_triggered` or `cooldown_timeout`
record exists; the overall final-suite peak was 43 C. No R9700 execution
occurred.

Only M=1 was measured. M>1/prefill measurements were deliberately omitted
after the staged M=1 result, not because the 85 C guard fired. A runtime
elapsed-time measurement of the old `SQ8_1` reference kernel was also not
collected, so the direct old-vs-new speedup is **unmeasured**; the available
before/after evidence is ISA/resource analysis plus the required `SQ8_0`
comparison.

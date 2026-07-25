# SQ8_0 / SQ8_1 gfx1030 fair comparison

## Scope

This evidence is an isolated V620/gfx1030 kernel experiment. It changed no
runtime dispatch or external ABI, no candidate/release/campaign/authorization,
no active manifest, no `/opt/ullm` state, and did not execute R9700/gfx1201.

The benchmark selected `HIP_VISIBLE_DEVICES=2`, then required
`hipDeviceGetPCIBusId` to resolve to `0000:03:00.0` / `AMD Radeon Pro V620` /
`gfx1030`. Its thermal guard resolved that same BDF to `card0` and its own
`/sys/class/drm/card0/device/hwmon/hwmon5/temp2_input` junction sensor. Every
timed point began after cooldown to <=42 C, samples during every launch, and
aborts at >=85 C.

The workload has Qwen3-14B `self_attn.q_proj` dimensions (`5120 x 5120`). The
weights and inputs are deterministic synthetic data; `SQ8_1` is requantized
from the benchmark's common `SQ8_0` E4M3 + F32 [128,128] source. These are
shape/kernel measurements, not an actual-model throughput or quality result.

## Final evidence

- `raw/final-row256-v4.jsonl`: complete machine-readable timing, numerical-gate,
  device-identity, and thermal event stream.
- `raw/final-row256-v4-thermal.jsonl`: the corresponding junction-only history.
- `summary.json`: derived values below, preserving per-run values and paired
  ratios instead of collapsing the clock-state variation into one absolute
  latency.
- `static-audit.json`: compact machine-readable compiler-resource, ISA-count,
  and gfx1201 non-interference result.
- `static-analysis.md`: exact HIPRTC static resource/ISA audit and gfx1201
  non-interference result.

The `smoke-*`, `prototype-*`, `static-prototype`, and earlier `final-row256-v1`
files are intermediate tuning evidence. They are not the final comparison
result. Generated executable files under `build/` and untracked code objects
are reproducibility by-products, not release artifacts.

## SQ8_0 gfx1030 specialization

The generic `SQ8_0` symbols retain their ABI and existing host launch geometry.
Only `#if defined(__gfx1030__)` bodies change:

- complete aligned K=16 payload segments use one `uint4` / 128-bit load;
- each 256-thread CTA performs eight logical wave32 shuffle reductions, writes
  eight F32 partials to 32 B LDS, executes one barrier, and has thread zero
  finish the sum;
- arbitrary scale grids, scale boundaries, unaligned data, and tails retain
  scalar E4M3 reconstruction semantics.

The exact gfx1030 direct kernel changed from 0 `global_load_dwordx4`, 0
`ds_bpermute`, 2 `s_barrier`, 3 `ds_read`, and 2 `ds_write` instructions to
1, 5, 1, 2, and 1 respectively. Fixed LDS fell from 1024 B to 32 B. Exact
runtime-source metadata is direct `31 VGPR / 48 SGPR / 32 B LDS`, batch
`31 / 52 / 32 B`, both with zero private memory and spills. The old generic
source was direct `22 / 42 / 1024 B`, batch `22 / 47 / 1024 B`.

An isolated `__launch_bounds__(256, 2)` static prototype remained at 30 VGPR,
48 SGPR, 32 B LDS, and zero spills, the same resource class as the ordinary
prototype; it did not justify an additional residency constraint. Profiler
occupancy and measured DRAM transactions are unconfirmed.

## Numerical gates

All gates passed before performance timing:

| path | checked output | max abs | relative L2 |
| --- | --- | ---: | ---: |
| `SQ8_0` direct | 8 rows | 2.384185791e-07 | 8.552191029e-07 |
| `SQ8_1` W8A16 direct | 8 rows | 1.132488251e-06 | 3.767329717e-06 |
| `SQ8_1` W8A8 direct | 8 rows | 0 | 0 |
| `SQ8_0` exact batch symbol | 2 x 8 rows | 2.384185791e-07 | 3.869252278e-07 |
| W8A16 benchmark-only 2-D batch prototype | 2 x 5120 rows | 2.205371857e-06 | 3.989831230e-06 |
| W8A8 benchmark-only prequantized 2-D batch prototype | 2 x 5120 rows | 1.788139343e-07 | 2.827435228e-07 |

## Fair M=1 result

All three paths were launched in rotating order in one process, with 32 warmups
and 31 timed samples in each of three cooldown-normalized runs. Absolute run 3
latencies are about 2.5x lower for all paths even though its start temperature
was comparable; the reason is unconfirmed. Therefore the fair conclusion uses
matched ratios within each run.

| run | SQ8_0 ms | W8A16 ms | W8A8 ms | SQ8_0 / W8A16 | SQ8_0 / W8A8 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.627807021 | 0.237481996 | 0.248962998 | 2.644x | 2.522x |
| 2 | 0.628367007 | 0.238682002 | 0.249202996 | 2.633x | 2.522x |
| 3 | 0.251682013 | 0.100920998 | 0.112040997 | 2.494x | 2.246x |
| paired-ratio median | — | — | — | **2.633x** | **2.522x** |

On this V620 generic path, `SQ8_1` therefore retains a format-path advantage
over equally optimized `SQ8_0`, but the historical 2.692x W8A16 value was not
a format-only result: it compared optimized `SQ8_1` with the pre-specialized
`SQ8_0` fallback. The fair matched result is 2.633x (W8A16), and 2.522x for
W8A8. This does not establish a gfx1201 result.

## M sweep

The existing exact runtime interface has only independent matvec launches. Its
M sweep is included separately because it is the deployed-kernel semantics;
it does not add a batch ABI or claim fused GEMM. W8A16 remained faster at every
requested M, so there is no W8A8 crossover through M=128 on that path.

| M | W8A16 median-of-runs ms | W8A8 median-of-runs ms | W8A16 / W8A8 paired-ratio median |
| ---: | ---: | ---: | ---: |
| 1 | 0.237403005 | 0.253161997 | 0.938x |
| 8 | 0.423444003 | 0.483125001 | 0.876x |
| 32 | 1.691699028 | 1.868780017 | 0.905x |
| 128 | 7.008595943 | 7.329880238 | 0.958x |

To isolate activation-quantization reuse, the benchmark also compiles an
explicitly non-runtime, two-stage prototype: one exact K=32 activation
quantization per input row, then a 2-D batch output grid. It changes neither
the runtime source nor ABI/dispatch. In this prototype W8A8 is already faster
at M=1; no sampled M>1 threshold is needed or observed.

| M | W8A16 median-of-runs ms | W8A8 prequant median-of-runs ms | W8A16 / W8A8 paired-ratio median |
| ---: | ---: | ---: | ---: |
| 1 | 0.237802997 | 0.168281004 | 1.415x |
| 8 | 1.562656999 | 0.460604996 | 3.393x |
| 32 | 1.309733987 | 0.591805995 | 2.214x |
| 128 | 5.224856853 | 2.095663071 | 2.493x |

This separates two facts: increasing M in the current direct API does not
hoist a distinct activation plane, while hoisting it out of the eight-output-
row CTA is valuable. A production prefill/batch ABI, dispatch admission,
full-model W8A8 quality, and profiler evidence remain unimplemented or
unconfirmed.

## Thermal result and omissions

The complete final stream spans 40–54 C: M=1 co-dispatch 40–43 C, exact direct
M sweep 41–54 C, and the prequant prototype sweep 41–50 C. The 85 C guard and
cooldown timeout both remained untriggered. All requested M={1,8,32,128}
points completed; none were omitted for heat.

Unmeasured for reasons other than thermal are profiler-derived occupancy/DRAM
traffic, a production batch/prefill implementation, full-model W8A8 quality,
and any R9700/gfx1201 execution.

# CP: R9700 hardware microbenchmark reliability rerun

## Decision

- Chose **(A)** for GEMM: retain the identical naive WMMA/MFMA implementation
  on both targets and label it as an effective memory-system-inclusive result.
  It has no LDS tiling and is not a matrix-core peak benchmark. The real
  `256x5120x5120` projection shape remains useful for uLLM correspondence.
- Took exactly one R9700-exclusive window. The runner stops the production
  service, acquires `/run/ullm/r9700.lock`, releases it before service start,
  then proves restoration. `active.json` was only hashed, never changed.

## STREAM read root cause and correction

The old read kernel made 65,535 CTAs reduce into one contended `atomicAdd`.
That was the bottleneck: the old 62.963 GB/s was measuring atomic contention
and reduction overhead rather than a read stream. Its ISA did contain a load,
so dead-load elimination was not the cause.

The new kernel has four independent accumulators over vector-width loads and
writes one checksum per CTA to a distinct location (4,096 CTAs). The new ISA
audit records two global/flat loads and zero global/flat atomics. R9700 read is
629.208 GB/s (98.314% of 640), above copy at 582.743 GB/s (91.054%); triad is
603.503 GB/s (94.297%). This confirms the defect is corrected.

## DPM evidence

`amd-smi metric --gpu 2 --temperature --clock --power --violation --json` was
sampled throughout warmup and every group. The 12-second BF16 WMMA warmup had
35 valid samples; the last three were 3332, 3336, and 3326 MHz, satisfying the
recorded gate (three samples >=1 GHz and within 5%). Active bandwidth recorded
3391 MHz / 54 C / 216 W. Active GEMM recorded 11 samples at 3297–3403 MHz,
54–57 C, and 172–278 W (median 3363 MHz / 55 C / 210 W).

Therefore the CO window's 234–430 MHz readings were idle boundary samples, not
the clock during timed kernels. `THROTTLED` appears in some active samples, but
all concrete violation fields are `N/A`; no conclusion is inferred from that
word alone.

## Remeasurement and timing

| Metric | CO | CP | Change |
|---|---:|---:|---:|
| STREAM read | 62.963 GB/s | 629.208 GB/s | +566.245 GB/s |
| STREAM copy | 584.167 GB/s | 582.743 GB/s | -0.244% |
| STREAM triad | 574.355 GB/s | 603.503 GB/s | +5.075% |
| BF16 real shape | 15.205 TFLOPS | 14.445 TFLOPS | -5.00% |
| FP8 real shape | 23.535 TFLOPS | 22.448 TFLOPS | -4.62% |

The build, ISA audit, DPM warmup, validation, bandwidth, and GEMM runner span
was 41.906 s (12:18:22.785–12:19:04.691 JST). Timed group wall durations after
warmup were 0.193 s validation, 0.845 s bandwidth, and 4.786 s GEMM; reported
throughputs remain HIP-event values only. This is the R9700 estimate for the
MI300X microbenchmark component, with host provisioning outside that interval.

gfx942 offline build and ISA audit pass after the changes. The required
`v_mfma_f32_16x16x32_fp8_fp8` is present; the same stream-read ISA audit finds
two global/flat loads and zero atomics.

## Production restoration

The manifest SHA-256 was unchanged before and after:
`a654d92fe8142fcc0904fe187c96b84c95e0dd18acac61ef25d0cfa6429a08cd`.
The unit returned `ActiveState=active`, `NRestarts=0`; the system journal shows
start at 12:19:04 and gateway readiness at 12:19:07 JST. A post-recovery
OpenWebUI bridge completion returned HTTP 200 with content `rest` (one-token
confirmation). The initial eight-token bridge probe ended with
`container_transport`; this is retained in the raw artifact, and the successful
one-token probe is the response confirmation.

Raw CP evidence: `benchmarks/results/2026-07-27/hw-microbench/r9700-cp-window/`.

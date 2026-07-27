# RDNA4 (R9700) / CDNA3 (MI300X) hardware microbenchmark

Status: R9700 was measured in the CO exclusive window on 2026-07-27; MI300X is
not yet rented. Empty MI300X fields are intentionally not zero.

| Item | R9700 (gfx1201) | MI300X (gfx942) |
|---|---|---|
| STREAM read | 62.963 GB/s (9.838% of 640) | pending |
| STREAM copy | 584.167 GB/s (91.276% of 640) | pending |
| STREAM triad | 574.355 GB/s (89.743% of 640) | pending |
| BF16 GEMM 256³ / 1024³ / 4096³ | 5.433 / 17.148 / 8.915 TFLOPS (2.844 / 8.978 / 4.667%) | pending |
| FP8 GEMM 256³ / 1024³ / 4096³ | 6.727 / 14.851 / 14.604 TFLOPS (1.756 / 3.877 / 3.813%) | pending |
| BF16 Qwen3-14B `256x5120x5120` | 15.205 TFLOPS (7.961%) | pending |
| FP8 Qwen3-14B `256x5120x5120` | 23.535 TFLOPS (6.145%) | pending |
| CPU numeric oracle | pass implied by successful validate-before-measurement run; original numeric line was not persisted | pending (run on rental) |
| ISA | pass: WMMA FP8 `v_wmma_f32_16x16x16_fp8_fp8` (count 1) | offline pass: MFMA evidence in `isa/` |
| telemetry / clocks / power | window start: edge 45 C / GFX 234 MHz / 13 W; after all groups: edge 48 C / GFX 430 MHz / 13 W | pending |

Metric definition: GEMM is dense `2*M*N*K` FLOPs and STREAM byte counts are
read=4N, copy=8N, triad=12N. Each reported throughput is the median of 11 HIP
event samples after 5 warmups; each sample contains 10 kernel launches.

## R9700 CO evidence

Raw JSONL, ISA disassembly/audit, compiler log, timing, and the attempted group
telemetry are in [`r9700-co-window/`](r9700-co-window/). The measured hardware
payload took 39 whole-clock seconds from build/audit start to finish; the
benchmark's group wall times were validate 0 s, bandwidth 2 s, and GEMM 5 s.
Those wall times are operational timing only: the reported GB/s and TFLOPS use
HIP events around kernel launches, not a profiler range.

The 2026-07-27 runner passed `amd-smi metric -j`, which this installed CLI
rejects. Therefore its six per-group telemetry files record that command error,
not a fabricated measurement. The enclosing window did record the actual R9700
metrics immediately before the hardware phase and after all groups, as shown in
the table. `THROTTLED` appeared at these idle sampling points; the concrete
violation fields are N/A and the recorded GFX clocks, not that word alone, are
the clock evidence. The runner was corrected after the measurement to use
`amd-smi metric --json`, so a MI300X run will save valid before/after telemetry
per group. No second R9700 window was taken solely to replace the telemetry.

Peak comparison inputs for the future MI300X table are memory 5,300 GB/s,
BF16 1,307.4 TFLOPS, and FP8 2,614.9 TFLOPS: respectively 8.28x, 6.85x, and
6.83x R9700's 640 GB/s, 191 TFLOPS, and 383 TFLOPS. AMD's [R9700 product
page](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html)
lists 640 GB/s, FP16 matrix 191 TFLOPS, and FP8 matrix 383 TFLOPS. It does not
list a separate BF16 peak: BF16 191 TFLOPS here is the user-provided
2026-07-27 value, consistent with the RDNA4 BF16=FP16 throughput premise and
with FP8 being approximately 2x it; it is not represented as an AMD BF16 claim.

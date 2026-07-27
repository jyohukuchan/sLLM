# RDNA4 (R9700) / CDNA3 (MI300X) hardware microbenchmark

Status: R9700 was remeasured in the single CP exclusive window on 2026-07-27;
MI300X is not yet rented. Empty MI300X fields are intentionally not zero.

| Item | R9700 (gfx1201) | MI300X (gfx942) |
|---|---|---|
| STREAM read | 629.208 GB/s (98.314% of 640) | pending |
| STREAM copy | 582.743 GB/s (91.054% of 640) | pending |
| STREAM triad | 603.503 GB/s (94.297% of 640) | pending |
| BF16 GEMM 256³ / 1024³ / 4096³ | 5.533 / 17.450 / 8.943 TFLOPS (2.897 / 9.136 / 4.682%) | pending |
| FP8 GEMM 256³ / 1024³ / 4096³ | 6.132 / 15.495 / 14.401 TFLOPS (1.601 / 4.046 / 3.760%) | pending |
| BF16 Qwen3-14B `256x5120x5120` | 14.445 TFLOPS (7.563%) | pending |
| FP8 Qwen3-14B `256x5120x5120` | 22.448 TFLOPS (5.861%) | pending |
| CPU numeric oracle | pass: BF16 max abs 0; FP8 OCP/FNUZ max abs 0 | pending (run on rental) |
| ISA | pass: WMMA FP8 `v_wmma_f32_16x16x16_fp8_fp8` (count 1) | offline pass: MFMA evidence in `isa/` |
| telemetry / clocks / power | DPM gate: 3332 / 3336 / 3326 MHz; active bandwidth sample 3391 MHz / 54 C / 216 W; active GEMM samples 3297–3403 MHz / 54–57 C / 172–278 W | pending |

Metric definition: GEMM is dense `2*M*N*K` FLOPs and STREAM byte counts are
read=4N, copy=8N, triad=12N. Each reported throughput is the median of 11 HIP
event samples after 5 warmups; each sample contains 10 kernel launches.

GEMM is **naive WMMA/MFMA with no LDS tiling**, so these TFLOPS are effective
values including global-memory operand loads and output stores, not matrix-core
peak measurements. The identical naive implementation remains useful for the
cross-architecture comparison and the real projection shape; a tiled peak
benchmark is explicitly out of scope for this table.

## R9700 CP evidence

The CP raw JSONL, valid `amd-smi --json` samples, ISA audit, service evidence,
and timing are in [`r9700-cp-window/`](r9700-cp-window/). The exclusive window
ran from 12:18:22.785 to 12:19:04.691 JST (41.906 s, including stop/restore
boundary work). The benchmark phases after the DPM warmup were validate 0.193 s,
bandwidth 0.845 s, and GEMM 4.786 s. Those wall times are operational timing
only: reported GB/s and TFLOPS use HIP events around kernel launches, not a
profiler range.

Before timing, a 12-second sustained BF16 WMMA warmup recorded 35 samples. Its
last three were 3332, 3336, and 3326 MHz (the fail-closed criterion is >=1 GHz
and <=5% spread). During active bandwidth, the recorded sample was 3391 MHz,
54 C, and 216 W; during active GEMM the 11 samples were 3297–3403 MHz, 54–57 C,
and 172–278 W. Thus the former 234–430 MHz window-boundary readings were idle
values, not measurement clocks. `THROTTLED` is present in some samples but all
concrete violation fields are N/A; it is not used as a throughput conclusion.

The former 62.963 GB/s read result was a defect: 65,535 CTAs each contended on
one `atomicAdd` after reduction. The fixed kernel uses four independent load
accumulators and one non-contended checksum store per CTA (4,096 CTAs). Its ISA
contains global/flat loads and zero atomics, and the remeasurement is 629.208
GB/s, above copy. Copy changed -0.24% and triad +5.08%, consistent with the
original memory-path result rather than a clock-only explanation.

Peak comparison inputs for the future MI300X table are memory 5,300 GB/s,
BF16 1,307.4 TFLOPS, and FP8 2,614.9 TFLOPS: respectively 8.28x, 6.85x, and
6.83x R9700's 640 GB/s, 191 TFLOPS, and 383 TFLOPS. AMD's [R9700 product
page](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html)
lists 640 GB/s, FP16 matrix 191 TFLOPS, and FP8 matrix 383 TFLOPS. It does not
list a separate BF16 peak: BF16 191 TFLOPS here is the user-provided
2026-07-27 value, consistent with the RDNA4 BF16=FP16 throughput premise and
with FP8 being approximately 2x it; it is not represented as an AMD BF16 claim.

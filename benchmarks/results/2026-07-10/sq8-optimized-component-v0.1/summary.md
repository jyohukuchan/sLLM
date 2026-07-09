# SQ8 P4 one-projection optimized component

Date: 2026-07-10

## 前回の要点

P3 selected Composable Kernel ABScale on the isolated R9700/gfx1201 route. P4 had to prove the real canonical q projection with dynamic activation quantization, frozen numerical gates, inclusive timing, batch scaling, native FP8 matrix execution, and a cache-artifact check.

## 今回の変更点

- Added a deterministic CPU optimized-path oracle and reproducible M=`1,2,4,8,16,32,128` fixtures.
- Added a GPU block-K128 activation quantizer plus CK ABScale benchmark. Every runnable candidate is checked against the oracle before selection.
- Added fail-closed gfx1201 isolation, a 2 GiB working-set budget, device-memory availability checks, exact activation FP8/scale checks, and JSON schema `ullm.sq8.ck_component.v2`.
- Added a 256 MiB cache-eviction mode whose checksum is independently verified and whose dispatch is ordered before, but excluded from, each GEMM timing interval.

All points passed with 6/38 supported CK instances, six numerically valid candidates, byte-exact activation FP8 and scale output, no non-finite values, and `fallback=not_used`.

| M | warm inclusive ms | warm TFLOP/s | evicted inclusive ms | evicted TFLOP/s | reference ms | warm speedup | rel. L2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.032261 | 1.625 | 0.061161 | 0.857 | 0.191704 | 5.94x | 0.001683 |
| 2 | 0.032440 | 3.232 | 0.061421 | 1.707 | 0.326286 | 10.06x | 0.001677 |
| 4 | 0.032861 | 6.382 | 0.061761 | 3.396 | 0.598335 | 18.21x | 0.001654 |
| 8 | 0.033460 | 12.535 | 0.063001 | 6.658 | 1.154280 | 34.50x | 0.001658 |
| 16 | 0.035620 | 23.550 | 0.064920 | 12.921 | 2.252239 | 63.23x | 0.001655 |
| 32 | 0.040760 | 41.161 | 0.069580 | 24.112 | 4.355896 | 106.87x | 0.001657 |
| 128 | 0.099801 | 67.243 | 0.116621 | 57.544 | 17.315155 | 173.50x | 0.001660 |

For M=8 warm mode, quantization is 0.010800 ms p50, GEMM is 0.028300 ms, and the promotion number is 0.033460 ms inclusive. Maximum relative L2 over the grid is 0.001683 and minimum cosine similarity is 0.99999858, both inside the frozen limits of 0.005 and 0.9999.

Warm aggregate throughput grows 3.878x from M=2 to M=8. With target buffers evicted it grows 3.900x, so both exceed the recommended 2.5x target. The matched warm optimized/reference comparison passes at every M. Evicted optimized M=8 is still 18.32x faster than the warm reference; this is a conservative cache-artifact check, not a matched-cache speedup claim.

Profiler traces show cache eviction, GPU quantization, and CK OCP FP8 GEMM in order. The selected CK code objects contain native FP8 WMMA instructions. rocprofiler 1.1 returns invalid zero memory/occupancy derived counters on gfx1201; the limitation and replacement evidence are recorded in `profiler-evidence.md`.

Verification completed with 12 targeted Rust oracle tests, `cargo check -p ullm-engine --examples`, a warning-clean `hipcc -Wall -Wextra -Wpedantic` build, JSON gate validation over all 28 fixture/result files, and warm/evicted GPU execution on the isolated gfx1201 device. The Rust commands retain pre-existing anonymous-namespace linkage warnings from `ullm-runtime-sys`.

P4 status: **green**. This is component evidence, not production runtime integration.

## 次の行動

Proceed to P5. Run the same source-correct fixture, candidate validation, warm/evicted scaling, and profiler gates for k/v, gate/up, and down shapes. Freeze a measured shape/M dispatch table, then integrate one complete decoder layer with shared-input quantization and independent intermediate/final-output checks.

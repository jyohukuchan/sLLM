# SQ8 P5-A profiler evidence

## Scope

The three new projection geometries were profiled on the isolated gfx1201 device at M=8 with zero warmups, one sample, and `--cache-mode evicted`:

- k: N=1024, K=5120;
- gate: N=17408, K=5120;
- down: N=5120, K=17408.

Each run uses the corresponding real layer-0 canonical weight and scale, exact activation fixture, and CPU optimized-path oracle. q/o geometry evidence is already recorded in the P4 result directory.

## Dispatch order

The final inclusive measurement in each trace is ordered as one 256 MiB cache-eviction dispatch, one row-by-K128 activation quantization dispatch, and one CK OCP FP8 ABScale GEMM dispatch:

- k trace dispatches 25/26/27;
- gate trace dispatches 18/19/20;
- down trace dispatches 18/19/20.

The following CK dispatch is the separate untimed output-correctness run. Every profiler-run JSON reports a passing activation-byte check, passing numerical gate, and no fallback.

The CK kernels use the P3-selected code objects, which contain native `v_wmma_f32_16x16x16_fp8_fp8` instructions. The 256 MiB eviction buffer remains 32x the reported 8 MiB L2 and 4x the reported 64 MiB L3. Its whole-buffer checksum is validated before measurement.

ROCm profiler SDK 1.1 has the same gfx1201 memory/occupancy-counter limitation documented in P4. P5-A does not reinterpret the invalid zero counters. It uses dispatch traces, nonzero performance scaling, warm-versus-evicted results, verified eviction volume, and the P3 WMMA disassembly.

## Files and hashes

- `profiler/k-m8-evicted_kernel_trace.csv`: `ae0a04b1e5172519594fc02d06474d6c63e05b039ef878c476baddfbf78f1602`
- `profiler/k-m8-evicted_kernel_stats.csv`: `a3136e95c5e63e836172e5416ef939c63a0ca67026ba838b01a418cd29b197c3`
- `profiler/gate-m8-evicted_kernel_trace.csv`: `c58eed254804b13440a39d931970a9d114a159be2f942b6c7de30374825cf09e`
- `profiler/gate-m8-evicted_kernel_stats.csv`: `6cacb582a8516d8903609274738e03e239d8eefe3c128dfeb0b55066d44390ba`
- `profiler/down-m8-evicted_kernel_trace.csv`: `e5393aae3e2b1e36ea41debb463e3e08921b32128baaa85dac4f7bf37e380a98`
- `profiler/down-m8-evicted_kernel_stats.csv`: `e51ffdc74bc9bd95110fb21f67db7282be8df95b2d4f0d921f5dbf0fcd110c03`

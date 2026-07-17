# SQ8 P3 execution route selection

## 前回の要点

- P2 established a source-correct canonical q projection and a native HIP reference result.
- P3 had to choose hipBLASLt, Composable Kernel, rocWMMA, or a bounded direct kernel in that order.

## 今回の変更点

- `tools/probe-sq8-hipblaslt.cpp` demonstrated that hipBLASLt supports ordinary FP8 GEMM on gfx1201 but rejects the required 128x128 block scale.
- Added the isolated CK ABScale probe in commit `bee7b8b`.
- Identified a mixed-GPU initialization issue: CK fails when its fatbin registers against the default gfx1030 device before `hipSetDevice(1)`. The probe now resolves the requested ordinal, exposes only that token, and re-executes before HIP initialization.
- CK passed the full M grid with 6/38 supported instances and exact all-ones output. The selected family is `mem_v1_default`, tile 16x128x128.
- Permanent rocprofiler evidence and code-object hashes are stored under `benchmarks/results/2026-07-10/sq8-route-selection-v0.1/`. The gfx1201 code objects contain native `v_wmma_f32_16x16x16_fp8_fp8` instructions.
- CK is the selected P3 candidate. rocWMMA and direct HIP are no longer evaluated unless the real-weight P4 gate fails.

## 次の行動

Run the fixed real q-projection fixtures through GPU activation quantization plus CK ABScale for M=`1,2,4,8,16,32,128`, freeze numerical and inclusive timing results, and compare them with the source-correct W8A16 reference path.

# SQ8 P4 profiler evidence

## Scope

- Device: isolated `gfx1201`, physical HIP visibility token `1`, internal device `0`.
- Shape: M=8, N=5120, K=5120 with the real canonical layer-0 q projection.
- Profiled path: GPU dynamic activation quantization followed by CK ABScale FP8 GEMM.
- The normal promotion runs use 10 warmups and 50 samples. Profiler runs use zero warmups and one sample to keep dispatch attribution unambiguous.

## Dispatch and matrix path

`profiler/component-m8-evicted_kernel_trace.csv` records the final measured sequence as:

1. dispatch 19: `ullm_sq8_evict_cache`, grid 1,048,576, workgroup 256;
2. dispatch 20: `ullm_sq8_quantize_activation_block128`, grid 40,960, workgroup 128;
3. dispatch 21: CK `kernel_gemm_xdl_cshuffle_v3` with OCP FP8 A/B, grid 10,240, workgroup 256.

The eviction dispatch is queued before the HIP start event. The quantizer and GEMM are queued after it, so the eviction is ordered but excluded from the reported interval. Dispatch 22 is the separate, untimed correctness readback GEMM.

The CK code objects are the P3-selected route. Their SHA-256 values are `f311c69c7c1608b5e38ea63795376dd5e41e7897fc9892241c4b3e5c69626523` and `b010aec51d78a93ca5bb1c3742e6c8735022df535591bb85c22f3ceab5a537ef`; each contains 240 `v_wmma_f32_16x16x16_fp8_fp8` instructions. The disassembly command and example instruction are frozen in `../sq8-route-selection-v0.1/profiler-evidence.md`.

## Cache-artifact check

- `rocminfo` reports 8 MiB L2 and 64 MiB L3 for gfx1201.
- Each GEMM sample in evicted mode first reads a separate 256 MiB GPU buffer, 32x L2 and 4x L3.
- The buffer is initialized on GPU with an index hash. The GPU checksum `144116358669847907` matches an independent CPU checksum over all 67,108,864 words.
- The validation pass takes about 1.1 ms, corresponding to roughly 215-244 GB/s in the recorded runs. This cost is not included in component latency.
- Evicted M=8 remains at 0.063001 ms inclusive and 6.6575 TFLOP/s, while warm M=8 is 0.0334605 ms and 12.5351 TFLOP/s. The difference proves that cache residency matters; the passing evicted result proves that the optimized result is not cache-only.

## Counters and limitation

ROCm profiler SDK 1.1 on this gfx1201 system does not return reliable derived memory or occupancy values:

- `MeanOccupancyPerActiveCU` and `OccupancyPercent` are zero while the same dispatches report nonzero `SQ_WAVES_sum` (quantizer 1,280; CK candidates 160 or 320).
- `FETCH_SIZE` and raw `GL2C_EA_RDREQ_128B_sum` also return zero for the checksum-validated 256 MiB read kernel and CK GEMMs.
- `WRITE_SIZE` is listed for other installed agents but is unavailable for gfx1201; combining the memory and wave counters exceeds the hardware counter profile and is rejected.

These zero values are recorded as an SDK limitation, not interpreted as zero traffic or occupancy. Dispatch dimensions, LDS/VGPR use, nonzero wave counts, the verified read volume, warm-versus-evicted behavior, and native FP8 WMMA disassembly remain usable evidence.

## Files and hashes

- Evicted trace: `profiler/component-m8-evicted_kernel_trace.csv`, SHA-256 `aacdec54508a270414e9a42df3acb05cd49a23a9ae11146b293b0e81c6d639d8`.
- Evicted stats: `profiler/component-m8-evicted_kernel_stats.csv`, SHA-256 `390e699d1bd424cc4a514fe37099a62bfba16343077b04b8c77a01ed97c3d821`.
- Occupancy/wave counters: `profiler/counters/component-m8_counter_collection.csv`, SHA-256 `f63d7c3e6fe2e444f6e9cba25daf5f46705050198525ec09f20b5a50d187cc61`.
- Derived fetch counters: `profiler/counters/component-m8-evicted-fetch_counter_collection.csv`, SHA-256 `b0e5dd009e5ffba38cf6f8eccc7698487a7aa4b2081d7348f88a5e2ef22fb98c`.
- Raw L2 read counters: `profiler/counters/component-m8-evicted-read128_counter_collection.csv`, SHA-256 `04b2851050fa8774e325813bfd5c14b20f12fb8b421a560fb4cb72cb2d5276dc`.

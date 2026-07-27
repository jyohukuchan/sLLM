# MI300X rental CR follow-up

All artifacts below were copied from the rental host before handoff. SHA-256
manifests are colocated with each capture.

## Transfer gate

One 89,128,960-byte SQ8_0 FP8 weight was streamed over SSH with compression
disabled to `/dev/null` on the rental host. It took 28.323 s: **3.00 MiB/s**
(25.17 Mbit/s). At this observed rate, the 15 GB SQ8_0 package alone would
take about 85 minutes. Neither package was transferred; full-model gfx942
execution and tok/s are therefore **unconfirmed**, not a runtime failure.

## A′ effective occupancy

`occupancy/hip_launch_occupancy.cpp` interposes `hipLaunchKernel` only for the
physical smoke process. For the exact function pointer passed by CK, it calls
`hipFuncGetAttributes` and `hipOccupancyMaxActiveBlocksPerMultiprocessor`; every
API status is zero. `hip-api-occupancy.stderr` is the raw capture and
`physical.stdout` confirms the same five zero-error A′ cases and B sentinel.

| A′ case (physical-smoke order) | CK tile | registers / static LDS | active blocks/CU | active waves/CU |
|---|---|---:|---:|---:|
| `k_or_v_tail_id1` | 16x128x128 | 83 / 18,432 B | 3 | 12 |
| `q_or_o_full_id1` | 16x128x128 | 83 / 18,432 B | 3 | 12 |
| `gate_or_up_tail_id2` | 16x128x256 KPadding | 250 / 36,864 B | 1 | 4 |
| `gate_or_up_full_id3` | 16x256x128 | 158 / 34,816 B | 1 | 4 |
| `down_tail_id4` | 16x128x256 | 166 / 36,864 B | 1 | 4 |

Thus the previously extracted maximum VGPR 454 / AGPR 198 metadata is not the
observed HIP resource allocation for these launched functions. Occupancy is
limited to one CTA (four wave64 waves) for three variants, but not by the
alleged 454+198 register allocation; the API reports 158–250 registers and
34–36 KiB static LDS for those variants.

## CK LDS-tiled FP8 GEMM

`ck-lds-gemm/ck_lds_gemm.cpp` launches the same 16x128x128 RCR CK ABScale
instance at 4096³. It verifies zero FP8 payload/scales produce zero BF16 output
before timing. After a 30,000-repeat CK warmup, 20 HIP-event timed repeats
average **0.406041 ms = 338.486 TFLOPS**. `telemetry.txt` records gfx clock
2100 MHz during the sustained load. This is an FP8 matrix-core measurement with
explicitly LDS-tiled CK, but zero data does not validate the nonzero
OCP-to-FNUZ scale convention; it is not a replacement for the A′ nonzero
numerical gate.

## Time accounting

| Work | Wall time |
|---|---:|
| P0 (from `../stage-timings.tsv`) | 174 s: CPU 82, HIPRTC 32, build 54, ISA 4, physical 2 |
| one 89,128,960-byte transfer probe | 28.323 s |
| occupancy instrumentation builds | 3 × about 56.6 s; only the final preload method yielded the exact CK function API data |
| occupancy physical recheck | about 1 s |
| CK LDS benchmark build | 5 s |
| CK LDS warmup + timed run | 12 s |

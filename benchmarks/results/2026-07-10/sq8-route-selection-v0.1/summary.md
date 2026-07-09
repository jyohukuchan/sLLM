# SQ8 P3 route selection

## 前回の要点

- hipBLASLt 1.2.2 can execute scalar-scaled FP8 on gfx1201, but rejects the required 128x128 block scale.
- CK exposes 38 matching FP8 ABScale instances, while rocWMMA and a direct kernel remain bounded fallback choices.

## 今回の変更点

- Selected Composable Kernel `mem_v1_default` with the 16x128x128 block tile.
- All M=`1,2,4,8,16,32,128` runs accepted 6/38 instances, returned exact all-ones BF16 output, and reported `fallback=not_used`.
- Kernel p50 was 0.02798/0.02796/0.02824/0.02824/0.02840/0.03038/0.07080 ms in M order.
- Isolating the selected R9700 before HIP initialization is mandatory on this mixed gfx1030/gfx1201 host. Without isolation, CK fatbin registration can bind to gfx1030 and terminate the process.
- rocprofiler traces CK matrix kernels, and both dumped gfx1201 code objects contain 240 native FP8 WMMA instructions.

## 次の行動

Use this CK route for the real canonical q projection, include dynamic activation quantization in timing, compare every M point with the frozen CPU oracle and `reference_w8a16`, and only then promote the route into the runtime.

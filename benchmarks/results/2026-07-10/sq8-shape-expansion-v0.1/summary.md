# SQ8 P5-A shape expansion

Date: 2026-07-10

## 前回の要点

P4 proved the layer-0 q projection on gfx1201 with source-correct dynamic activation quantization, CK ABScale native FP8 execution, warm and cache-evicted timing, and frozen numerical gates. P5-A had to extend that evidence to every projection used by one decoder layer before production integration.

## 今回の変更点

- Built byte-exact one-tensor canonical artifacts for layer-0 o/k/v/gate/up/down. The q result is carried forward from the P4 evidence directory.
- Generated deterministic fixtures for all seven q/o/k/v/gate/up/down tensors at M=`1,2,4,8,16,32,128`.
- Recorded 49 source-correct HIP references and 98 optimized CK results: warm and target-buffers-evicted for every tensor/M pair.
- Validated every result with `tools/summarize-sq8-shape-expansion.py` and froze the exact measured CK type strings in `dispatch-table.json`.
- Profiled the three new geometries at M=8 with cache eviction: k/v `[1024,5120]`, gate/up `[17408,5120]`, and down `[5120,17408]`.

All 196 fixture/result files pass. Every optimized point reports 6/38 supported candidates, six numerically valid candidates, byte-exact activation FP8 and scale output, no NaN/Inf, and `fallback=not_used`.

| tensor | N | K | warm M8 ms | warm TFLOP/s | evicted M8 ms | reference M8 ms | warm speedup | M8/M2 throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q | 5120 | 5120 | 0.033460 | 12.535 | 0.063001 | 1.154280 | 34.50x | 3.878x |
| o | 5120 | 5120 | 0.033261 | 12.610 | 0.062941 | 1.184996 | 35.63x | 3.877x |
| k | 1024 | 5120 | 0.023660 | 3.545 | 0.028841 | 0.661352 | 27.95x | 3.851x |
| v | 1024 | 5120 | 0.023781 | 3.527 | 0.028920 | 0.595360 | 25.04x | 3.896x |
| gate | 17408 | 5120 | 0.167002 | 8.539 | 0.172021 | 4.749101 | 28.44x | 3.932x |
| up | 17408 | 5120 | 0.166921 | 8.543 | 0.172002 | 4.748551 | 28.45x | 3.936x |
| down | 5120 | 17408 | 0.161841 | 8.812 | 0.168541 | 4.149290 | 25.64x | 3.868x |

Across all M values and tensors, maximum relative L2 is 0.00169415 and minimum cosine similarity is 0.999998566. The minimum warm optimized/reference speedup is 3.25x at the smallest point. Even the conservative evicted-optimized versus warm-reference comparison remains at least 2.66x.

Every projection exceeds the recommended M=8/M=2 throughput ratio of 2.5x. The minimum is 3.851x warm and 3.884x with target buffers evicted. The shape gate is therefore green.

The measured warm dispatch families are:

- q/o and k/v, all measured M: `mem_v1_default`, block tile `16x128x128`;
- gate/up, M=1 through 32: `mem_v1_kpadding`, block tile `16x128x256`;
- gate/up, M=128: `mem_v1_default`, block tile `16x256x128`;
- down, M=1 through 32: `mem_v1_default`, block tile `16x128x256`;
- down, M=128: `mem_v1_default`, block tile `16x128x128`.

`dispatch-table.json` keeps the complete CK `GetTypeString()` values and rejects unmeasured shape/M combinations. P5-A is complete; the overall P5 phase remains in progress because the optimized primitive and one-layer runner are not yet integrated.

## 次の行動

Implement the CK projection primitive behind an explicit ROCm/gfx1201 build feature. Expose separate activation quantization and GEMM calls so QKV shares one quantization and gate/up shares one quantization. The primitive must accept only entries present in `dispatch-table.json`, return the selected implementation ID, produce F32 output for existing layer operations, and reject implicit fallback. Then build the narrow one-layer audit runner and compare its intermediate and final tensors with an independent oracle.

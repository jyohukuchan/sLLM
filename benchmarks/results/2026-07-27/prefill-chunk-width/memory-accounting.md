# SQ8_0 prefill chunk-width VRAM accounting

## Scope and method

This is an allocation-contract calculation for the current Qwen3-14B
`SQ8_0` serving implementation.  It counts buffers allocated by the serving
loader and layer workspace source, rather than treating a profiler's range or
GPU-wide used-memory number as model memory.  It therefore excludes HIP
allocator/page-table/module overhead; actual co-resident loading remains a
separate runtime observation.

The R9700 capacity is `34,208,743,424 B` (`31.859375 GiB`).  The simultaneous
`AQ4_0` Qwen3.5-9B production observation is `7,426,916,352 B`, taken from
`../aq4-runtime-hardening-activation-execute-r3/postflight.json`.  That value
is an observed process allocation, while the SQ8_0 column is a requested
buffer sum; this mixed accounting is deliberately labelled rather than
presented as a measured co-resident peak.

## Width-dependent allocations

For `H=5120`, `KV=8*128=1024`, and `I=17408`, the shared layer workspace in
`Qwen3Sq8LayerWorkspace::allocate` contains:

| component | bytes/token | derivation |
| --- | ---: | --- |
| F32 layer intermediates | 430,080 | `(10*H + 4*KV + 3*I) * 4` |
| BF16 projection workspace | 34,816 | `I * 2` (the largest projection output) |
| dynamic activation FP8 + F32 scales | 33,792 | `3*(H + H/128*4) + (I + I/128*4)` |
| shared stack workspace | 498,688 | sum of the preceding rows |
| stack resident hidden F32 | 20,480 | `H * 4` |
| prompt chunk hidden F32 | 20,480 | `H * 4` |
| **total that grows with M** | **539,648** | **527 KiB/token** |

The F32 cached-prefix Flash2 attention path has no global temporary buffer.
Its generic CTA LDS is `256*4 + 64*4 + 4*4 = 1,296 B`; the currently selected
R9700 GQA-grouped implementation uses `12,624 B` of CTA LDS
(`20*128 + 256 + 5*64 + 4*5` F32 values).  Neither is multiplied by the
resident width.  This is a per-CTA hardware allocation, not a persistent
VRAM buffer.

The fixed SQ8_0 requested allocation is `17,674,004,992 B`.  It includes the
`13,212,057,600 B` FP8 projection payload plus the **expanded F32 runtime
scale buffers** (`806,400 * 4 = 3,225,600 B`); the artifact scales themselves
are BF16 on disk (`1,612,800 B`) but are converted by
`load_sq8_canonical_resident_tensor`.  It also includes layer norms,
BF16 embedding/head, F32 head/embedding auxiliaries, F32 K/V caches and
their per-layer non-KV state, and the separate M=1 decode workspace.  Counting
the F32 scale allocation avoids undercounting this live runtime by 3,225,600 B.

| fixed live allocation | bytes |
| --- | ---: |
| FP8 projection payloads | 13,212,057,600 |
| expanded F32 projection scales | 3,225,600 |
| layer F32 norms | 1,679,360 |
| BF16 embedding | 1,555,824,640 |
| BF16 LM head | 1,555,824,640 |
| F32 embedding/head auxiliaries and logits | 689,664 |
| F32 KV cache (40 layers) | 1,342,177,280 |
| cache block tables / non-KV serving state | 2,007,040 |
| independent M=1 decode workspace | 519,168 |
| **fixed SQ8_0 total** | **17,674,004,992** |

## Capacity table

| fixed M | variable SQ8_0 bytes | SQ8_0 requested total | + observed AQ4_0 | remaining to R9700 capacity | result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 128 | 69,074,944 | 17,743,079,936 (16.525 GiB) | 25,169,996,288 (23.441 GiB) | 9,038,747,136 (8.418 GiB) | analytical fit |
| 256 | 138,149,888 | 17,812,154,880 (16.589 GiB) | 25,239,071,232 (23.506 GiB) | 8,969,672,192 (8.354 GiB) | analytical fit |
| 512 | 276,299,776 | 17,950,304,768 (16.718 GiB) | 25,377,221,120 (23.634 GiB) | 8,831,522,304 (8.225 GiB) | analytical fit |
| 1024 | 552,599,552 | 18,226,604,544 (16.975 GiB) | 25,653,520,896 (23.892 GiB) | 8,555,222,528 (7.968 GiB) | analytical fit |
| 2048 | 1,105,199,104 | 18,779,204,096 (17.489 GiB) | 26,206,120,448 (24.406 GiB) | 8,002,622,976 (7.453 GiB) | analytical fit |
| 4096 | 2,210,398,208 | 19,884,403,200 (18.519 GiB) | 27,311,319,552 (25.436 GiB) | 6,897,423,872 (6.424 GiB) | analytical fit |

Thus M=4096 is the largest requested width in the 4096-token serving
context and has material analytical headroom even beside the observed AQ4_0
process.  It has **not** been co-resident-loaded in this run, so this is not a
claim of an observed maximum.  For the requested 4095-token workload,
M=4096 is not useful under the no-padding rule: there is no earlier real
4096-token prefix to replay, so the planner deliberately keeps M=1 seeds.
M=2048 is the largest candidate that reduces the 4095-token prefill to fixed
real-token chunks.

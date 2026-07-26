# Serial GQA grouped Flash2 static audit

This is an offline HIPRTC audit of the arithmetic-schedule-preserving candidate,
not a timing result.  It was compiled with the same options used by the runtime:
`--offload-arch=gfx1201 --std=c++17 -O3`.

| body | code bytes | wavefront | VGPR | SGPR | LDS/group segment | private segment | VGPR spill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| generic F32 Flash2 | 9,664 | 32 | 21 | 46 | 1,292 B | 0 B | 0 |
| serial GQA grouped F32 Flash2 | 18,712 | 32 | 42 | 50 | 12,628 B | 0 B | 0 |

The candidate's LDS consists of one 20×128 F32 K-or-V staging allocation,
a 256-entry reduction buffer, five 64-score rows, and five scalar online
softmax states.  K and V reuse the same staging allocation, so they are never
resident simultaneously.  The metadata says wave32 and has no dynamic stack,
AGPR use, private segment, or VGPR spill.

The audit does not measure achieved occupancy, cache behavior, HBM traffic, or
throughput.  Those remain device-measurement questions.  In particular, the
candidate's 12,628 B LDS allocation is only a resource bound, not evidence of
its actual resident workgroup count.

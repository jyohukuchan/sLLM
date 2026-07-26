# `AQ4_0` grouped decode full-model window

This is the final pre-promotion A/B for the shape-closed `AQ4_0` 4:1×256
GQA grouped split implementation in source commit
`c8074928e22b27801df78d65508fdd619d13a748`.  It ran only the R9700
(`gfx1201`) at C=1339 with six warmup and 32 timed full-model decode steps per
run.  The profile driver's timed decode interval supplies tok/s; no ROCprof
range is presented as throughput.

| mode | run A tok/s | run B tok/s | paired mean tok/s |
| --- | ---: | ---: | ---: |
| direct | 74.230040 | 73.991913 | 74.110977 |
| `aq4_gqa_grouped_split` | 74.717598 | 74.302063 | 74.509830 |

The grouped specialization is **1.005382×** the same-build direct control
(+0.398854 tok/s).  This agrees with the earlier isolated trace signal and is
intentionally reported as a small full-model improvement, not as a transfer
of the `SQ8_0` 1.790050× result.

Each of the four 32-token sequences was the same (`4445` repeated).  That is a
functional diagnostic only.  The actual text-quality gate is the separate
promotion transaction in `../aq4_0-grouped-promotion-c8074928-20260727T020500Z/`.

The window stopped only `ullm-openai.service`, acquired `/run/ullm/r9700.lock`
after it was released, and restored the service.  The original active P3
manifest SHA-256 remained
`a98910dc5bf59dc768e5bcd20bcf58968699540eb1b33df33066dcb6f274fe49`
before and after this non-mutating window; `NRestarts` remained zero.

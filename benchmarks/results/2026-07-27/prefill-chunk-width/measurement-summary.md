# Full-model wide-M result

## Scope and provenance

This is an isolated full-model `SQ8_0` Qwen3-14B run on the R9700
(`gfx1201`) only.  It deliberately separates the product scheduler change
from an execution-only source overlay: the committed product code makes M
selectable and keeps its default at M=128, while the temporary overlay lifted
the BP/BX-owned lower validation lists so that M=256..2048 could be measured
without editing those shared files.  The overlay changes no Flash2 HIPRTC or
launcher body and is not a production admission change.

The performance and trace sweep is
`run-20260727T024801+0900`.  Its first numerical step correctly failed closed
because the split-decode HIP guard was omitted; timings and traces had already
completed.  Commit `88607fe0` added that guard and `43cd16dd` added the
validation-only continuation.  The successful correctness/decode/generation
continuation is `run-20260727T044042+0900` (`window-finished status=0`,
service restored active, `NRestarts=0`).  The raw files, preflight, thermal
logs, and service records remain in both run directories.

All prefill rates use the prescribed single-sequence, same-length warm-up,
five unprofiled synchronized repetitions.  They are not profiler-range
durations.  Every condition entered at edge temperature <=45 C.

## Prefill throughput

Cells are `SQ8_0 tok/s (llama.cpp tok/s divided by SQ8_0)`.  llama.cpp values
are the required Q8_0 / F32-KV references.  `*` means N is smaller than the
selected resident M, so the no-padding scheduler correctly uses audited M=1
seeds; it is not a wide-M attention result.

| resident M | 128 (llama 1165.756) | 512 (llama 1195.722) | 1024 (llama 1145.351) | 2048 (llama 1058.379) | 4095 (llama 1008.683) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 883.091 (1.320x) | 562.525 (2.126x) | 357.928 (3.200x) | 196.566 (5.384x) | 104.965 (9.610x) |
| 256 | 33.614* (34.680x) | 552.791 (2.163x) | 352.386 (3.250x) | 190.425 (5.558x) | 101.689 (9.919x) |
| 512 | 33.648* (34.645x) | 565.624 (2.114x) | 383.452 (2.987x) | 209.058 (5.063x) | 110.541 (9.125x) |
| 1024 | 33.625* (34.669x) | 33.417* (35.782x) | 388.258 (2.950x) | 224.805 (4.708x) | 121.813 (8.281x) |
| 2048 | 33.474* (34.826x) | 33.399* (35.801x) | 32.939* (34.772x) | 232.765 (4.547x) | **126.686 (7.962x)** |

At the target N=4095, M=2048 is 1.207x the same-run M=128 control and
reduces the llama.cpp gap from 9.610x to 7.962x.  M=4096 is an analytical
capacity shape but has no legal fixed real-token unit at N=4095: using it
would require a fabricated 4096th row, so it is intentionally not timed as a
wide-M case.

## Trace: actual cached-prefix attention calls at N=4095

| M | calls across 40 layers | attention time | all selected kernel time | attention share |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1,280 | 35,734.130 ms | 38,421.640 ms | 93.005% |
| 256 | 640 | 36,623.181 ms | 39,590.656 ms | 92.505% |
| 512 | 320 | 33,722.748 ms | 36,648.044 ms | 92.018% |
| 1024 | 160 | 30,220.653 ms | 33,203.580 ms | 91.016% |
| 2048 | **80** | 28,598.954 ms | 31,729.444 ms | 90.134% |

Thus the requested dispatch reduction is real (1,280 -> 80 at M=2048), but
the generic grouped Flash2 body remains the dominant cost.  Reducing dispatch
count alone is insufficient to approach llama.cpp's 40 attention calls or its
1,008.683 tok/s result.

## Numerical and generation evidence

For each candidate, every prompt at or above its selected M was F32-byte
exact for both final hidden state and logits (`max_abs=0`, no non-finite
elements), and the one-token greedy result matched.  The only byte differences
were the intended M=1 fallback cases:

| resident M | fallback prompts with non-byte-exact F32 | largest hidden `max_abs` | largest logits `max_abs` | non-finite | greedy token |
| ---: | --- | ---: | ---: | ---: | --- |
| 256 | 128 | 1.435986 | 1.100868 | 0 | matched |
| 512 | 128 | 1.435986 | 1.100868 | 0 | matched |
| 1024 | 128, 512 | 1.435986 | 1.100868 | 0 | matched |
| 2048 | 128, 512, 1024 | 1.601738 | 1.100868 | 0 | matched |

The fixed 10-case lightweight suite was 9/10 exact text for every candidate;
the only changed case was `ja_long_summary`, with a short semantic-preserving
wording variation (for example, "state" versus "data").  The policy's
obvious-collapse diagnostics reported zero blocking findings for all four
candidates.  This is qualitative evidence, not a scalar numerical gate.

The real-token N=4000 prefix exercises actual M=256/512/1024/2048 chunks and
the real-token tail replay.  All five M values generated the identical 83
token IDs and decoded 467-character text versus M=128.  The retained
completion and direct comparisons are in
`run-20260727T044042+0900/generation-long/summary.{json,md}`.

## Decode and next kernel handoff

A fresh M=128 decode measurement at prompt 1024 is **27.552769 tok/s**, above
the 27.378731 reference (and BR's 27.411786 remeasurement).  The wide-M
generation runs also complete their post-prefill M=1 decode path.

No BX-owned kernel source change was needed to *execute* wide M: CK shape
admission, the F32 paged-KV writer, and the cached-prefix launcher already
take runtime M.  A kernel-level redesign is nevertheless now required for a
material further prefill win because Flash2 remains 90.134% of selected
kernel time at M=2048.  The concrete continuation contract is recorded in
[`lower-runtime-handoff.md`](lower-runtime-handoff.md).

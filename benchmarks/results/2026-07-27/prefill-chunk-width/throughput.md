# Full-model prefill throughput and llama.cpp comparison

## Method

The five-prompt/five-width sweep is in
`run-20260727T024801+0900/throughput/`.  Each rate is the mean from five
unprofiled synchronized repetitions after the prescribed same-length warm-up;
it is not a profiler range duration.  All conditions used R9700 gfx1201 with
an edge-temperature gate of <=45 C.  The M=128 control in this run agrees with
the preceding BR control within normal run variation.

Cells below are `SQ8_0 tok/s (llama.cpp/SQ8_0)`.  llama.cpp values are the
requested Q8_0 / F32-KV references.

| resident M | 128<br>(llama 1165.756) | 512<br>(llama 1195.722) | 1024<br>(llama 1145.351) | 2048<br>(llama 1058.379) | 4095<br>(llama 1008.683) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 883.091 (1.320x) | 562.525 (2.126x) | 357.928 (3.200x) | 196.566 (5.384x) | 104.965 (9.610x) |
| 256 | 33.614* (34.680x) | 552.791 (2.163x) | 352.386 (3.250x) | 190.425 (5.558x) | 101.689 (9.919x) |
| 512 | 33.648* (34.645x) | 565.624 (2.114x) | 383.452 (2.987x) | 209.058 (5.063x) | 110.541 (9.125x) |
| 1024 | 33.625* (34.669x) | 33.417* (35.782x) | 388.258 (2.950x) | 224.805 (4.708x) | 121.813 (8.281x) |
| 2048 | 33.474* (34.826x) | 33.399* (35.801x) | 32.939* (34.772x) | 232.765 (4.547x) | **126.686 (7.962x)** |

`*` is intentional M=1 fallback, not a wide-M timing.  A resident stack has
one allocation M and cannot issue a smaller fixed chunk for an initial prompt
with `N < M` without adding a second resident shape or fabricating rows.  The
no-padding invariant therefore makes those cells materially slower; default
M=128 avoids this behavior for the normal short-prompt path.

## Target interpretation

At N=4095, M=2048 reduces the same-run M=128 time from 104.965 to 126.686
tok/s (1.207x).  The gap to llama.cpp narrows from 9.610x to 7.962x, not to
parity.  M=256 is slower than M=128 at N=4095 despite half the attention
calls; M=512 and M=1024 are intermediate.  Thus selecting a wider M is not a
monotonic speed guarantee.

M=4096 fits the allocation calculation but has no legal fixed real-token
chunk at N=4095.  A measurement would only benchmark 4095 M=1 seeds, so no
M=4096 "wide-M" rate is reported.

The actual dispatch counts and selected-kernel trace accounting are in
[`measurement-summary.md`](measurement-summary.md), not inferred from this
rate table.

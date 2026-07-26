# Execution accounting

The AY normalized logical-work denominator is unchanged.  For logical prompt
length `N`, its canonical prefill shape remains `K = ceil(N / 128)`.  The
projection, KV, and LM-head accounting remains in
[`../r9700-prefill-comparison/accounting.md`](../r9700-prefill-comparison/accounting.md).

The distinction below matters: logical prompt tokens are never duplicated in
request accounting, while the overlapping tail deliberately recomputes a
small suffix of **real** token rows to keep the resident M=128 stack fixed.

| N | canonical K | old actual advances | new actual advances | old executed rows | new executed rows | new replay rows |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 1 | 1 | 128 | 128 | 0 |
| 512 | 4 | 4 | 4 | 512 | 512 | 0 |
| 1024 | 8 | 8 | 8 | 1024 | 1024 | 0 |
| 2048 | 16 | 16 | 16 | 2048 | 2048 | 0 |
| 4095 | 32 | 158 | 32 | 4095 | 4096 | 1 |

For 4095, the old implementation did `31 × 128 + 127 × 1` calls.  The new
implementation does `31 × 128 + 128`, where the final chunk starts at 3967
and logically commits only 3968..4094.  The one additional executed row is
the real prompt token at 3967; it is not padding and it is not a second
logical token.

Other non-divisible examples follow the same rule: 1000 executes 1024 real
rows over 8 advances and commits 1000 logical tokens (24 replay rows); 129
executes 256 real rows over 2 advances and commits 129 logical tokens (127
replay rows).  The raw oracle schedules retain logical widths, and source
planner tests bind the execution widths and rewind positions.

The comparison’s common logical numerator intentionally stays at `K`; it is
not adjusted to make the new scheduler look cheaper than it is.  Actual
advance count is reported separately because it is the scheduling defect this
change removes.

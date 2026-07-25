# SQ8_1 K=32 activation outlier analysis

This is a derived summary of `per-tensor.jsonl`, not a separate activation run.  The source run
uses 64 deterministically selected real token rows per each of 248 Linear inputs (8 rows from each
of 8 real Qwen3.5-9B chat forwards), with dynamic K=32 FP16-upward-scale quantization.

## Block distribution

For every token/K=32 block, `r = max(abs(x))/RMS(x)` was computed before quantization.

| `r` bin | K=32 groups | share | tensors represented | median per-tensor activation relative L2 in bin |
| --- | ---: | ---: | ---: | ---: |
| `[1,2)` | 85,073 | 3.3285% | 248 | 0.00426518 |
| `[2,4)` | 2,032,440 | 79.5194% | 248 | 0.00653891 |
| `[4,8)` | 438,391 | 17.1521% | 248 | 0.01163607 |
| `[8,inf)` | 0 | 0% | 0 | not applicable |

The maximum observed block ratio was 5.65683.  Thus outlier-dominated blocks do increase the
local error (the `[4,8)` median is about 1.78 times the `[2,4)` median), but no observed K=32
block reached the extreme `r >= 8` bin in this sample.  Dynamic K=32 scaling confines the effect
to one 32-channel block rather than allowing one channel to set an entire projection's scale.

## Channel observations and output tails

The largest channel-level `max_t(abs(x_tj))/RMS_t(x_tj)` values are near 8 in several
`linear_attn.out_proj`, `mlp.down_proj`, and `self_attn.o_proj` inputs.  That ratio is calculated
from only 64 sampled token rows, so it has an effective sample-size ceiling near 8; it is evidence
of transient channel spikes, not a population quantile claim.

The sampled W8A8 linear-output relative-L2 distribution is median 0.0102430, p90 0.0159153,
p99 0.0262817, and maximum 0.0583117.  The maximum relative case is
`model.layers.0.linear_attn.out_proj`; its maximum absolute output error in the sampled rows is
only 0.00231166, so the high relative ratio is associated with a small sampled reference norm.
Across all 253,952 sampled outputs, W8A8 relative L2 is 0.00775109 and maximum absolute error is
0.102548.

## Design implication

No outlier separation or per-channel activation scale is admitted into the base `SQ8_1` format.
Such a path would add zero-point/per-channel correction work that weakens the direct int8-dot
inner loop.  Instead, the base path keeps symmetric dynamic K=32 scales and makes a separately
calibrated SmoothQuant-style diagonal transform or sparse outlier side path a conditional future
mitigation: it is considered only if the required full-model W8A8 logit gate fails on a larger,
held-out corpus.  Those mitigations were not measured in this run.

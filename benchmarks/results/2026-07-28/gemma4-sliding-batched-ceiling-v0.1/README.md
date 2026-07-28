# Gemma4 sliding-reader batching ceiling v0.1

## Decision

Proceed with a Gemma-specific batched sliding reader.  The prior rejection
was correct for the pre-batched full-attention implementation but is no longer
applicable: after the seven full layers moved to their 512-wide batched reader,
the remaining 28 sliding layers dominate the complete resident attention
region.

| prompt | sliding attention region | full attention region | sliding share | ideal attention-region ceiling | ideal whole-prefill ceiling |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 4.691891 s | 1.275725 s | 78.623% | 4.678x | 1.579x |
| 2048 | 26.505603 s | 15.607216 s | 62.940% | 2.698x | 1.630x |

The attention-region ceiling assumes the sliding region takes zero time while
the currently measured full region remains unchanged.  The whole-prefill
ceiling uses the same impossible assumption against the measured cold-prefill
wall time: `wall / (wall - sliding_region)`.  It is the decision metric, and
it clears the project's 1.10x implementation bar at both contexts.

## Method and scope

`ullm-gemma4-resident --mode attention-profile` ran one cold prefill (no
warmup or decode) on HIP ordinal 1 / R9700 gfx1201 after
`cargo clean -p ullm-runtime-sys` and a release relink.  Each run held
`/run/ullm/r9700.lock`; `ullm-openai` was restored afterward and is active
with zero restarts.

DT's complete attention-region timer originally covered the M=1 resident
attention function only.  DV's layer-major path bypasses that function, so it
reported a misleading zero split.  This increment records the matching
input-norm through post-attention-residual region in the layer-major path,
classified by the same Gemma4 full/sliding layer map.  It does not change
execution.  Region calls are batched-layer chunks (512: 112 sliding / 28 full;
2048: 448 sliding / 112 full), while physical reader launches are separately
validated at 14,364 and 57,456: `28 * N` sliding M=1 launches plus
`7 * ceil(N / 128)` full-reader launches.

The two pre-instrumentation JSON reports are retained under `raw/superseded/`.
Their zero region counters are intentionally not used: they document the
instrumentation hole discovered during this measurement, rather than a timing
result.

No `runtime/src` source changed, so runtime translation-unit guard
`8e7da3071dc0c68be61978818ec264aa6ccca9ed4d50416fc9a8a7e7b06ee9b3` remains
applicable.


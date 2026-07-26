# `AQ4_0` 4:1×256 GQA grouped-decode prototype

## Decision

BH's `SQ8_0` Qwen3-14B body is **not** directly reusable: it is a 5:1,
head/value-dimension 128 implementation with split tile 20.  The served
`AQ4_0` Qwen3.5-9B is 4:1 and head/value dimension 256, and its production
split configuration is tile 128 with the 256-token crossover.  Enabling the
BH selector against the old `AQ4_0` runtime would therefore select its generic
fallback, not the BH body.

The GQA-cooperation idea is nevertheless applicable.  An isolated prototype
adds a shape-closed 4:1×256 body: four consumer wave32s compute the four query
heads for one KV head while four other wave32s stage each 256-element K and V
row once in LDS.  It writes the existing split workspace and leaves the merge
kernel and every non-matching shape on the existing implementation.  This is a
new `AQ4_0` specialization, not a claim that the literal `SQ8_0` tile-20 code
was reused.

The runtime source files were concurrently owned by the prefill-attention
workstream.  The prototype was consequently built and run only from the
isolated worktree `/tmp/ullm-bq-aq4-grouped-13Mlo6`; no concurrent source edit
was overwritten.  Integration must rebase the small shape-gated diff onto the
then-current owner version, followed by a new release build and candidate
quality run.

## Measured full-model result

The profile uses Qwen3.5-9B `AQ4_0`, C=1339, six warmup decode steps and 32
timed decode steps.  Its tok/s fields come from the profile driver's measured
decode interval, not a ROCprof range.  Both windows acquired
`/run/ullm/r9700.lock` only after stopping the active service and restored that
service with `NRestarts=0`; the active P3 manifest SHA-256 was unchanged
before and after each window:

`a98910dc5bf59dc768e5bcd20bcf58968699540eb1b33df33066dcb6f274fe49`.

| window | direct tok/s | grouped tok/s | ratio | interpretation |
| --- | ---: | ---: | ---: | --- |
| `aq4_0-grouped-prototype-window-FfrGd2` | 69.256, 73.890 | 74.707, 74.386 | 1.04155 mean | first direct pass was cold/variable; do not use this as the estimate |
| `aq4_0-grouped-trace-window-EZieKA` | 73.923, 73.760 | 74.434, 74.044 | **1.005378** | stable repeat used for the candidate estimate |

Each direct/grouped pair in the stable repeat generated the same 32 token IDs
(all `4445`).  That is a narrow functional diagnostic only; it is not an
output-quality gate.  The credible full-model result is therefore a modest
**73.842 → 74.239 tok/s** (+0.397 tok/s, about +0.54%), not the 4.16% first-
window arithmetic mean and not the 1.790050x `SQ8_0` result.

The prototype profile binary SHA-256 was
`4a11fa29157c9d5b3d4383c14514e9ab0c616e2a48e8d86a6faa091f355cb668` in both
windows.  Its source/binary provenance is intentionally separate from the
served P3 worker; it is not a production deployment artifact.

## ROCprof confirmation

The candidate trace in `aq4_0-grouped-trace-window-EZieKA/` attributes only
`hipModuleLaunchKernel`-correlated dispatches whose launch began inside an
outer `AQ4_0` decode marker.  It is inclusive kernel-time composition, not
throughput.

| route at C=1339 | partial grid X | partial LDS / VGPR | partial mean ns | split-attention time | share of inclusive decode kernel time |
| --- | ---: | ---: | ---: | ---: | ---: |
| current P3 direct-per-Q-head split | 45,056 (= 16 × 11 × 256) | 1,024 B / 32 | 140,616 | 37.379 ms | 9.08552% |
| 4:1×256 grouped prototype | 11,264 (= 4 × 11 × 256) | 3,584 B / 48 | 132,217 | 35.315 ms | 8.59574% |

The candidate has four times fewer partial workgroups and a 6.35% lower mean
partial-kernel duration.  Its merge is slightly slower (5,733 vs 5,395 ns per
marker-attributed launch), so the net split-attention time is 5.52% lower.
The current-P3 9.08552% attention share makes that a roughly 0.5% Amdahl-scale
effect, consistent with the stable full-model repeat.  It is much smaller than
the `SQ8_0` result because only eight of 32 layers use full attention and the
4:1×256 body costs materially more LDS/VGPR resources.

The raw candidate trace retains Agent 2 (`gfx1201`) dispatches only.  It has
zero marker-attributed `ullm_paged_decode_attn_f32_kernel` launches, 256 split
partial launches and 256 split merge launches, so it also reconfirms that this
`AQ4_0` path is not the direct paged-attention kernel route.

## Status at this evidence point

The prototype establishes technical applicability of a new `AQ4_0`
specialization and a small positive full-model signal.  It does **not** yet
authorize production promotion: the final candidate must use the integrated
source, a typed served-model manifest, a fresh worker binary, the same-model
ten-prompt quality comparison, and the promotion/rollback tooling.  `SQ8_0`
remains a separately evaluated candidate and must never replace the active
`AQ4_0` product through `active.json`.

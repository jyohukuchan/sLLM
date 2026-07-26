# SQ8 numerical gate v0.2 — preliminary tile evaluation

## Verdict

This is a **preliminary** evaluation, not an admission result. Both private
source-tile candidates failed the v0.2 preliminary metric subset:

| candidate | multi-tile exercised | preliminary outcome | speed measurement |
| --- | ---: | --- | --- |
| tile 128 | 645 sequential-M=1 positions (547 prompt, 98 decode) | `fail_metric_subset` (90 failures) | not run |
| tile 256 | 64 sequential-M=1 prompt positions | `fail_metric_subset` (10 failures) | not run |

The failure rule is intentional: no end-to-end decode timing was taken for a
numerically failed candidate. Consequently, this run does **not** reproduce or
support the previous `1.2365×` observation. Flash2 was not run because the
requested tile 128 then tile 256 priority consumed the available window.

## Fixed input set and provenance

The read-only snapshot is
[`reference-snapshot-2160.json`](reference-snapshot-2160.json), whose SHA-256
is `d0ac40dfc5d911f7356b7d93d7469f35439d5b823fa92c3b4231aad9d7baa540`.
It is the complete list of the 2,160 source positions used here; its canonical
position-manifest SHA-256 is
`967a85d76cc64594de0ae6a071bf082babfb56b8ebf9ef4fe6b0925844ae318c`.
The source root
`../sq8-fp32-reference/cpu-f32-parallel-reference-v1/` was read only.

| mode / case | source positions | GPU-capture executable positions |
| --- | ---: | ---: |
| M=128 `chat-p2048-g512` | 228 | 1 |
| M=128 `chat-p3584-g512` | 328 | 2 |
| M=128 `raw-p4095-g1` | 319 | 2 |
| sequential M=1 `chat-p2048-g512` | 242 | 242 |
| sequential M=1 `chat-p3584-g512` | 246 | 246 |
| sequential M=1 `raw-p0001-g1024` | 226 | 226 |
| sequential M=1 `raw-p1023-g4` | 251 | 251 |
| sequential M=1 `raw-p4095-g1` | 320 | 320 |
| **total** | **2,160** | **1,290** |

The 870 difference is the M=128 interior positions which this capture route
cannot materialize individually. It is a coverage limitation, not a passed
substitute for the frozen v0.2 M=128 requirement.

## Authoritative receipts

Use the rerun receipts in
[`attempt-3/evaluations-recomputed/`](attempt-3/evaluations-recomputed/):

- [`tile128.json`](attempt-3/evaluations-recomputed/tile128.json) and
  [`tile128.md`](attempt-3/evaluations-recomputed/tile128.md)
- [`tile256.json`](attempt-3/evaluations-recomputed/tile256.json) and
  [`tile256.md`](attempt-3/evaluations-recomputed/tile256.md)

The earlier `attempt-3/evaluations/` copies are retained but superseded. They
classified source-tile exposure as decode-only; the rerun correctly counts all
sequential-M=1 `PagedDecodeState` work, including prompt positions. Apart from
`selector_exposure`, the numerical result objects are identical.

The compact machine-readable handoff is
[`preliminary-result.json`](preliminary-result.json). It records every hard
top-1 regression, representative metric/position failures, immutable hashes,
the three service windows, and the non-admission status.

## Failures that determine the verdict

Tile 128 has, among others, logits `max_abs` `2.543147087097168` versus a
`1.990684199333191` threshold at
`sequential_m1:raw-p0001-g1024:decode:00188`, final-hidden P99 relative-L2
`0.20372229973103717` versus `0.17362422008268527` at the same position, and
top-1 Wilson lower bound `0.9678961737926518` versus `0.9731245214414421`.
It has 13 hard top-1 regressions; the full list is in the JSON receipt.

Tile 256 did exercise its multi-tile branch: layer-05 P99 relative-L2 is
`0.04898892478442091` versus `0.04881303307700018` at
`sequential_m1:raw-p4095-g1:prompt:00287`; its top-1 Wilson lower bound is
`0.9696649304986341` versus `0.9731245214414421`. Its six hard top-1
regressions are at raw-p4095 prompt positions 262, 294, 299, 302, 304, and
318.

## Coverage and Wilson limit

The one-sided 95% zero-error Wilson lower bound for 2,160 source positions is
`99.8749001%`; for the formal 4,096 primary positions it is `99.9339903%`.
That is a `0.0590902` percentage-point shortfall even before accounting for the
actual 1,290-position GPU capture set. This run also has one shared control and
one capture per candidate, rather than the formal three-control / two-candidate
repeat envelope. Its repeat envelope is therefore not estimated.

## Isolation and restoration

Only R9700 GPU 2 (`0000:47:00.0`, `gfx1201`) was used. The third and final
window captured the shared control, tile 128, and tile 256 together, then
restored `ullm-openai.service` to `active/running`, `Result=success`,
`NRestarts=0`. Two earlier stop/restore windows ended before candidate results
because they exposed deterministic harness accounting and capture-identity
errors; both were repaired before the final integrated window. No production
default, active manifest, activation, campaign, or remote state changed.

## Interpretation

llama.cpp's partial `(max, sum, weighted-V)` reweight/merge demonstrates that
the split-KV algorithm class can work, but does not make this implementation
numerically conformant. Here both candidates are worse than the matched direct
control against the same frozen FP32 reference. The known differing association
of per-tile online-softmax merges, amplified by SQ8 activation quantization, is
consistent with the result; the first quantizer/lane-level step responsible is
**unconfirmed**.

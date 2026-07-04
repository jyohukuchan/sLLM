# Qwen layer 8 MLP tracked column diagnostics

Artifacts:

- `qwen-layer-module-trace-actual-input-rotary64-layer8-token7-hidden3994-mlp-proj-rowdot-trackedcols-layer6-layer10-p4p46-inproj.jsonl`
- `qwen-layer-module-trace-actual-input-rotary64-layer8-token7-hidden3994-mlp-proj-rowdot-trackedcols-layer6-layer10-p4p46-inproj.md`

## Trace Change

`export-qwen-layer-module-trace.py` now supports repeatable
`--tracked-column COLUMN` arguments. For each selected projection row, the trace
still emits `top_dot_error_terms`, and also emits `tracked_dot_error_terms` for
the requested columns on every token.

This is needed because a column can be important for downstream compensation
even when it is not in the top dot-error terms for a particular token.

## Tracked Columns

The first tracked run used the union of token `3` and token `7` top dot-error
terms for `mlp.up_proj[6340]` and `mlp.gate_proj[6340]`:

`22, 220, 310, 577, 933, 1304, 1571, 1679, 1726, 1778, 2086, 2560, 2805, 3098, 3115, 3384, 3461, 3608, 3842, 3908, 3994`

## Key MLP Up Candidates

For `mlp.up_proj[6340]`, column `3994` has the largest known leverage but moves
token `3` and token `7` in the same input direction:

| column | token 3 input | token 7 input | weight error | token 3 dot term | token 7 dot term |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3994 | 20.500000 | 23.625000 | 0.001060490 | 0.021740036 | 0.025054066 |

This explains the source-restore and LS-fit results: changing column `3994`
alone can improve the token `7` target while worsening the token `3` max.

Some tracked `mlp.up_proj[6340]` columns have opposite token signs and are better
candidates for a downstream hidden-error fit:

| column | token 3 input | token 7 input | weight error | source-restore direction | downstream-fit note |
| ---: | ---: | ---: | ---: | --- | --- |
| 933 | -1.851562 | 0.886719 | -0.002150904 | positive delta | negative delta would raise token 3 and lower token 7 |
| 3461 | 0.165039 | -1.992188 | 0.001189880 | negative delta | positive delta would raise token 3 and lower token 7 |
| 3608 | 0.398438 | -0.902344 | 0.000288365 | negative delta | positive delta would raise token 3 and lower token 7 |

These are not automatically good package corrections. The important point is
that a downstream objective may need deltas that are not source-restoration
deltas.

## Interpretation

- Column `3994` is a high-leverage reconstruction error, but it is not a good
  one-dimensional correction direction for the full-prefix objective.
- Token `3` and token `7` need at least a two-direction fit: one direction for
  reducing the token `7` chain and another for avoiding or reversing the token
  `3` hidden `3994` worsening.
- The next experiment should fit a small set of cells such as
  `up_proj[6340,{3994,933,3461,3608}]` directly against downstream hidden
  errors, not source row-dot error.

# Qwen layer 8 MLP sparse cell-delta experiment

Artifacts:

- `package-cell-delta-overrides-layer8-up6340-col3994-p4p46-inproj.json`
- `package-cell-delta-overrides-layer8-gateup6340-col3994-p4p46-inproj.json`
- `package-cell-delta-overrides-layer8-up6340-col3994-lsfit-p4p46-inproj.json`
- `package-cell-delta-overrides-layer8-gateup6340-col3994-lsfit-p4p46-inproj.json`
- `package-golden-prefix-cpu-actual-prefix0-12-rotary64-manifest-row-scale-layer6-layer10-cell-delta-layer8up6340col3994-p4p46-inproj.jsonl`
- `package-golden-prefix-cpu-actual-prefix0-12-rotary64-manifest-row-scale-layer6-layer10-cell-delta-layer8gateup6340col3994-p4p46-inproj.jsonl`
- `package-golden-prefix-cpu-actual-prefix0-12-rotary64-manifest-row-scale-layer6-layer10-cell-delta-layer8up6340col3994-lsfit-p4p46-inproj.jsonl`
- `package-golden-prefix-cpu-actual-prefix0-12-rotary64-manifest-row-scale-layer6-layer10-cell-delta-layer8gateup6340col3994-lsfit-p4p46-inproj.jsonl`

## Implementation

`package-golden-prefix-smoke` now accepts an optional smoke-only
`CELL_DELTA_OVERRIDES_JSON` positional argument after `SAMPLED_TOKEN_INDICES`.

Schema:

```json
{
  "schema_version": "package-cell-delta-overrides-v0.1",
  "overrides": [
    {
      "layer_index": 8,
      "tensor_suffix": "mlp.up_proj.weight",
      "row_index": 6340,
      "col_index": 3994,
      "delta": -0.0010604895651340485
    }
  ]
}
```

The override is applied after AQ4 materialization by copying the materialized
F32 matrix to host, adding `delta` to the selected cell, and copying the matrix
back to the runtime buffer. This is intentionally smoke-only and does not alter
package metadata or the production loader.

## Tested Cells

Layer `8`, row `6340`, column `3994` was selected from the dot-term trace:

| projection | package weight | source weight | delta applied |
| --- | ---: | ---: | ---: |
| `mlp.up_proj.weight` | 0.006980899721 | 0.005920410156 | -0.001060489565 |
| `mlp.gate_proj.weight` | -0.003788416740 | -0.002746582031 | 0.001041834708 |

The smoke report confirms the up-only override changed
`mlp.up_proj[6340,3994]` from `0.006980899721` to `0.005920410156`.

## Least-Squares Cell Fit

A second pass fitted the same column `3994` cell against all `16` tokens for
the selected row, minimizing package-vs-source row-dot error:

| projection | source-restore delta | LS delta | row-dot RMSE before | row-dot RMSE after |
| --- | ---: | ---: | ---: | ---: |
| `mlp.up_proj[6340,3994]` | -0.001060489565 | -0.001297428526 | 0.032796258 | 0.015284518 |
| `mlp.gate_proj[6340,3994]` | 0.001041834708 | 0.001825227037 | 0.045137044 | 0.019261286 |

## Full Prefix Results

All runs use:

- package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer10.ullm.d`
- fixture: `benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16`
- run mode: `actual_prefix`
- layers: `0..12`
- `rotary_dim=64`
- CPU backend

| variant | layer 8 max_abs | layer 8 max token/hidden | layer 11 max_abs | layer 11 max token/hidden |
| --- | ---: | --- | ---: | --- |
| baseline layer6/layer10 row-scale | 0.578010559 | token 3 / hidden 3994 | 0.645338058 | token 7 / hidden 3994 |
| up cell `6340,3994` | 0.580806732 | token 3 / hidden 3994 | 0.654584885 | token 7 / hidden 3994 |
| gate+up cell `6340,3994` | 0.583854675 | token 3 / hidden 3994 | 0.654893875 | token 7 / hidden 3994 |
| up cell `6340,3994` LS fit | 0.581432343 | token 3 / hidden 3994 | 0.656669617 | token 7 / hidden 3994 |
| gate+up cell `6340,3994` LS fit | 0.586801529 | token 3 / hidden 3994 | 0.657253265 | token 7 / hidden 3994 |

Layer `8`, token `7`, hidden `3994` improves locally:

| variant | token 7 hidden 3994 output diff | token 7 hidden 3994 MLP output |
| --- | ---: | ---: |
| baseline | 0.296178818 | 0.077883139 |
| up cell `6340,3994` | 0.284414291 | 0.066119172 |
| gate+up cell `6340,3994` | 0.283128738 | 0.064832471 |
| up cell `6340,3994` LS fit | 0.281785965 | 0.063490726 |
| gate+up cell `6340,3994` LS fit | 0.279504776 | 0.061208840 |

However, layer `8`, token `3`, hidden `3994` worsens:

| variant | token 3 hidden 3994 output diff | token 3 hidden 3994 MLP output |
| --- | ---: | ---: |
| baseline | -0.578010559 | 0.160099775 |
| up cell `6340,3994` | -0.580806732 | 0.157302916 |
| gate+up cell `6340,3994` | -0.583854675 | 0.154254228 |
| up cell `6340,3994` LS fit | -0.581432343 | 0.156677932 |
| gate+up cell `6340,3994` LS fit | -0.586801529 | 0.151308775 |

## Interpretation

- The sparse cell correction is mechanically valid and hits the intended
  materialized matrix value.
- Returning the high-leverage cell to the source weight improves the target
  layer `8`, token `7`, hidden `3994` coordinate.
- Least-squares fitting of the same cell improves row-dot RMSE and the token `7`
  target further, but worsens token `3` and the full-prefix layer `11` max even
  more.
- This confirms the row-scale result with a narrower intervention: the objective
  cannot be "restore one suspicious source cell" or "minimize one row-dot error".
  The correction must be fitted against a downstream multi-token hidden-error
  objective, or the quantizer must handle the full activation-weighted row/group
  error rather than one cell independently.

Next useful experiment: solve a small least-squares cell/group compensation
using multiple tokens for `mlp.up_proj[6340]` and `mlp.gate_proj[6340]`, then
evaluate full prefix before considering any package metadata promotion.

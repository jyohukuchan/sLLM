# Qwen layer 8 MLP projection row-scale experiment

Baseline package:

- `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer10.ullm.d`

Derived packages:

- `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer8up-layer10.ullm.d`
- `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer8gateup-layer10.ullm.d`
- `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer8upfit-layer10.ullm.d`
- `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer8gateupfit-layer10.ullm.d`

## Row Scales

Single-token scales were fit from layer `8`, token `7`, hidden `3994`, MLP feature `6340`.
All-token scales were fit from the same selected projection row across all `16` tokens in
the full-reference actual-input trace.

| variant | gate scale | up scale | notes |
| --- | ---: | ---: | --- |
| baseline | - | - | layer6/layer10 row-scale only |
| layer8 up | - | 1.082882107 | token7 source/package row-dot |
| layer8 gate+up | 1.016960733 | 1.082882107 | token7 source/package row-dot |
| layer8 upfit | - | 1.035102073 | all-token LS fit |
| layer8 gate+upfit | 1.011957038 | 1.035102073 | all-token LS fit |

All-token fit quality for feature `6340`:

| projection | optimal scale | original RMSE | scaled RMSE | improvement | original max abs | scaled max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlp.gate_proj[6340]` | 1.011957038 | 0.045137044 | 0.012981628 | 0.712395264 | 0.072030053 | 0.024087158 |
| `mlp.up_proj[6340]` | 1.035102073 | 0.032796258 | 0.025626328 | 0.218620371 | 0.053133676 | 0.044085904 |

## Full Prefix Result

Command class:

- `package-golden-prefix-smoke ... 0 12 64 10000000 0 ... actual_prefix none none 7`

| variant | prefix max abs | max layer | token | hidden | diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.645338058 | 11 | 7 | 3994 | -0.645338058 |
| layer8 up | 0.653738022 | 11 | 7 | 3994 | -0.653738022 |
| layer8 gate+up | 0.655612946 | 11 | 7 | 3994 | -0.655612946 |
| layer8 upfit | 0.648880005 | 11 | 7 | 3994 | -0.648880005 |
| layer8 gate+upfit | 0.650119781 | 11 | 7 | 3994 | -0.650119781 |

Layer `8` max location:

| variant | layer8 max abs | token | hidden | diff |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.578010559 | 3 | 3994 | -0.578010559 |
| layer8 up | 0.590913773 | 3 | 3994 | -0.590913773 |
| layer8 gate+up | 0.594572067 | 3 | 3994 | -0.594572067 |
| layer8 upfit | 0.583475113 | 3 | 3994 | -0.583475113 |
| layer8 gate+upfit | 0.585937500 | 3 | 3994 | -0.585937500 |

## Target Coordinate

Layer `8`, token `7`, hidden `3994`, compared against the full-reference actual-input trace:

| variant | package output diff | local delta error | MLP activation-path error | MLP feature 6340 activation diff |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.296178818 | 0.171178818 | 0.070061073 | 0.269271016 |
| layer8 up | 0.279756546 | 0.154756546 | 0.053637680 | 0.121943831 |
| layer8 gate+up | 0.275905609 | 0.150905609 | 0.049786586 | 0.087397814 |
| layer8 upfit | 0.289222717 | 0.164222717 | 0.063105538 | 0.206876397 |
| layer8 gate+upfit | 0.286626816 | 0.161626816 | 0.060509227 | 0.183586121 |

## Interpretation

- Internal projection row-scale improves the targeted layer `8` token `7` coordinate, especially the MLP feature `6340` activation drift.
- The same row-scale worsens the layer `8` token `3` max and propagates to a worse layer `11` max.
- All-token least-squares scales reduce the overfit, but still do not beat the layer6/layer10 baseline.
- Simple scalar row-scale is therefore not a good promotion candidate for `mlp.gate_proj[6340]` or `mlp.up_proj[6340]`.
- The next useful direction is a row reconstruction or quantizer-side calibration experiment for sensitive internal projection rows, optimized against multi-token downstream loss/error rather than a single projection row-dot.

# Qwen layer 8 MLP projection row reconstruction notes

Artifacts:

- `qwen-layer-module-trace-actual-input-rotary64-layer8-token7-hidden3994-mlp-proj-rowdot-tokenfit-dotterms-layer6-layer10-p4p46-inproj.jsonl`
- `qwen-layer-module-trace-actual-input-rotary64-layer8-token7-hidden3994-mlp-proj-rowdot-tokenfit-p4p65-inproj.jsonl`
- `qwen-layer-module-trace-actual-input-rotary64-layer8-token7-hidden3994-mlp-proj-rowdot-tokenfit-p4p6.jsonl`
- `package-row-quant-error-layer8-mlp-gate-up-row6340-p4p46-inproj.json`
- `package-row-quant-error-layer8-mlp-gate-up-row6340-p4p46-inproj.md`

## Existing Policy Coverage

Layer `8` MLP projection rows are identical across the checked p4p46, p4p65, and p4p6 packages:

| tensor | p4p46 | p4p65 | p4p6 |
| --- | --- | --- | --- |
| `mlp.gate_proj.weight` | `aq4_e4m3_g16_ts_flloyd16` | `aq4_e4m3_g16_ts_flloyd16` | `aq4_e4m3_g16_ts_flloyd16` |
| `mlp.up_proj.weight` | `aq4_e4m3_g16_ts_flloyd16` | `aq4_e4m3_g16_ts_flloyd16` | `aq4_e4m3_g16_ts_flloyd16` |

So the existing p4p65-inproj and p4p6 package variants do not change the
`mlp.gate_proj[6340]` or `mlp.up_proj[6340]` row reconstruction.

## Weight Row Error

Package row quantization error for layer `8`, row `6340`:

| tensor | row RMS | row relative MSE | row max abs | source RMS | recon RMS | worst group |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlp.gate_proj.weight` | 0.000785230 | 0.004946714 | 0.003092449 | 0.011164477 | 0.011129792 | 38 |
| `mlp.up_proj.weight` | 0.000716250 | 0.005367165 | 0.003012434 | 0.009776694 | 0.009735063 | 162 |

The row-level error is modest. The downstream dot error is large because a few
small weight errors are multiplied by large `post_normed` inputs.

## Dot Error Terms

Layer `8`, token `7`, feature `6340`:

| projection | row-dot error | top column | group | input | weight error | top term | top term / error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlp.up_proj[6340]` | 0.036712819 | 3994 | 249 | 23.625000 | 0.001060490 | 0.025054066 | 0.682 |
| `mlp.gate_proj[6340]` | -0.072030053 | 3994 | 249 | 23.625000 | -0.001041835 | -0.024613345 | 0.342 |

Additional `mlp.up_proj[6340]` tokens:

| token | row-dot error | top column | input | top term | top term / error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0.053133676 | 3994 | 20.500000 | 0.021740036 | 0.409 |
| 10 | 0.052790132 | 3994 | 24.000000 | 0.025451750 | 0.482 |

For `mlp.gate_proj[6340]`, token `3` has total row-dot error `0.006678195`,
while column `3994` contributes `-0.021357612`; other columns cancel it. This
helps explain why scalar row-scale is brittle: it moves many terms together,
including terms that already cancel.

## Interpretation

- The problem is not an obviously bad whole row. It is an activation-weighted row error where column `3994` is repeatedly high leverage.
- Existing p4p65/p4p6 package variants do not exercise this MLP row, so they cannot answer the current question.
- Scalar row-scale is too broad for this failure mode. A better next experiment is sparse column/group compensation or activation-weighted quantization for sensitive MLP projection rows, with the objective evaluated over multiple tokens and downstream hidden error.

# Qwen layer 8 token 7 hidden 3994 MLP projection row-dot diagnostic

Fixture:

- `benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16`

Package:

- `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer10.ullm.d`

Input dump:

- `/tmp/qwen35-9b-prefix0-12-layer6-layer10-actual-inputs-rotary64-p4p46-inproj`

Generated artifacts:

- `qwen-layer-module-trace-actual-input-rotary64-layers7-9-token7-hidden3994-mlp-proj-rowdot-layer6-layer10-p4p46-inproj.jsonl`
- `qwen-layer-module-trace-actual-input-rotary64-layers7-9-token7-hidden3994-mlp-proj-rowdot-layer6-layer10-p4p46-inproj.md`
- `qwen-module-trace-comparison-actual-input-rotary64-layers7-9-token7-hidden3994-mlp-proj-rowdot-layer6-layer10-p4p46-inproj.json`
- `qwen-module-trace-comparison-actual-input-rotary64-layers7-9-token7-hidden3994-mlp-proj-rowdot-layer6-layer10-p4p46-inproj.md`

Trace changes:

- `export-qwen-layer-module-trace.py` schema bumped to `qwen-layer-module-trace-v0.6`.
- Added `--token-index` so projection row-dot traces can target a fixed token instead of each layer's max-delta token.
- Added `projection_row_dot.mlp_gate_projection` and `projection_row_dot.mlp_up_projection` for both linear-attention and self-attention layers.
- MLP projection row-dot feature selection now follows top `mlp_activation` features for the traced token.

Layer `8`, token `7`, hidden `3994`:

| item | value |
| --- | ---: |
| package local delta error vs fullref actual-input trace | 0.171178818 |
| attention row-only error | -0.053268506 |
| attention activation-path error | 0.125136728 |
| MLP row-only error | -0.017845260 |
| MLP activation-path error | 0.070061073 |
| MLP activation feature `6340` diff | 0.269271016 |

Runtime MLP feature `6340` comparison:

| stage | fullref | package | diff |
| --- | ---: | ---: | ---: |
| MLP gate projection | 4.312500000 | 4.261680126 | -0.050819874 |
| MLP gate SiLU | 4.255476952 | 4.202431202 | -0.053045750 |
| MLP up projection | -0.480468750 | -0.422994196 | 0.057474554 |
| MLP activation | -2.046875000 | -1.777603984 | 0.269271016 |

Projection row-dot with the full-reference `post_normed` vector:

| row | module output | source row dot | package row dot | pkg-source | pkg-module |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mlp.gate_proj[6340]` | 4.312500000 | 4.318901433 | 4.246871380 | -0.072030053 | -0.065628620 |
| `mlp.up_proj[6340]` | -0.480468750 | -0.479665107 | -0.442952289 | 0.036712819 | 0.037516461 |

MLP feature `6340` product split, using captured runtime values:

| term | value |
| --- | ---: |
| `(pkg_gate_silu - ref_gate_silu) * ref_up` | 0.025486825 |
| `ref_gate_silu * (pkg_up - ref_up)` | 0.244581638 |
| interaction term | -0.003048781 |
| product-space sum | 0.267019682 |
| captured activation diff | 0.269271016 |

Interpretation:

- The layer `8` MLP feature `6340` activation drift is still mainly the `up_proj` runtime difference multiplied by a large positive gate-SiLU value.
- The `up_proj[6340]` package row-dot error on the full-reference input explains about `0.0375 / 0.0575`, or roughly 65%, of the runtime `up_proj` difference. The rest is attributable to input-path drift into `post_normed`.
- `gate_proj[6340]` also has measurable package row-dot error, but its product contribution is smaller because it is multiplied by the reference `up_proj` value.
- The next focused experiment should target internal MLP projection row sensitivity, especially `mlp.up_proj[6340]`, rather than adding another final `down_proj` or attention output row-scale override.

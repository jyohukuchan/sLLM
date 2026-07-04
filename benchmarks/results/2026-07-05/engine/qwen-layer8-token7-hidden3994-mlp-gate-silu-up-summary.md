# Qwen layer 8 token 7 hidden 3994 MLP gate/up diagnostic

Fixture:

- `benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16`

Package:

- `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer10.ullm.d`

Input dump:

- `/tmp/qwen35-9b-prefix0-12-layer6-layer10-actual-inputs-rotary64-p4p46-inproj`

Generated artifacts:

- `package-golden-prefix-cpu-actual-prefix0-10-rotary64-manifest-row-scale-layer6-layer10-sample-t7-mlp-gate-silu-up-p4p46-inproj.jsonl`
- `qwen-layer-module-trace-actual-input-rotary64-layers7-9-hidden3994-mlp-gate-silu-up-layer6-layer10-p4p46-inproj.jsonl`
- `qwen-layer-module-trace-actual-input-rotary64-layers7-9-hidden3994-mlp-gate-silu-up-layer6-layer10-p4p46-inproj.md`
- `qwen-module-trace-comparison-actual-input-rotary64-layers7-9-token7-hidden3994-mlp-gate-silu-up-layer6-layer10-p4p46-inproj.json`
- `qwen-module-trace-comparison-actual-input-rotary64-layers7-9-token7-hidden3994-mlp-gate-silu-up-layer6-layer10-p4p46-inproj.md`

Layer `8`, token `7`, hidden `3994`:

| item | value |
| --- | ---: |
| local delta error | 0.171178818 |
| attention row-only error | -0.053268506 |
| attention activation-path error | 0.125136728 |
| MLP row-only error | -0.017845260 |
| MLP activation-path error | 0.070061073 |

Attention feature `845`:

| stage | fullref | package | diff |
| --- | ---: | ---: | ---: |
| attention input normed | -0.421875000 | -0.422090501 | -0.000215501 |
| attention recurrent | 0.003967285 | 0.004505743 | 0.000538458 |
| attention pre-gate normed | 1.237266779 | 1.381341815 | 0.144075036 |
| attention gate SiLU | 2.327531576 | 2.394657612 | 0.067126036 |
| attention projection input | 2.875000000 | 3.307840586 | 0.432840586 |

MLP feature `6340`:

| stage | fullref | package | diff |
| --- | ---: | ---: | ---: |
| MLP gate projection | 4.312500000 | 4.261680126 | -0.050819874 |
| MLP gate SiLU | 4.255476952 | 4.202431202 | -0.053045750 |
| MLP up projection | -0.480468750 | -0.422994196 | 0.057474554 |
| MLP activation | -2.046875000 | -1.777603984 | 0.269271016 |

MLP feature `6340` product split, using the captured gate-SiLU and up values:

| term | value |
| --- | ---: |
| `(pkg_gate_silu - ref_gate_silu) * ref_up` | 0.025486825 |
| `ref_gate_silu * (pkg_up - ref_up)` | 0.244581638 |
| interaction term | -0.003048781 |
| product-space sum | 0.267019682 |
| captured activation diff | 0.269271016 |

The small residual between the product-space sum and captured activation diff is
from the full-reference BF16 activation value.

Interpretation:

- Layer `8` token `7` is still the largest local package-error point in the
  rotary64 hidden `3994` chain.
- The attention side is not a final-row scale case. A small recurrent feature
  difference is magnified by head-wise RMSNorm and a positive gate.
- The MLP side is also activation-path dominated. For feature `6340`, the main
  term is the `up_proj` difference multiplied by a large positive gate-SiLU
  value, not the gate difference itself.
- This supports a sensitivity/quantization analysis of internal projection rows
  feeding high-gain activation paths, rather than adding another final-row scale
  override.

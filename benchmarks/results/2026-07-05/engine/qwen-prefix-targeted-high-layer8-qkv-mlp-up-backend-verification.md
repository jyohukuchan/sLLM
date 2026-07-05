# Qwen Targeted Layer8 QKV + MLP Up Backend Verification

## Candidate

- baseline package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer10.ullm.d`
- candidate package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-layer8-qkv-mlp-up-high-row-scale-layer6-layer10.ullm.d`
- policy change:
  - `model.language_model.layers.8.linear_attn.in_proj_qkv.weight` high
  - `model.language_model.layers.8.mlp.up_proj.weight` high
  - existing layer6/layer10 row3456 manifest compensation preserved

## Gate Results

| backend | device | fixture set | decision | fixtures | mean improvement | median improvement | max regression |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| CPU | `0`, host CPU fallback | tokens1/tokens101/tokens201/tokens301/tokens401 | accept | 5 | `0.0479898453` | `0.022603035` | `0` |
| R9700 | `2`, AMD Radeon Graphics | tokens1/tokens101/tokens201/tokens301/tokens401 | accept | 5 | `0.0479856491` | `0.0226106644` | `0` |
| V620 | `1`, AMD Radeon Pro V620 | tokens1/tokens101/tokens201/tokens301/tokens401 | accept | 5 | `0.0479856491` | `0.0226106644` | `0` |

## CPU Five-Fixture Detail

| fixture | baseline | candidate | delta | layer | token | hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tokens1 | `0.645338058` | `0.629640579` | `-0.0156974792` | 11 | 7 | 3994 |
| tokens101 | `1.0805254` | `1.0805254` | `0` | 7 | 12 | 3994 |
| tokens201 | `1.140728` | `1.00050735` | `-0.140220642` | 11 | 13 | 3994 |
| tokens301 | `1.37130928` | `1.30988121` | `-0.0614280701` | 10 | 12 | 3994 |
| tokens401 | `0.959306717` | `0.936703682` | `-0.022603035` | 8 | 9 | 3994 |

## R9700 Five-Fixture Detail

| fixture | baseline | candidate | delta | layer | token | hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tokens1 | `0.645345688` | `0.629638672` | `-0.015707016` | 11 | 7 | 3994 |
| tokens101 | `1.08053589` | `1.08053589` | `0` | 7 | 12 | 3994 |
| tokens201 | `1.140728` | `1.00051689` | `-0.140211105` | 11 | 13 | 3994 |
| tokens301 | `1.37132263` | `1.30992317` | `-0.0613994598` | 10 | 12 | 3994 |
| tokens401 | `0.959306717` | `0.936696053` | `-0.0226106644` | 8 | 9 | 3994 |

## V620 Five-Fixture Detail

| fixture | baseline | candidate | delta | layer | token | hidden |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tokens1 | `0.645345688` | `0.629638672` | `-0.015707016` | 11 | 7 | 3994 |
| tokens101 | `1.08053589` | `1.08053589` | `0` | 7 | 12 | 3994 |
| tokens201 | `1.140728` | `1.00051689` | `-0.140211105` | 11 | 13 | 3994 |
| tokens301 | `1.37132263` | `1.30992317` | `-0.0613994598` | 10 | 12 | 3994 |
| tokens401 | `0.959306717` | `0.936696053` | `-0.0226106644` | 8 | 9 | 3994 |

## Artifact Index

- CPU summary: `qwen-prefix-targeted-high-layer8-qkv-mlp-up-five-fixture-summary.json`
- CPU gate: `qwen-prefix-targeted-high-layer8-qkv-mlp-up-five-fixture-gates.json`
- R9700 matrix: `qwen-prefix-smoke-matrix-targeted-high-layer8-qkv-mlp-up-r9700-five-fixture/`
- R9700 summary: `qwen-prefix-targeted-high-layer8-qkv-mlp-up-r9700-five-fixture-summary.json`
- R9700 gate: `qwen-prefix-targeted-high-layer8-qkv-mlp-up-r9700-five-fixture-gates.json`
- V620 matrix: `qwen-prefix-smoke-matrix-targeted-high-layer8-qkv-mlp-up-v620-representative-three-fixture/`
- V620 summary: `qwen-prefix-targeted-high-layer8-qkv-mlp-up-v620-representative-three-fixture-summary.json`
- V620 gate: `qwen-prefix-targeted-high-layer8-qkv-mlp-up-v620-representative-three-fixture-gates.json`
- V620 additional matrix: `qwen-prefix-smoke-matrix-targeted-high-layer8-qkv-mlp-up-v620-additional-two-fixture/`
- V620 five-fixture summary: `qwen-prefix-targeted-high-layer8-qkv-mlp-up-v620-five-fixture-summary.json`
- V620 five-fixture gate: `qwen-prefix-targeted-high-layer8-qkv-mlp-up-v620-five-fixture-gates.json`

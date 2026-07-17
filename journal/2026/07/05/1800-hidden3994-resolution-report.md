# 18:00 Hidden3994 Resolution Report

作成時刻: 2026-07-05 18:00 JST

## 前回の要点

- hidden `3994` のprefix driftは、単純なrow-scale補正やp4p65系policyでは5 fixture gateを通らなかった。
- layer8 `mlp.up_proj.weight` high-onlyはtokens1を改善したがtokens401を悪化させた。
- layer8 `linear_attn.in_proj_qkv.weight` high-onlyはtokens401を改善したがtokens1を悪化させた。

## 今回の変更点

- `ullm-quant` の既存 `--aq-high-tensor` overrideを使い、次の2 tensorを同時にhigh formatへ上げたpackageを評価した。
  - `model.language_model.layers.8.linear_attn.in_proj_qkv.weight`
  - `model.language_model.layers.8.mlp.up_proj.weight`
- row3456 manifest compensationは維持した。
- CPU、R9700、V620 2枚でfive-fixture gateを確認した。

## 結論

この問題は、今の状態では「さらに闇雲にデバッグを続ける問題」ではなくなった。

`targeted-high-layer8-qkv-mlp-up` packageは、固定した5 fixture gateをCPU/R9700/V620-A/V620-Bすべてで通った。次の仕事は、accepted candidateを一時的なdebug packageのままにするか、small named quantizer policyとして昇格するかを決めること。

## Accepted Candidate

- package:
  - `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-layer8-qkv-mlp-up-high-row-scale-layer6-layer10.ullm.d`
- recipe:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-prefix-targeted-high-layer8-qkv-mlp-up-accepted-package-recipe.md`
- backend verification:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-prefix-targeted-high-layer8-qkv-mlp-up-backend-verification.md`

## Gate Results

| backend | device | decision | fixtures | mean improvement | median improvement | max regression |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| CPU | `0` host CPU fallback | accept | 5 | `0.0479898453` | `0.022603035` | `0` |
| R9700 | `2` AMD Radeon Graphics | accept | 5 | `0.0479856491` | `0.0226106644` | `0` |
| V620-A | `1` AMD Radeon Pro V620 | accept | 5 | `0.0479856491` | `0.0226106644` | `0` |
| V620-B | `3` AMD Radeon Pro V620 | accept | 5 | `0.0479856491` | `0.0226106644` | `0` |

CPU five-fixture detail:

| fixture | baseline | candidate | delta |
| --- | ---: | ---: | ---: |
| `tokens1` | `0.645338058` | `0.629640579` | `-0.0156974792` |
| `tokens101` | `1.0805254` | `1.0805254` | `0` |
| `tokens201` | `1.140728` | `1.00050735` | `-0.140220642` |
| `tokens301` | `1.37130928` | `1.30988121` | `-0.0614280701` |
| `tokens401` | `0.959306717` | `0.936703682` | `-0.022603035` |

## Saved Commits

- `a3b0f9d` Accept Qwen layer8 qkv MLP up high package
- `56247ed` Extend Qwen accepted package check to V620 five fixture
- `9d87f26` Verify Qwen accepted package on second V620

Earlier supporting commits in this branch:

- `7f1b676` Add Qwen targeted high tensor policy override
- `686d5ab` Evaluate Qwen targeted layer8 MLP up high package
- `ac8edd4` Evaluate Qwen targeted layer8 qkv high package

## Remaining Work

1. Decide whether to keep the solution as explicit `--aq-high-tensor` overrides or introduce a named policy preset for this Qwen hidden3994 fix.
2. If it becomes a named preset, add a small test proving that exactly the intended layer8 qkv and MLP-up tensors are promoted.
3. After policy naming, regenerate the accepted package from the recipe and rerun the same gate once as a final promotion check.

## Stop State

- Work stopped after 18:00 JST as requested.
- No further debug run was started after the report window.

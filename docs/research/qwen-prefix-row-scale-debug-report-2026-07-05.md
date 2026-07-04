# Qwen Prefix Row-Scale Debug Report 2026-07-05

## 前回の要点

- Qwen3.5-9B p4p46 in-projection packageのprefix smokeでは、hidden `3994` の差分が複数fixtureで支配的になっていた。
- Layer6 `mlp.down_proj.weight[3994]` row-scaleは、layer6/layer7の局所差分を強く下げる有望候補だった。
- ただし、単一fixtureの局所row-dot RMSE改善だけでは、後段layerの最終max absが悪化する例が出ていた。

## 今回の変更点

- 複数fixtureのprefix smoke結果を集約する `tools/summarize-qwen-prefix-smokes.py` を追加した。
- `qwen-layer-module-trace` の `row_dot.<projection>.scale_fit` からrow-scale候補を抽出する `tools/extract-qwen-row-scale-candidates.py` を追加した。
- 候補ごとの受入ゲートを機械判定する `tools/evaluate-qwen-prefix-candidate-gates.py` を追加した。
- OOMを避けるため、fixture/condition matrixを1 smokeずつ逐次実行する `tools/run-qwen-prefix-smoke-matrix.py` を追加した。
- `docs/plans/multi-fixture-row-scale-validation-plan-v0.1.md` に進捗と判定結果を追記した。

## 現在の判断

この問題は引き続きデバッグする価値がある。理由は、残差がbackendノイズではなく、特定のlayer/token/hiddenとrow-scale候補に局所化できているため。

ただし、今の扱いは「row-scale候補をすぐ昇格する」ではない。複数fixtureのend-to-end smokeで悪化しない候補だけを受け入れる。

## Gate Result

Input summary:

- `benchmarks/results/2026-07-05/engine/qwen-prefix-smoke-multi-fixture-summary.json`
- `benchmarks/results/2026-07-05/engine/qwen-prefix-candidate-gates.md`

Gate settings:

- fixtureごとの最終max abs悪化が `0.001` を超えたらreject
- median improvementは `0.005` 以上を優先
- paired fixtureは最低 `3`

| condition | decision | fixtures | median improvement | max regression | reason |
| --- | --- | ---: | ---: | ---: | --- |
| `layer6` | reject | 3 | 0.0081653595 | 0.00455665588 | tokens201 regression exceeds `0.001` |
| `combined` | needs_more_fixtures | 1 | 0.0343608856 | 0 | only one paired fixture |

Interpretation:

- Layer6 hidden3994 row-scale is a real local compensation candidate.
- It should not be promoted unconditionally under the initial multi-fixture gate.
- Layer8 QKV V845 cell remains smoke-only because current paired fixture coverage is insufficient.

## Key Evidence

| fixture | baseline | layer6 h3994 row-scale | result |
| --- | ---: | ---: | --- |
| token ids `1..16` | 0.645338058 | 0.637172699 | improves |
| token ids `101..116` | 1.080525398 | 1.043153763 | improves |
| token ids `201..216` | 1.140727997 | 1.145284653 | worsens final max |

Layer6 row-scale still improves early layers on tokens201:

- layer6: `0.537414551 -> 0.476898193`
- layer7: `0.966460228 -> 0.497438431`
- layer11 final max: `1.140727997 -> 1.145284653`

Layer6 `mlp.down_proj.weight[3994]` row-dot scale is stable across the two available v0.10 traces:

- token ids `1..16`: scale `1.026471714`, RMSE `0.117735388 -> 0.063680278`
- token ids `101..116`: scale `1.023383096`, RMSE `0.131756300 -> 0.061972585`

This means the local correction is real, but it changes later propagation in a way that the current policy cannot accept.

## Verification

- `python3 -m py_compile tools/summarize-qwen-prefix-smokes.py`
- `python3 -m py_compile tools/extract-qwen-row-scale-candidates.py`
- `python3 -m py_compile tools/evaluate-qwen-prefix-candidate-gates.py`
- `python3 -m py_compile tools/run-qwen-prefix-smoke-matrix.py`
- JSON parse checks for generated summary/candidate/gate artifacts.
- `tools/run-qwen-prefix-smoke-matrix.py` dry-run with tokens1/tokens101 and baseline/layer6 conditions.
- Regenerated tokens1 layer6 hidden3994 trace with `qwen-layer-module-trace-v0.10`.

## Next Action

1. Use `tools/run-qwen-prefix-smoke-matrix.py` for a three-fixture candidate matrix rather than running ad hoc smokes.
2. Regenerate missing v0.10 traces for token ids `1..16` and additional tokens201 layer11 candidates.
3. Search paired candidates that reduce tokens201 layer11 without increasing tokens1/tokens101 final max.
4. Keep layer6 hidden3994 and layer8 QKV V845 as candidates, not promoted package policy, until the gate passes.

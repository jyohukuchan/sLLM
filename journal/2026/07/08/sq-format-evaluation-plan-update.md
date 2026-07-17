# SQ format evaluation plan update

## 前回の要点

- R9700/RDNA4のcached-prefix/cold-prefill componentは、SQ候補評価を始める前提速度として一旦十分と判断した。
- T1はlogical full-package gridとselected-layer hybrid model-loop smokeまで進んだ。
- T2は `kup6_gate5_down5` の6層strict-top1 regression subsetと、実FP8 payload artifactのmaterialize smokeまで進んだ。

## 今回の変更点

- `docs/plans/fp8-sq-r9700-batch-throughput-prefill-plan-v0.1.md` を更新し、主線を追加attention kernel開発からSQ format design/evaluationへ移した。
- full-package real batch runnerは最終比較に必要だが、SQ候補探索の開始blockerにはしない方針にした。
- SQ候補軸として、W8A16 F32 scale、scale16、scale8、W8A8、hybrid fallbackを明示した。
- 品質guardは当面strict top1を正式条件にし、top-k overlap、AQ4 top1 rank、logit gapは診断扱いに固定した。
- overlay host materialize/load timingをSQ速度として読まないことを改めて明記した。

## 次の行動

1. `sq-fp8-kup6-gate5-down5-policy-v0.1.json` を基準に、候補matrixを機械可読manifestへ落とす。
2. `kup6_gate5_down5` から広げる方向と、scale/layoutを強める方向を分けて品質探索を行う。
3. selected-layer stackへtoken-id embedding入力、final norm/lm_head、quality guardを接続する。
4. T1aとしてfull-package real batch runnerを継続し、AQ4/FP8の `batch=1/4/8` 比較行を作る。

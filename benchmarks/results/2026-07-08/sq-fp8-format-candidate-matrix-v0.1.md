# SQ FP8 Format Candidate Matrix v0.1

## 前回の要点

- `kup6_gate5_down5` は6層strict-top1 regression subsetであり、full SQ policyではない。
- 現在の実FP8 payload artifactはmaterialize smokeまで確認済みだが、throughput結果ではない。

## 今回の変更点

- SQ format候補を機械可読matrixとして固定した。
- full-package real batch runnerは最終比較には必要だが、候補探索の開始blockerにはしない。
- overlay host materialize/load timingを速度結果として使わない方針をmatrixにも入れた。

## Candidate Matrix

| candidate | status | scale dtype | activation | compact resident GiB | next action |
| --- | --- | --- | --- | ---: | --- |
| sq-fp8-w8a16-r9700-v0 | current_regression_subset_not_full_policy | f32 | bf16_or_f32 | 17.304 | broaden quality coverage or connect this policy to selected-layer throughput guard |
| sq-fp8-w8a16-r9700-v1-scale16 | planned_experiment | fp16_or_bf16 | bf16_or_f32 | 17.255 | add scale dtype option to artifact builder and rerun strict top1 prompt bundle |
| sq-fp8-w8a16-r9700-v1-scale8 | planned_risk_probe | fp8_e4m3_or_e5m2 | bf16_or_f32 | 17.231 | only run after scale16 clarifies quality and runtime overhead |
| sq-fp8-w8a8-r9700-v0 | planned_after_w8a16_baseline | f32 | fp8_e4m3 | 17.304 | defer until W8A16 quality guard and selected-layer throughput path are stable |
| sq-fp8-hybrid-r9700-v0 | planned_conservative_policy_family | f32 | bf16_or_f32 | 17.304 | use as fallback direction if broader W8A16 strict top1 fails |

## 次の行動

1. このmatrixから候補artifactを生成し、strict top1 prompt bundleを通す。
2. selected-layer stackへtoken-id embedding、final norm/lm_head、quality guardを接続する。
3. T1aのfull-package real batch runnerを継続し、最終AQ4/SQ比較行を作る。

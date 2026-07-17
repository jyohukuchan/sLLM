# T2 SQ FP8 candidate matrix

## 前回の要点

- `kup6_gate5_down5` は6層strict-top1 regression subsetであり、full SQ policyではない。
- full-package real batch runnerは最終比較には必要だが、SQ候補探索の開始blockerにはしない方針に更新した。

## 今回の変更点

- `tools/build-sq-fp8-candidate-matrix.py` を追加した。
- `sq-fp8-kup6-gate5-down5-policy-v0.1.json` とpolicy artifact resultから、SQ FP8候補matrixを再生成できるようにした。
- 生成物:
  - `benchmarks/results/2026-07-08/sq-fp8-format-candidate-matrix-v0.1.json`
  - `benchmarks/results/2026-07-08/sq-fp8-format-candidate-matrix-v0.1.md`
- 候補軸:
  - `sq-fp8-w8a16-r9700-v0`
  - `sq-fp8-w8a16-r9700-v1-scale16`
  - `sq-fp8-w8a16-r9700-v1-scale8`
  - `sq-fp8-w8a8-r9700-v0`
  - `sq-fp8-hybrid-r9700-v0`
- `sq-r9700-state-freeze-v0.1.{json,md}` と計画書へcandidate matrixの状態を反映した。

## 次の行動

1. selected-layer stackへtoken-id embedding、final norm/lm_head、quality guardを接続する。
2. `scale16` を試す場合は、artifact builderへscale dtype optionを追加し、strict top1 prompt bundleを再実行する。
3. T1aのfull-package real batch runnerを継続する。

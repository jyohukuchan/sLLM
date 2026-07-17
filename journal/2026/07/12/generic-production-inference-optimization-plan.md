# 汎用production推論最適化計画の作成

## 前回の要点

- AQ4 residentのprefillはtokenwiseで、既存batch componentの性能が製品推論へ届いていなかった。
- 長context decodeは8 self-attention層のKV全走査により低下していた。

## 今回の変更点

- `uLLM-project/docs/plans/generic-production-inference-optimization-plan-v0.1.md`を作成した。
- 計画書をcommit `8d040ce`（`docs: plan generic production inference optimization`）として保存した。
- model adapter、model graph、state schema、batch planner、backend registry、generic executor、production execution traceへ責務を分けた。
- AQ4/Qwen3.5を最初の縦切りにしつつ、SQ8/Qwen3と追加モデルadapterでも同じ実行基盤を使うことを完成条件にした。
- component benchmarkだけではpromotionできず、resident worker/OpenWebUIでexecutor IDと実batch幅を証明するgateを追加した。
- 実装、build、service起動、GPU測定は行っていない。

## 次の行動

1. P0でgraph/state/registry/evidenceのADR/specを固定する。
2. P1でtyped graphとCPU reference executorを作る。
3. AQ4 kernel変更はP2以降のregistryとP3の共通batched operator境界から進める。

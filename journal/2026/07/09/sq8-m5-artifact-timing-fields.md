# SQ8_0 M5 Artifact Timing Fields

## 前回の要点

- M5では、SQ8_0のmodel-loop / batch比較行にartifact loadとmaterializationのwall timeを残す必要があった。
- 既存parserは`layer_load_ms`を`layer_load_wall_time_seconds`として保存していたが、SQ8_0比較用の意味がフィールド名から読み取りにくかった。
- M10のvLLM + FP8比較は、SQ8_0 full-package rowが実装有効になった後半で実行する位置づけとして計画済み。

## 今回の変更点

- `tools/run-external-benchmark.py`でmodel-loop metricsに次の比較用フィールドを追加した。
  - `artifact_load_wall_time_seconds`
  - `artifact_materialization_wall_time_seconds`
  - `load_excluded_total_wall_time_seconds`
  - `load_included_total_wall_time_seconds`
- `artifact_load_wall_time_seconds`は明示的な`artifact_load_ms`があれば優先し、無ければ既存の`layer_load_ms`を使う。
- `artifact_materialization_wall_time_seconds`は明示的なmaterialization系timerがある場合だけ保存し、`layer_load_ms`とは混同しない。
- `docs/specs/inference-benchmark-result-v0.1.md`と`docs/plans/sq8-implementation-plan-v0.1.md`に保存フィールド名を明記した。

## 次の行動

- parser testとpy_compileを通したうえで、M5の残りであるfull-package SQ8_0 rowの取得とprompt-suite regression status保存へ進む。

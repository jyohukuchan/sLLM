# SQ8_0 M5 Inference Row Guard Bundle Attachment

## 前回の要点

- SQ candidate runtime rowには、prompt-suite regression statusとguard metric summaryを保存できるようになった。
- ただし`inference-benchmark-result-v0.1` row側には、throughput rowへprompt-suite結果を添付する入口がなかった。
- M10のvLLM + FP8比較では、uLLM側rowの実装状態と回帰診断が同じ比較単位で見える必要がある。

## 今回の変更点

- `tools/run-external-benchmark.py`に`--prompt-guard-bundle-json`を追加した。
- 指定されたguard bundleをベンチ実行前に読み、rowへ次を添付するようにした。
  - `quality.prompt_suite_regression_status`
  - `guards.prompt_guard_bundle.*`
  - `artifacts.prompt_guard_bundle_json`
- `docs/specs/inference-benchmark-result-v0.1.md`と`docs/plans/sq8-implementation-plan-v0.1.md`へ任意添付フィールドを追記した。

## 次の行動

- 既存または新規のfull mixed SQ8_0 throughput rowへguard bundleを添付し、M5の比較可能rowとして保存する。

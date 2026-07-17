# SQ8_0 M5 Prompt-Suite Regression Status

## 前回の要点

- M5では、SQ8_0比較行にfinal logits guardとbehavioral prompt-suiteの結果を回帰診断として残す必要があった。
- prompt-suiteの実行と比較は、既に`run-package-token-prompt-suite.py`、`compare-package-token-prompt-suite.py`、`run-package-prompt-guard-bundle.py`で行える。
- ただしcandidate runtime rowには、guard bundle全体の`passed`とartifact pathだけが保存され、behavioral/strictの内訳が見えにくかった。

## 今回の変更点

- `tools/build-sq-candidate-runtime-row.py`で、guard bundleの`prompt_suite_token_logits` checkを読み取るようにした。
- `quality.prompt_suite_regression_status`を追加し、prompt-suite checkの状態を`passed` / `failed` / `not_attached`で保存する。
- `guards.prompt_guard_bundle`に、acceptance mode、strict/behavioral pass、比較case数、generated token/text match数、top-logit差分を保存する。
- `docs/specs/sq-candidate-runtime-result-v0.1.md`と`docs/plans/sq8-implementation-plan-v0.1.md`に保存フィールドを追記した。

## 次の行動

- M5の残りは、実際のfull-package SQ8_0 rowを取得して、artifact timing、projection telemetry、prompt-suite statusが同じ比較単位で揃うことを確認する。

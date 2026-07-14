# P2 resident smoke execute launcher

## 前回の要点

strict live-preflight runner、validator、B sidecarは固定済みだったが、execute launcherのsafety evidenceと全段TOCTOU再検証が独立QAを満たしていなかった。

## 今回の変更点

- runner未開始`false`、開始後不明`unknown`、到達証明済み`true`としてGPU command/model load safetyを状態遷移から記録した。
- fake runnerの成功、起動失敗、途中失敗、sudo keepalive失敗とfinally復旧を回帰testにした。
- execute開始前にinput/B/Python/validator/R/resident/served/self/ROCm toolsをsnapshotし、validator前、runner前、runner後、finalize前にFD/path identity/SHAを再検証した。
- 4地点のTOCTOU swapを全て拒否した。
- launcherを`bb7a5fb`、SHA-256 `4f547d50b4a321196dbb8b2e7703843657c87a4d3d215c0325a6f8d267db5382`としてcanonical artifactの`launcher-trust.json`へ固定した。
- canonical bindingは`actual_eligible=false`、live-preflight未生成、QA待ちのまま保存した。
- actual GPU command、model load、service操作は実行していない。

## 次の行動

独立QA結果を待つ。QAが通っても、別の明示的なactual承認と全live gate通過まではblocked状態を維持する。

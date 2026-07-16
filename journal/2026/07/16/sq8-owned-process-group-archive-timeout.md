# SQ8 owned process-group and archive timeout hardening

## 前回の要点

外側captureはdirect child PIDだけをTERM/KILLしていたため、stdout/stderrを継承した子孫が残った。
source archive SHA計算はstdout EOFを先に待つため、停止した`git archive`に30秒deadlineが届かなかった。

## 今回の変更点

- captureを`start_new_session=True`で起動し、PGIDが親PIDと一致する所有確認後にgroup全体へ
  bounded TERM→KILLを行う。direct childをwaitし、group消失も有界に確認する。
- 親だけが成功終了してpipe継承子が残る場合もtyped cleanup failureとしてgroupを回収する。
- PID/PGID不一致では`killpg`を呼ばず、`process_group_identity_invalid`でfail closedにする。
- runnerとGate builderのsource archive SHA計算を30秒deadline下の並行stdout/stderr処理へ変更した。
  stdoutはstreaming SHA、stderrはbounded diagnosticとし、timeout時は所有groupを回収する。
- TERM無視、pipe継承、親子残留、成功親のorphan、stalled archive、2 MB stderrの実process故障試験を追加した。
- GPU、service、authorization、actual executionは実行していない。

## 検証

jobs=1で主要5 suiteは`257 passed`、関連9 suiteは`328 passed, 1 deselected`だった。
除外1件は既知の隔離worktree絶対path fixtureである。`compileall`と`git diff --check`も通した。

## 次の行動

commit/tree/archive identityを固定し、独立監査でprocess group残留とarchive deadlineを再確認する。

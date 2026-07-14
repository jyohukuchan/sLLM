# P2 resident one-case operator command v6

## 前回の要点

- v5はKFD canonical parser対応時点のbase ready artifactとmaintenance harnessを固定していた。
- その後、resident driver failure captureの強化とv2 execute outputへの更新が完了したため、旧manifestは再利用しない。

## 今回の変更点

- base ready artifactをcommit `b5e85b60`、maintenance harnessをcommit `7e597eb6`へ固定した。
- canonical argvを`resident-one-case-smoke-operator-command-v6/command-manifest.json`へ9要素の絶対path配列、`shell=false`、最大1回として固定した。
- maintenance evidenceは新規`resident-one-case-smoke-maintenance-evidence-v6`を指定した。ready artifactが固定するlauncher evidence v2とrunner output v2を含む3出力は監査時点で存在しない。
- Python、harness、ready artifact、argv、出力pathをSHA-256で固定した。manifestとjournalはsecret-freeで、actual、sudo、service、GPU、HTTP probeを実行していない。

## 次の行動

- 別の明示指示がある場合だけ、manifestのcwdとargvを変更せず、同一PTY、`shell=false`で最大1回実行する。
- 実行直前に全input hash、manifestのSHA256SUMS、権限、3出力の不存在を再確認する。いずれかが異なる場合は実行しない。

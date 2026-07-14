# P2 resident one-case operator command v5

## 前回の要点

- v4 actualはKFD `gpuid`実rawが改行なしだったため旧parserで停止し、launcherを開始せずserviceを復元した。maintenance evidence `v4`は再利用しない。
- KFD gpuid canonical parser連鎖の独立QAは完了している。最新base ready artifactはcommit `e5d30b47`、maintenance harnessはcommit `74ac7559`である。

## 今回の変更点

- canonical argvを`resident-one-case-smoke-operator-command-v5/command-manifest.json`へ9要素の絶対path配列、`shell=false`、最大1回として固定した。
- maintenance evidenceは未使用の`resident-one-case-smoke-maintenance-evidence-v5`へ更新した。ready artifactが固定するlauncher evidence `resident-one-case-smoke-execute-evidence-v1`とrunner output `resident-one-case-smoke-execute-v1`は一度も作成されておらず、監査時点でも存在しない。implicit 2出力はready artifact SHA-256とfield名から導出してhash-boundした。
- 相対ready artifactはdependency call 0で拒否された。絶対ready artifactはfake dependencyだけでdurable markerまで進み、service stop直前に停止してsystemctl stop、actual process、launcherを0回に保った。canonical dry-runの全process countも0だった。
- Python、harness、ready artifactのSHA-256とGit identityを固定した。manifestとjournalにはsecretを保存しておらず、actual、sudo、service、GPU、HTTP probeは実行していない。

## 次の行動

- 別の明示指示がある場合だけ、manifestのcwdとargvを変更せず、同一PTY、`shell=false`で最大1回実行する。
- 実行直前に全input hash、manifestのSHA256SUMS、権限、3出力の不存在を再確認する。いずれかが異なる場合は実行しない。

# P2 resident one-case operator command

## 前回の要点

- base ready artifact `da5dd372`とmaintenance harness `76feccbe`は、actual one-caseを最大1回だけ許可している。
- actual実行、sudo、service停止、GPU、HTTP probeはまだ行っていない。

## 今回の変更点

- canonical actual argvを`resident-one-case-smoke-operator-command-v1/command-manifest.json`へ絶対pathの配列として固定した。
- Python、harness、ready artifactのSHA-256とGit identity、3つのfresh output、`--confirm-one-case`のexact 1回を固定した。
- 相対`--ready-artifact`はdependency call 0で拒否され、絶対pathはfake pre-stop snapshotとdurable markerまで進んだ後、systemctl stop 0回で停止することを確認した。
- canonical dry-runは全process count 0だった。manifestとjournalにはsecretを保存していない。

## 次の行動

- actual実行を担当するLunaは、別の明示指示を受けた場合にだけmanifestの`argv`を配列のまま、`shell=false`で最大1回実行する。
- 実行直前にmanifestの全input hashと3つのfresh outputを再確認する。journalの文章からcommandを再構成しない。

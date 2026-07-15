# P2 resident profile diagnostic operator command

## 前回の要点

- base one-case smoke v9は成功し、外側maintenance restoreまで完了した。
- profile-ready chainはcommit `8b2fa37d`、maintenance harnessは`3e24fb66`、capture toolは`b4d515f`へ固定済みである。

## 今回の変更点

- `resident-one-case-smoke-profile-operator-command-v1/command-manifest.json`へprofile diagnosticの10要素exact argv、repo cwd、`shell=false`、最大1回を固定した。
- maintenance evidenceはfresh `resident-one-case-smoke-profile-maintenance-evidence-v1`を指定した。profile launcher evidence v1、runner output v1、P3 capture directory/artifact、rocprof stdout/stderrを含む全7出力は監査時点で存在しない。
- profile-ready `SHA256SUMS`の全member、raw target manifest SHAとself-hash、capture/profiler/launcher/ROCTx SHA、12 balanced range、capture capability、`outer harness -> capture -> rocprof -> launcher -> runner`を再確認した。
- canonical profile dry-runは全actual process count 0である。profile fake/dry testsは16件、capture/ROCTx testsは16件通過した。
- actual、sudo、service、GPU、model load、rocprof captureは実行していない。profile readiness判定は`PASS`である。

## 次の行動

- 別の明示承認がある場合だけ、manifestのcwdとargvを変更せず、同一PTY、`shell=false`で最大1回実行する。
- 実行直前に全input hash、manifestの`SHA256SUMS`、権限、全7出力の不存在を再確認する。いずれかが異なる場合は実行しない。

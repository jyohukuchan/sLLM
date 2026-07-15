# P2 resident profile diagnostic operator command v2

## 前回の要点

- v1はprofile-ready `8b2fa37d`、harness `3e24fb66`、capture `b4d515f`、fresh profile/capture outputsを固定した。
- quiet-window evidenceはcommit `430f0889`、SHA-256 `a2b09caf...`、`status=go`で確定した。

## 今回の変更点

- v1 commit `5e7ecce8`を前提に、quiet-windowのservice epoch、主・terminal各25 samples、formal health一致、terminal process観測0をv2 manifestへ固定した。
- exact argvのmaintenance evidenceだけをfresh `resident-one-case-smoke-profile-maintenance-evidence-v2`へ更新した。launcher evidence、runner output、capture directory/artifact、rocprof stdout/stderrを含む全7出力は不存在である。
- cwdはrepo root、`shell=false`、profile flagとconfirm flagは各1、最大1回、output再利用禁止、secret-freeである。
- profile fake/dry testsは16件通過した。actual、sudo、service、GPU、model load、rocprof captureは実行していない。

## 次の行動

- 別の明示承認がある場合だけ、v2 manifestのcwdとargvを変更せず、同一PTY、`shell=false`で最大1回実行する。
- 実行直前にquiet-window GO証拠と全input SHA、manifest `SHA256SUMS`、全7出力の不存在を再確認する。差異があれば実行しない。

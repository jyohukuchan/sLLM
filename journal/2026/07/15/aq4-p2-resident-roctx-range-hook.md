# AQ4 P2 resident ROCTx range hook

## 前回の要点

- diagnostic capture toolはexact 12 ROCTx markerを要求する。
- 現行runnerにはmarker hookがなく、live integrationはfail-closedで停止していた。

## 今回の変更点

- runnerへ明示的`--profile-roctx-ranges`を追加した。
- libroctx64 invocationのancestor/leaf symlink chain、resolved実体、inode、SHA、push/pop symbolを固定した。
- run send直前からvalidated complete直後まで同一PID/threadでrangeを張る。
- exception、OOM、timeoutでもfinally popし、missing/unbalanced/12件未満を拒否する。
- exact marker nameと12 range audit sidecarを追加した。
- flagなしの通常runはROCTx load/marker validationを行わない。

## 検証

- fake ROCTx専用: 5 passed
- runner + live-preflight + ROCTx: 55 passed
- GPU、service、model loadは実行していない。

## 残課題

- prepared runner、validator、B、launcher、maintenance harnessのtrust hashは旧runnerを指している。
- trust chain再pinまでactual one-case profileを実行できない。

## 次の行動

- runner sourceをprepared bundleへ反映し、全canonical artifactを新commit/hashへ再pinする。

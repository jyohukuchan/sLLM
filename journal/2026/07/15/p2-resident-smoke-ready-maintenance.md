# P2 resident smoke ready maintenance

## 前回の要点

execute launcher再QAは120 testsでPASSしたが、artifactはblockedで、サービス停止・復帰を外側で保証していなかった。

## 今回の変更点

- maintenance harness `fabd520`を新設した。
- sudo prevalidation、停止前service/health/hash/PID/NRestarts/GPU/lock snapshot、durable marker、停止後zero-owner gate、bb launcher、外側finally復旧を固定した。
- 復旧時は新gateway/worker PID、NRestarts不変、manifest/worker/package不変、lock/GPU owner、gateway/OpenWebUIを再検査する。
- fake sudo/systemctl/launcherで成功、launcher起動失敗、途中失敗、各gate失敗、restore失敗、output reuseを検査し、23 testsが通過した。
- ready artifactは最大1回、promotion不可、output no-reuseである。live-preflight SHAは停止後生成のため事前nullとし、最終evidenceでpath/SHAを固定する。
- canonical dry-runは全actual process count 0で通過した。
- actual service停止、GPU command、model loadは実行していない。

## 次の行動

ready artifactとharnessの独立QAを依頼する。明示的なactual承認があるまでは実行しない。

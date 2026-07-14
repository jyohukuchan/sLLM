# P2 active fidelity gate attempt 2

## 前回の要点

attempt 1は、`systemctl` の対話認証不足でservice停止・復旧が失敗した。serviceは復旧後もactive/runningで、GPU captureへは到達していない。

## 今回の変更点

`sudo -n systemctl` 固定配列を含むgate（SHA256 prefix `5bcd9d74`）でattempt 2を開始した。preflightとservice停止は通過したが、停止後にhomelab1が `/run/ullm` を `mkdir` できず `Permission denied` となり、gateはexit 90で停止した。GPU capture開始前の失敗であり、outputは生成されなかった。

cleanupはsudo cacheで復旧を完了した。post状態はservice active/running、MainPID `728405`、`NRestarts=0`で、active/package/workerおよびruntime/lock identityは不変だった。

## 次の行動

停止後のRuntimeDirectory再作成を固定配列 `RUNTIME_DIR_INSTALL=(sudo -n -- install -d -o homelab1 -g homelab1 -m 0750)` へ切り替え、lockはhomelab1が作成する。修正中はGPU/serviceを再実行しない。実行者は同一PTYで事前に `sudo -v` を行い、パスワードを記録しない。

# P2 resident container namespace health ready

## 前回の要点

production gatewayはOpenWebUI container内からhealthyだったが、maintenance harnessの正式health gateはhost名前空間からcontainer network gatewayへ接続していたため、経路不一致でNO-GOになっていた。

## 今回の変更点

- 正式health gateを固定済み`/usr/bin/docker`から固定済みOpenWebUI container IDへの`docker exec`へ変更し、container内の固定済みcurlでgateway `/healthz`、`/readyz`、認証済み`/v1/models`、OpenWebUI `/health`を検査するようにした。
- Dockerのpath/SHA/identity/client version、container ID/image/name/running/health、network ID/IP/gateway、curl path/version/SHA、API key file identityを事前・復旧後で固定する。認証headerは`docker exec -i ... curl --header @-`のstdinだけで渡し、tokenをargv、log、evidenceへ保存しない。
- host直結HTTPはtimeoutを許容する診断情報へ降格した。container不在・差し替え、image/health/network変更、curl失敗、model不一致、secret echo、Docker/API key同一path差し替え、復旧後health失敗を偽装試験で閉じた。
- harnessを`3918c6e3`へ固定し、base/profile readyと両canonical dry-runを再生成した。主要180、marker 55、diagnostic capture 11、capture関連85 testsが通過した。dry-runではdocker/docker exec/container curlを含む全process countが0である。
- actual HTTP、service停止・起動、GPU command、model load、rocprof captureは実行していない。

## 次の行動

別途明示されたactual一回実行時だけ、fresh output、同一PTY sudo cache、全pre-stop/live gateを満たした状態でこのcontainer名前空間health経路を使用する。いずれかの固定値が異なる場合は実行しない。

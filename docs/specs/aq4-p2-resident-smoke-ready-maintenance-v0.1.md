# AQ4 P2 resident smoke ready maintenance v0.1

## 前回の要点

旧ready chainはexecute launcher `bb7a5fb`、strict runner `774f6dd`、maintenance harness `fabd520`を固定していた。ROCTx marker対応runner `e93a2c1`とprofile diagnostic captureを使うには、runnerからready artifactまでの再固定が必要だった。

## 今回の変更点

maintenance harnessをcommit `7f1ad5c1785c50d99e0952afa7ce018a0204d29c`、tree `0740198f8b02c5171856b0044613f2b1654f888e`、Git blob `5b4f54f120fcd25429d7d2c8050a33080bb00c0c`、SHA-256 `7a0a3d48453eec49821c78e9a99f11cfc9973112b63cce29dee4c80c1bbe0468`へ再固定した。launcherはcommit `eec6922fa9c90267213d2749c5dc816be54de527`、tree `f6cef14d1e2a75dc1a12371d2a8e2a754d506482`、Git blob `c422e4235a2ee6595cf43656c573b7e863489f9e`、SHA-256 `607b7c9ad0bf7aa8e8b9303f60209b4a6dc998886dbd8af86d83955984232835`である。

harnessは同一PTYで事前prime済みの`sudo -n -v`だけを受け入れる。パスワード、API key、HTTP authorization、prompt/response本文をargv、environment、stdout/stderr evidenceへ保存しない。

実行順序を次に固定する。

1. sudo cache、active/running service、gateway PID、worker PID、NRestarts、ControlGroupを検査する。
2. served manifest、worker binary、package manifest、1045-file package treeをchunked SHA-256で照合する。
3. AMD SMI index 2、BDF `0000:47:00.0`、UUID `a8ff7551-0000-1000-80e9-ddefa2d60f55`、KFD ID `51545`、node 2、単一worker owner、device lock保持を確認する。正式なHTTP gateは固定済み`/usr/bin/docker`から、固定済みOpenWebUI container IDへ`docker exec`し、コンテナ名前空間内の固定済みcurlでgateway `/healthz`、`/readyz`、認証済み`/v1/models`、OpenWebUI `/health`を検査する。host直結HTTPは到達性診断だけでありgateにしない。
4. 復旧必須のdurable markerをatomic no-replace、mode 0444、fsync済みで保存する。
5. `sudo -n systemctl stop ullm-openai.service`を実行し、開始時に一度だけ固定したmonotonic absolute deadline 30秒まで、0.25秒から1秒までのbackoffで停止資源をpollする。各観測と各probeの前後で期限を確認し、subprocess timeoutは2秒と残時間の小さい方へ制限する。sudo keepaliveもprobe間で期限を再確認し、同じ2秒・残時間上限を使う。対象service inactive、worker不在、AMD/KFD owner 0、lock free、VRAM freeを期限内に連続2回観測してからだけlauncherへ進む。2回目のstable観測または単一probeが期限を越えた場合もtimeoutとしてfail-closedにする。pre-stop worker PIDのAMD/KFD解放とpre-stop service MainPIDのlock解放だけを待機対象とし、unknown/foreign/new PIDやzero後の再出現は即時fail-closedにする。各観測はsource別raw SHA、parsed PID、VRAM、raw非保存のproc cmdline SHA、期限checkpoint、timeoutしたprobeの部分証拠、分類を独立したmode 0444 JSONへatomic保存する。poll中は10秒間隔でsudo cacheを維持する。
6. base modeでは`eec6922` launcherへone-case bindingを渡す。launcher自身が30秒間隔でsudo keepaliveを行う。
7. 外側`finally`はlauncher開始前・途中・完了の成否にかかわらず`sudo -n systemctl start`を試行する。
8. active/running、新gateway PID、新worker PID、NRestarts不変、ControlGroup不変、lock再保持、manifest/worker/package hash不変、単一GPU owner、固定済みcontainer/image/network/curlとgateway/OpenWebUI復帰を最大120秒で検査する。API keyは固定済みfile identityから読み、Authorization headerだけを`docker exec -i ... curl --header @-`のstdinへ渡し、argv、log、evidenceへ保存しない。

base ready artifactは`resident-one-case-smoke-ready-v1`である。`ready-binding.json` SHA-256は`d1b3dbdbe52d26e2e5ca409bd4a1f5c9efe058341c1f844ff6e63740f8240670`、`harness-trust.json`は`32e525bd0168f12601d7d0f16ec039b62e69912816c59eaa9511bbcbd7ce1c17`、`qa-attestation.json`は`bac8dcc71368d5463bc9e6d6589236014560129292d0b69700c4246bb68cf15c`、`SHA256SUMS`は`a70bca0fa953966da7b867f61e4aec361019345b2b3ae07b91cebe3ec30c6258`である。execute bindingのlauncher trustは、launcher sourceが不変なため上記`eec6922`固定を維持し、`launcher-trust.json` SHA-256は`21e397f172951ce40550bacc4f906fb569c7593e343df63863b851f47b3b6279`である。

artifactは`execution_mode=one_case`、`status=ready_for_one_case`、`actual_eligible=true`、`promotion_eligible=false`、run ID固定、最大1回、output no-reuseである。通常one-case argvにはprofile optionを追加しない。実行runnerはmarker-aware strict `e93a2c1`、validatorは`8263545`、Bは`7e59bae`、residentは`319d618`へ固定した。

live-preflight SHA-256は事前にはnullである。これは入力不足ではなく、service停止後にlauncherのlive gateが全て通過してから新規生成するというpolicyである。schema、run ID、device mapping、VRAM下限とzero-owner gateはready artifactで固定し、生成後のpath、SHA-256、inode、modeはlauncher最終evidenceとstrict runner summary/raw evidenceへ束縛する。

canonical dry-run evidenceは`resident-one-case-smoke-ready-dry-run-v1`に保存した。process countsはsudo/keepalive/stop/launcher/start/rocprof/capture tool/docker/docker exec/container curl total/version/endpoint/stopped-gate poll/probeの全て0、service/GPU/model loadは未実行である。evidence SHA-256は`cb191034d19110a1f02734cb0b08f3f01b6847952f901c07d5ea195bd3509297`、`SHA256SUMS`は`43b353ab3ef44f21d269a94ac1cce099e0c1fe0798b5d31349f44ade2693bf75`である。

正式probe 1回のprocess countはDocker 9、docker exec 6、互換container curl total 6である。互換totalはargv内の固定curl path包含数であり、version 1と非version endpoint区分5へ分けて記録する。旧値5はservice停止前に拒否する。

回帰は既存chainとprofile追加を含む194 tests、marker chain 55 tests（25 subtests）、diagnostic capture 11 testsが通過した。capture、producer、profiler、selectorの関連集合85 testsも通過し、marker独立QAの手動境界15件をattestationへ記録した。

## 次の行動

base one-case実行は別の明示承認を必要とする。同一PTY sudo cache、全pre-stop/live gate、未使用outputのいずれかが不成立なら実行しない。one-case結果はpromotionへ使用しない。

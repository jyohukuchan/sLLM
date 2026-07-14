# AQ4 P2 resident smoke ready maintenance v0.1

## 前回の要点

旧ready chainはexecute launcher `bb7a5fb`、strict runner `774f6dd`、maintenance harness `fabd520`を固定していた。ROCTx marker対応runner `e93a2c1`とprofile diagnostic captureを使うには、runnerからready artifactまでの再固定が必要だった。

## 今回の変更点

maintenance harnessをcommit `f586f9a124e5af302fc35653a33702c2d56ad77c`、tree `2485708f38de148b5d931459cb2a0315124655f6`、Git blob `9b523dac5f587ea3cdd309cc5fc18d46f431a6de`、SHA-256 `53b7a1f233721efc254245d5c04666514c99e9884aedc74f525e2f73594c541c`へ再固定した。launcherはcommit `eec6922fa9c90267213d2749c5dc816be54de527`、tree `f6cef14d1e2a75dc1a12371d2a8e2a754d506482`、Git blob `c422e4235a2ee6595cf43656c573b7e863489f9e`、SHA-256 `607b7c9ad0bf7aa8e8b9303f60209b4a6dc998886dbd8af86d83955984232835`である。

harnessは同一PTYで事前prime済みの`sudo -n -v`だけを受け入れる。パスワード、API key、HTTP authorization、prompt/response本文をargv、environment、stdout/stderr evidenceへ保存しない。

実行順序を次に固定する。

1. sudo cache、active/running service、gateway PID、worker PID、NRestarts、ControlGroupを検査する。
2. served manifest、worker binary、package manifest、1045-file package treeをchunked SHA-256で照合する。
3. AMD SMI index 2、BDF `0000:47:00.0`、UUID `a8ff7551-0000-1000-80e9-ddefa2d60f55`、KFD ID `51545`、node 2、単一worker owner、device lock保持を確認する。正式なHTTP gateは固定済み`/usr/bin/docker`から、固定済みOpenWebUI container IDへ`docker exec`し、コンテナ名前空間内の固定済みcurlでgateway `/healthz`、`/readyz`、認証済み`/v1/models`、OpenWebUI `/health`を検査する。host直結HTTPは到達性診断だけでありgateにしない。
4. 復旧必須のdurable markerをatomic no-replace、mode 0444、fsync済みで保存する。
5. `sudo -n systemctl stop ullm-openai.service`を実行し、最大30秒、0.25秒から1秒までのbackoffで停止資源をpollする。対象service inactive、worker不在、AMD/KFD owner 0、lock free、VRAM freeを連続2回観測してからだけlauncherへ進む。pre-stop worker PIDのAMD/KFD解放とpre-stop service MainPIDのlock解放だけを待機対象とし、unknown/foreign/new PIDやzero後の再出現は即時fail-closedにする。各観測はsource別raw SHA、parsed PID、VRAM、raw非保存のproc cmdline SHA、分類を独立したmode 0444 JSONへatomic保存する。poll中は10秒間隔でsudo cacheを維持する。
6. base modeでは`eec6922` launcherへone-case bindingを渡す。launcher自身が30秒間隔でsudo keepaliveを行う。
7. 外側`finally`はlauncher開始前・途中・完了の成否にかかわらず`sudo -n systemctl start`を試行する。
8. active/running、新gateway PID、新worker PID、NRestarts不変、ControlGroup不変、lock再保持、manifest/worker/package hash不変、単一GPU owner、固定済みcontainer/image/network/curlとgateway/OpenWebUI復帰を最大120秒で検査する。API keyは固定済みfile identityから読み、Authorization headerだけを`docker exec -i ... curl --header @-`のstdinへ渡し、argv、log、evidenceへ保存しない。

base ready artifactは`resident-one-case-smoke-ready-v1`である。`ready-binding.json` SHA-256は`d4740f201030f1d5836fff1ea16aba177f7214b83c30e9e2dd32e6b36a44c473`、`harness-trust.json`は`541d0287d2c894f370e409e26140475680da426739ffbea90c958bf3cd6f7e95`、`qa-attestation.json`は`3afd7589e8351ef084a2aca2da2f33e2fb0fde7c3c148701b23acb2c8b02306b`、`SHA256SUMS`は`6f4b0162d660811fbb6f5ee67202c00dc26c7ad7823e4a9783399f62c8c5951a`である。execute bindingのlauncher trustは、launcher sourceが不変なため上記`eec6922`固定を維持し、`launcher-trust.json` SHA-256は`21e397f172951ce40550bacc4f906fb569c7593e343df63863b851f47b3b6279`である。

artifactは`execution_mode=one_case`、`status=ready_for_one_case`、`actual_eligible=true`、`promotion_eligible=false`、run ID固定、最大1回、output no-reuseである。通常one-case argvにはprofile optionを追加しない。実行runnerはmarker-aware strict `e93a2c1`、validatorは`8263545`、Bは`7e59bae`、residentは`319d618`へ固定した。

live-preflight SHA-256は事前にはnullである。これは入力不足ではなく、service停止後にlauncherのlive gateが全て通過してから新規生成するというpolicyである。schema、run ID、device mapping、VRAM下限とzero-owner gateはready artifactで固定し、生成後のpath、SHA-256、inode、modeはlauncher最終evidenceとstrict runner summary/raw evidenceへ束縛する。

canonical dry-run evidenceは`resident-one-case-smoke-ready-dry-run-v1`に保存した。process countsはsudo/keepalive/stop/launcher/start/rocprof/capture tool/docker/docker exec/container curl total/version/endpoint/stopped-gate poll/probeの全て0、service/GPU/model loadは未実行である。evidence SHA-256は`9e70f88aaecc2ef36f14e846897bcc141dd01b1896993ccc3de4689fa59eff95`、`SHA256SUMS`は`fee1f70bdb750886133eebc72b7b95300c593ced810e22f9b073e2a30d6fb77b`である。

正式probe 1回のprocess countはDocker 9、docker exec 6、互換container curl total 6である。互換totalはargv内の固定curl path包含数であり、version 1と非version endpoint区分5へ分けて記録する。旧値5はservice停止前に拒否する。

回帰は既存chainとprofile追加を含む190 tests、marker chain 55 tests、diagnostic capture 11 testsが通過した。capture、producer、profiler、selectorの関連集合85 testsも通過し、marker独立QAの手動境界15件をattestationへ記録した。

## 次の行動

base one-case実行は別の明示承認を必要とする。同一PTY sudo cache、全pre-stop/live gate、未使用outputのいずれかが不成立なら実行しない。one-case結果はpromotionへ使用しない。

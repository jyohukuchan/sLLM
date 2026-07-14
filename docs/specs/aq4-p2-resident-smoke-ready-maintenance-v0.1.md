# AQ4 P2 resident smoke ready maintenance v0.1

## 前回の要点

旧ready chainはexecute launcher `bb7a5fb`、strict runner `774f6dd`、maintenance harness `fabd520`を固定していた。ROCTx marker対応runner `e93a2c1`とprofile diagnostic captureを使うには、runnerからready artifactまでの再固定が必要だった。

## 今回の変更点

maintenance harnessをcommit `ffbb9cc33d662aac1d5b52480323cd3a9c5b801b`、tree `3580d2fe7934353adb7059ffe16f9339ffcf8024`、Git blob `5305b471b6432d10d91df4dd44e31f18238b81cb`、SHA-256 `a3d8c77015e8af7952dc5027205d7a4cc59bb955f0728467ba071fe76eb6b34f`へ再固定した。launcherはcommit `0994367b08534909ff42771ee5b080ec56ca4d01`、tree `602605dafde92876ceffcb8f79c825934509b549`、Git blob `c60a80299a8ac281875fe45b763e52dfed7c9a29`、SHA-256 `9e1e7ddd3ec9911326aa7eb81702219766ff1422494fef22cc6a2be87154c036`である。

harnessは同一PTYで事前prime済みの`sudo -n -v`だけを受け入れる。パスワード、API key、HTTP authorization、prompt/response本文をargv、environment、stdout/stderr evidenceへ保存しない。

実行順序を次に固定する。

1. sudo cache、active/running service、gateway PID、worker PID、NRestarts、ControlGroupを検査する。
2. served manifest、worker binary、package manifest、1045-file package treeをchunked SHA-256で照合する。
3. AMD SMI index 2、BDF `0000:47:00.0`、UUID `a8ff7551-0000-1000-80e9-ddefa2d60f55`、KFD ID `51545`、node 2、単一worker owner、device lock保持を確認する。正式なHTTP gateは固定済み`/usr/bin/docker`から、固定済みOpenWebUI container IDへ`docker exec`し、コンテナ名前空間内の固定済みcurlでgateway `/healthz`、`/readyz`、認証済み`/v1/models`、OpenWebUI `/health`を検査する。host直結HTTPは到達性診断だけでありgateにしない。
4. 復旧必須のdurable markerをatomic no-replace、mode 0444、fsync済みで保存する。
5. `sudo -n systemctl stop ullm-openai.service`を実行し、開始時に一度だけ固定したmonotonic absolute deadline 30秒まで、0.25秒から1秒までのbackoffで停止資源をpollする。各観測と各probeの前後で期限を確認し、subprocess timeoutは2秒と残時間の小さい方へ制限する。sudo keepaliveもprobe間で期限を再確認し、同じ2秒・残時間上限を使う。対象service inactive、worker不在、AMD/KFD owner 0、lock free、VRAM freeを期限内に連続2回観測してからだけlauncherへ進む。2回目のstable観測または単一probeが期限を越えた場合もtimeoutとしてfail-closedにする。pre-stop worker PIDのAMD/KFD解放とpre-stop service MainPIDのlock解放だけを待機対象とし、unknown/foreign/new PIDやzero後の再出現は即時fail-closedにする。各観測はsource別raw SHA、parsed PID、VRAM、raw非保存のproc cmdline SHA、期限checkpoint、timeoutしたprobeの部分証拠、分類を独立したmode 0444 JSONへatomic保存する。poll中は10秒間隔でsudo cacheを維持する。
6. base modeでは`0994367` launcherへone-case bindingを渡す。launcher自身が30秒間隔でsudo keepaliveを行う。AMD process JSONはlauncherとharnessの共通strict parserで検査し、owner zeroはROCm 7.2.1が出力するexact `No running processes detected` sentinelだけを受理する。active ownerは既知のroot/entry/process_info fieldsと正のinteger PIDを要求し、sentinel混在、別文字列、extra fields、bool PID、他GPUを拒否する。raw本文は保存せずSHA-256を常に残し、拒否時はsecret-freeなreason code、top-level type、root/entry keysをimmutable evidenceへ保存する。
7. 外側`finally`はlauncher開始前・途中・完了の成否にかかわらず`sudo -n systemctl start`を試行する。
8. active/running、新gateway PID、新worker PID、NRestarts不変、ControlGroup不変、lock再保持、manifest/worker/package hash不変、単一GPU owner、固定済みcontainer/image/network/curlとgateway/OpenWebUI復帰を最大120秒で検査する。API keyは固定済みfile identityから読み、Authorization headerだけを`docker exec -i ... curl --header @-`のstdinへ渡し、argv、log、evidenceへ保存しない。

base ready artifactは`resident-one-case-smoke-ready-v1`である。`ready-binding.json` SHA-256は`effc1f9c47fe7835df9518e00dab3a6980a32a1973e3143c058ebcc1ddd100e8`、`harness-trust.json`は`e1d593914511982df6aa966fb61f816fc8d517c1b24777e750d62f0a32560c2c`、`qa-attestation.json`は`2bf196ab4df74aac1104e559c66c22ba9ebad3e4e3cb84818dfa039fad2812ae`、`SHA256SUMS`は`27b7a671935395a202c87ae2eb558df8bffce2babe7de0e9a5d6ec8577654861`である。execute bindingは新launcherを固定し、`execute-binding.json` SHA-256は`db4bd2abd1aad931ae99a720b065baafd7176de1736aa493b43e772893f08361`、`launcher-trust.json`は`6ab9a008736563fcbce8bd8dd1af3478b512c54215d85905f3fd4297333ff8a2`である。

artifactは`execution_mode=one_case`、`status=ready_for_one_case`、`actual_eligible=true`、`promotion_eligible=false`、run ID固定、最大1回、output no-reuseである。通常one-case argvにはprofile optionを追加しない。実行runnerはmarker-aware strict `e93a2c1`、validatorは`8263545`、Bは`7e59bae`、residentは`319d618`へ固定した。

live-preflight SHA-256は事前にはnullである。これは入力不足ではなく、service停止後にlauncherのlive gateが全て通過してから新規生成するというpolicyである。schema、run ID、device mapping、VRAM下限とzero-owner gateはready artifactで固定し、生成後のpath、SHA-256、inode、modeはlauncher最終evidenceとstrict runner summary/raw evidenceへ束縛する。

canonical dry-run evidenceは`resident-one-case-smoke-ready-dry-run-v1`に保存した。process countsはsudo/keepalive/stop/launcher/start/rocprof/capture tool/docker/docker exec/container curl total/version/endpoint/stopped-gate poll/probeの全て0、service/GPU/model loadは未実行である。evidence SHA-256は`7f41c7ca0a997b653f753b7849238d3df3ce43509b14e8f71c93409dcbbd4c8a`、`SHA256SUMS`は`3ded679e32dc399717a378cdc24f75ce40cb3657babeb238da2a0586de04ca32`である。

正式probe 1回のprocess countはDocker 9、docker exec 6、互換container curl total 6である。互換totalはargv内の固定curl path包含数であり、version 1と非version endpoint区分5へ分けて記録する。旧値5はservice停止前に拒否する。

回帰は既存chainとstrict AMD schema追加を含む204 tests、marker chain 55 tests（25 subtests）、diagnostic capture 11 testsが通過した。capture、producer、profiler、selectorの関連集合85 testsも通過し、marker独立QAの手動境界15件をattestationへ記録した。

## 次の行動

base one-case実行は別の明示承認を必要とする。同一PTY sudo cache、全pre-stop/live gate、未使用outputのいずれかが不成立なら実行しない。one-case結果はpromotionへ使用しない。

# AQ4 P2 resident smoke ready maintenance v0.1

## 前回の要点

旧ready chainはexecute launcher `bb7a5fb`、strict runner `774f6dd`、maintenance harness `fabd520`を固定していた。ROCTx marker対応runner `e93a2c1`とprofile diagnostic captureを使うには、runnerからready artifactまでの再固定が必要だった。

## 今回の変更点

maintenance harnessをcommit `426290f1087ef7d9003dda921c24468067cd1b5a`、tree `8a932c6e25861bd225c325fb2a100a972fdb0cce`、Git blob `99d0dcca20de6535352bc02a7f700ae1b646ff36`、SHA-256 `f2d8e67db466770b19024ea296288f899d7d2c7f3712659abc673041643ff90e`へ再固定した。launcherはcommit `eec6922fa9c90267213d2749c5dc816be54de527`、tree `f6cef14d1e2a75dc1a12371d2a8e2a754d506482`、Git blob `c422e4235a2ee6595cf43656c573b7e863489f9e`、SHA-256 `607b7c9ad0bf7aa8e8b9303f60209b4a6dc998886dbd8af86d83955984232835`である。

harnessは同一PTYで事前prime済みの`sudo -n -v`だけを受け入れる。パスワード、API key、HTTP authorization、prompt/response本文をargv、environment、stdout/stderr evidenceへ保存しない。

実行順序を次に固定する。

1. sudo cache、active/running service、gateway PID、worker PID、NRestarts、ControlGroupを検査する。
2. served manifest、worker binary、package manifest、1045-file package treeをchunked SHA-256で照合する。
3. AMD SMI index 2、BDF `0000:47:00.0`、UUID `a8ff7551-0000-1000-80e9-ddefa2d60f55`、KFD ID `51545`、node 2、単一worker owner、device lock保持、gateway `/readyz`、OpenWebUI `/health`を確認する。
4. 復旧必須のdurable markerをatomic no-replace、mode 0444、fsync済みで保存する。
5. `sudo -n systemctl stop ullm-openai.service`を実行し、strict launcher gateで対象service inactive、worker不在、AMD/KFD owner 0、lock free、exact environmentを確認する。
6. base modeでは`eec6922` launcherへone-case bindingを渡す。launcher自身が30秒間隔でsudo keepaliveを行う。
7. 外側`finally`はlauncher開始前・途中・完了の成否にかかわらず`sudo -n systemctl start`を試行する。
8. active/running、新gateway PID、新worker PID、NRestarts不変、ControlGroup不変、lock再保持、manifest/worker/package hash不変、単一GPU owner、gateway/OpenWebUI復帰を最大120秒で検査する。

base ready artifactは`resident-one-case-smoke-ready-v1`である。`ready-binding.json` SHA-256は`b71a67a240668626c1c5d9c35cbd9b5b658d282265120aa7721f75516ba67c69`、`harness-trust.json`は`b73d52752a3cb5470c41ae3d4ca9067036737d2ff2600af1803b171e82cd69a4`、`qa-attestation.json`は`d68e78656aabdf9a09fcf660566cec43d970753f7a79329c3990f3196a1e2edc`、`SHA256SUMS`は`45e543a5eeb42705e09c3639c9ae5219437fe763dbcf4e49f8139f7cdb34ebd8`である。execute bindingのlauncher trustは上記`eec6922`へ固定し、`launcher-trust.json` SHA-256は`21e397f172951ce40550bacc4f906fb569c7593e343df63863b851f47b3b6279`である。

artifactは`execution_mode=one_case`、`status=ready_for_one_case`、`actual_eligible=true`、`promotion_eligible=false`、run ID固定、最大1回、output no-reuseである。通常one-case argvにはprofile optionを追加しない。実行runnerはmarker-aware strict `e93a2c1`、validatorは`8263545`、Bは`7e59bae`、residentは`319d618`へ固定した。

live-preflight SHA-256は事前にはnullである。これは入力不足ではなく、service停止後にlauncherのlive gateが全て通過してから新規生成するというpolicyである。schema、run ID、device mapping、VRAM下限とzero-owner gateはready artifactで固定し、生成後のpath、SHA-256、inode、modeはlauncher最終evidenceとstrict runner summary/raw evidenceへ束縛する。

canonical dry-run evidenceは`resident-one-case-smoke-ready-dry-run-v1`に保存した。process countsはsudo/stop/launcher/start/rocprof/capture toolの全て0、service/GPU/model loadは未実行である。evidence SHA-256は`479f9b4dd263c16d80d2f370dbf4c45fbb03b34ef89f0e0f8ca554b9918fda13`、`SHA256SUMS`は`4f1b0b5fc4cf14e18f6487687385bb47975f890b3f5f7e9aa62c717e2b4b3ed4`である。

回帰は既存chainとprofile追加を含む168 tests、marker chain 55 tests、diagnostic capture 11 testsが通過した。capture、producer、profiler、selectorの関連集合85 testsも通過し、marker独立QAの手動境界15件をattestationへ記録した。

## 次の行動

base one-case実行は別の明示承認を必要とする。同一PTY sudo cache、全pre-stop/live gate、未使用outputのいずれかが不成立なら実行しない。one-case結果はpromotionへ使用しない。

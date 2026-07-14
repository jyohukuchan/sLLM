# AQ4 P2 resident smoke ready maintenance v0.1

## 前回の要点

旧ready chainはexecute launcher `bb7a5fb`、strict runner `774f6dd`、maintenance harness `fabd520`を固定していた。ROCTx marker対応runner `e93a2c1`とprofile diagnostic captureを使うには、runnerからready artifactまでの再固定が必要だった。

## 今回の変更点

maintenance harnessをcommit `c5f6f2ac0130642f3d5c31204e84e15eecaf1e29`、tree `abb71b1f736414aebe0d43f05c065787d2b360eb`、Git blob `c70126325fab72612c78c45935d747ba70a74c62`、SHA-256 `7a04647c50334f3f7df12e4def0272e43e1dcb37db953f680a589f9fe33aebf0`へ再固定した。launcherはcommit `bdb06083ca3646c8f934fea10dac691a6efd4626`、SHA-256 `ddd84b5b85dc303f381f44048b03eeb1e542bcf8173dfe506aeb0e7ebd235ee5`である。

harnessは同一PTYで事前prime済みの`sudo -n -v`だけを受け入れる。パスワード、API key、HTTP authorization、prompt/response本文をargv、environment、stdout/stderr evidenceへ保存しない。

実行順序を次に固定する。

1. sudo cache、active/running service、gateway PID、worker PID、NRestarts、ControlGroupを検査する。
2. served manifest、worker binary、package manifest、1045-file package treeをchunked SHA-256で照合する。
3. AMD SMI index 2、BDF `0000:47:00.0`、UUID `a8ff7551-0000-1000-80e9-ddefa2d60f55`、KFD ID `51545`、node 2、単一worker owner、device lock保持、gateway `/readyz`、OpenWebUI `/health`を確認する。
4. 復旧必須のdurable markerをatomic no-replace、mode 0444、fsync済みで保存する。
5. `sudo -n systemctl stop ullm-openai.service`を実行し、strict launcher gateで対象service inactive、worker不在、AMD/KFD owner 0、lock free、exact environmentを確認する。
6. `bdb0608` launcherへone-case bindingを渡す。launcher自身が30秒間隔でsudo keepaliveを行う。
7. 外側`finally`はlauncher開始前・途中・完了の成否にかかわらず`sudo -n systemctl start`を試行する。
8. active/running、新gateway PID、新worker PID、NRestarts不変、ControlGroup不変、lock再保持、manifest/worker/package hash不変、単一GPU owner、gateway/OpenWebUI復帰を最大120秒で検査する。

base ready artifactは`resident-one-case-smoke-ready-v1`である。`ready-binding.json` SHA-256は`8e8d062b5cf8c883f4256af7a6d40057aa0342c1cf93e31db8ccc65cd683edc7`、`harness-trust.json`は`5a41f0d6b3897b29c44f5ea62bd6c4c00b858653d6c2138dca0f5cdebaa9aa55`、`qa-attestation.json`は`72225d27f773d1e8f2b7f51d1466c19104a505a4bf8612f9bcd6a68a482106ef`、`SHA256SUMS`は`116f8df5955b21732f4b05941e1b38b25891fb21822cafd5bdaf4951819a1280`である。

artifactは`execution_mode=one_case`、`status=ready_for_one_case`、`actual_eligible=true`、`promotion_eligible=false`、run ID固定、最大1回、output no-reuseである。通常one-case argvにはprofile optionを追加しない。実行runnerはmarker-aware strict `e93a2c1`、validatorは`8263545`、Bは`7e59bae`、residentは`319d618`へ固定した。

live-preflight SHA-256は事前にはnullである。これは入力不足ではなく、service停止後にlauncherのlive gateが全て通過してから新規生成するというpolicyである。schema、run ID、device mapping、VRAM下限とzero-owner gateはready artifactで固定し、生成後のpath、SHA-256、inode、modeはlauncher最終evidenceとstrict runner summary/raw evidenceへ束縛する。

canonical dry-run evidenceは`resident-one-case-smoke-ready-dry-run-v1`に保存した。process countsはsudo/stop/launcher/start/rocprof/capture toolの全て0、service/GPU/model loadは未実行である。evidence SHA-256は`d87cc2120a4e519658dcea7a9ba6248883b7eaa8fba70bd4a60a006c473cb2a9`、`SHA256SUMS`は`5df2142eb99f0993d6108a6fd9dcce4c21cc4156262ac845f87ba459da9dcc30`である。

回帰は既存chainとprofile追加を含む155 tests、marker chain 55 tests、diagnostic capture 8 testsが通過した。marker独立QAの手動境界15件もattestationへ記録した。

## 次の行動

base one-case実行は別の明示承認を必要とする。同一PTY sudo cache、全pre-stop/live gate、未使用outputのいずれかが不成立なら実行しない。one-case結果はpromotionへ使用しない。

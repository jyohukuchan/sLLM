# AQ4 P2 resident smoke ready maintenance v0.1

## 前回の要点

immutable execute launcher `bb7a5fb`とstrict runner `774f6dd`は独立QAを通過した。blocked artifactはlive-preflight未生成、`actual_eligible=false`であり、サービス停止と復帰をlauncherの外側で保証する仕組みを持たなかった。

## 今回の変更点

最小maintenance harnessをcommit `fabd520daaa878eca5c93b24d1faf092cafe3448`、tree `834861e57eb615d5075d8f23e5e58a442f7851e0`、Git blob `6b9c2316002c164df356e0dfcfa7adfe0c426332`、SHA-256 `7b45825e31df363d714880712eb2ea0f15332a9de6707fa404c87eb1d888621b`へ固定した。launcher本体は変更せず、commit `bb7a5fb00c453d911cd0ccb9499a47863c6eb07c`、SHA-256 `4f547d50b4a321196dbb8b2e7703843657c87a4d3d215c0325a6f8d267db5382`を使う。

harnessは同一PTYで事前prime済みの`sudo -n -v`だけを受け入れる。パスワード、API key、HTTP authorization、prompt/response本文をargv、environment、stdout/stderr evidenceへ保存しない。

実行順序を次に固定する。

1. sudo cache、active/running service、gateway PID、worker PID、NRestarts、ControlGroupを検査する。
2. served manifest、worker binary、package manifest、1045-file package treeをchunked SHA-256で照合する。
3. AMD SMI index 2、BDF `0000:47:00.0`、UUID `a8ff7551-0000-1000-80e9-ddefa2d60f55`、KFD ID `51545`、node 2、単一worker owner、device lock保持、gateway `/readyz`、OpenWebUI `/health`を確認する。
4. 復旧必須のdurable markerをatomic no-replace、mode 0444、fsync済みで保存する。
5. `sudo -n systemctl stop ullm-openai.service`を実行し、strict launcher gateで対象service inactive、worker不在、AMD/KFD owner 0、lock free、exact environmentを確認する。
6. `bb7a5fb` launcherへone-case bindingを渡す。launcher自身が30秒間隔でsudo keepaliveを行う。
7. 外側`finally`はlauncher開始前・途中・完了の成否にかかわらず`sudo -n systemctl start`を試行する。
8. active/running、新gateway PID、新worker PID、NRestarts不変、ControlGroup不変、lock再保持、manifest/worker/package hash不変、単一GPU owner、gateway/OpenWebUI復帰を最大120秒で検査する。

ready artifactは`resident-one-case-smoke-ready-v1`である。`ready-binding.json` SHA-256は`4a257dd0edaec5eafa27e05b0804311435e57fbda5b4a5ba6e74a1f1c80a028a`、`harness-trust.json`は`4f4e6dfc695f20fdabb2a2970307697a78d8c722ef635749ed907a4506669a65`、`qa-attestation.json`は`c44c5035c223fe9ceb82632cddaff2bbc06c3aad805d984beaa97b21ca2b3763`、`SHA256SUMS`は`6f1ddfb356f78ab5abde577329b71256f164f80ea928e5c3725c7192668b48b2`である。

artifactは`status=ready_for_one_case`、`actual_eligible=true`、`promotion_eligible=false`、run ID固定、最大1回、output no-reuseである。実行runnerはstrict `774f6dd`で、`ee341c0`はdriver argv透過を導入した祖先としてlineageに残す。validator `481ae68`、B `d465ac2`、resident `319d618`も固定する。

live-preflight SHA-256は事前にはnullである。これは入力不足ではなく、service停止後にlauncherのlive gateが全て通過してから新規生成するというpolicyである。schema、run ID、device mapping、VRAM下限とzero-owner gateはready artifactで固定し、生成後のpath、SHA-256、inode、modeはlauncher最終evidenceとstrict runner summary/raw evidenceへ束縛する。

canonical dry-run evidenceは`resident-one-case-smoke-ready-dry-run-v1`に保存した。process countsはsudo/stop/launcher/startの全て0、service/GPU/model loadは未実行である。evidence SHA-256は`73134967d9bf99bef47b685309e84a07a0be16bdc460664bc903681009e0f8da`、`SHA256SUMS`は`14687c852c68641fb7b002372965d9a6c0f7ffb0624d7b813d5719755685defb`である。

## 次の行動

ready artifactとharness commitの独立QAを行う。QA完了後もone-case実行は別の明示承認を必要とし、同一PTY sudo cache、全pre-stop/live gate、未使用outputのいずれかが不成立なら実行しない。one-case結果はpromotionへ使用しない。

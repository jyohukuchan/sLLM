# AQ4 P2 resident one-case smoke execute launcher v0.1

## 前回の要点

prepared input root、strict runner `774f6dd`、validator `481ae68`、B sidecar `d465ac2`までは固定済みである。prepared rootの`preflight.json`は合成契約であり、actual runのGPU、service、owner、lock、VRAM状態を証明しない。旧dry-run launcherもactual GPU commandを禁止していた。

## 今回の変更点

execute launcherをcommit `bb7a5fb00c453d911cd0ccb9499a47863c6eb07c`、tree `fafc6045b024b40749aeae2c526b1aae952806d0`、Git blob `cf17c68f4a6fec65c99eb3b32832d7fc193aaf23`、SHA-256 `4f547d50b4a321196dbb8b2e7703843657c87a4d3d215c0325a6f8d267db5382`へ固定した。

launcherはactual runnerの前に次をfail-closedで検査する。

- input root exact 19 members、B exact members、Python、validator、strict runner、resident driver、served manifest、launcher selfのpath、single-link regular file、FD/path identity、SHA-256
- AMD SMIとrocminfoの呼出path、symlink chain、解決済みROCm 7.2.1実体、systemctl、pgrep、sudoのSHA-256
- `sudo -n -v`、対象2 service inactive、旧worker不在、AMD SMI GPU index 2とBDF/UUID/KFD ID/node 2の一意対応、AMD SMI/KFD owner 0、mode 0600 device lockのnonblocking取得可能性
- runtime device index 1、visible token `1`、served manifest/build commit、30 guardsだけを含むexact runner environment

live-preflight sidecarはgate通過後に新規evidence rootへatomic no-replace、mode 0444で作る。prepared合成preflightのpath/SHA/role、runtime mapping、service、worker、owner、lock、environment、VRAM、8 probeのlabel別exact argv/exit/lowercase SHA/timestampをnested unknownなしで記録する。strict runnerはactual one-caseでsidecarを必須とし、dry-runでは禁止する。sidecarはrunner前後とfinalize前にもidentity/SHAを再検証する。

trust snapshotはvalidator直前、runner直前、runner終了後、evidence finalize直前に全fileをFDから再読込し、path identityとSHAを照合する。input/B directoryとROCm symlinkも同じ4地点で再検証する。各地点の置換負例を回帰testに持つ。

evidence safetyは実行状態から決める。runner未開始は`gpu_command_executed=false`かつ`model_load_executed=false`、開始済みで到達点が不明なら両方`unknown`としてrunを失敗扱いにする。runner outcomeが到達を証明した正常完了または途中失敗だけ両方`true`にする。初期値のままactual成功を記録しない。

sudo credential cacheは外部の同一PTYで事前prime済みであることを要求し、launcherはパスワードをargv、environment、stdout/stderr evidenceへ保存しない。runner中は30秒間隔で`sudo -n -v`を行い、失敗時はrunnerを中断してfinallyの状態復旧を優先する。

canonical artifactは`resident-one-case-smoke-execute-binding-v1`に置く。`execute-binding.json` SHA-256は`7105ab60c691aa864e71f15acc7e9ffc5bb08eafd56f05ed11fa888a4f51e48a`、launcher selfを独立固定する`launcher-trust.json`は`2c6bd46d342663dd4d7e586dd8af05b63c9cabc1592fc196460ed68af9e5ba73`、`SHA256SUMS`は`77c4f10facf17452720840437d7080514fcb75d23694483ea7b8ccf2c4455e67`である。artifactは`status=blocked_pending_live_preflight_and_qa`、`actual_eligible=false`、live-preflight SHAはnullであり、実行許可ではない。

## 次の行動

独立QAでlauncher commitとcanonical artifactを検証する。QA完了後も、同一PTYのsudo cache、service/worker/owner/lock gateが全て通る新しい明示的なexecute承認がない限りactualへ進まない。今回の成果物ではGPU command、model load、service停止を実行しない。

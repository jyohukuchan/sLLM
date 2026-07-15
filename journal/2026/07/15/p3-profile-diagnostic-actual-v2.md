# P3 profile diagnostic actual v2

## 前回の要点

- profile operator v1 は別 PTY の service stop と競合し、`pre-stop-snapshot` で `/run/ullm/r9700.lock` が消失したため未到達で失敗した。
- commit `430f0889ef09582deb22b4726a8483711b7c16a3` の quiet-window は primary 25 samples と terminal 25 samples で service epoch、worker、lock、AMD/KFD owner、formal health が安定し、外部 service process と `pts/5` の観測数が0だった。

## 今回の変更点

### 単一実行契約

- operator manifest commit: `2bb73a94baf1b63ebdbd36b3d77fe0a52d5d39dc`
- manifest SHA-256: `d32ecad79dcbb04bee60237b98a324355b1b28aa956bb0d49124edd41d08aee4`
- canonical command SHA-256: `7474807b74e7f460a7ab61e71d510321f394b3bd379636993b3fc9e52fb798dc`
- exact argv count: 10、`shell=false`、manifest cwd exact、maximum invocation 1。
- 開始直前に operator/quiet-window/profile-ready の全 `SHA256SUMS`、全 input hash、7 fresh outputs の不存在、quiet-window GO、外部/`pts/5`/systemctl/maintenance process 0を再検証した。
- 現行 epoch は service PID `1722227`、`NRestarts=1`、worker PID `1722613`、lock inode `762398`、AMD/KFD owners `[1722613]` で一致した。formal health は Docker 9、docker exec 6、curl 6、4 endpoints HTTP 200だった。
- canonical start: `1784088403841085864` ns (`2026-07-15T13:06:43.841086+09:00`)
- canonical end: `1784091283022794701` ns (`2026-07-15T13:54:43.022795+09:00`)
- elapsed: `2879181708837` ns。return code `1`。invocation count 1。再試行なし。

### service、capture、cleanup

- maintenance は pre-stop snapshot、durable marker、service stop、stable2 stopped gates、profile capture、service startまで進行した。
- `ullm-openai.service` は `13:07:07.166317` に停止開始、`13:07:07.383341` に停止完了した。stopped gates は2 polls / 10 probesでPASSし、両service inactive、old worker/AMD/KFD owners空、trusted lock substrate freeだった。
- trusted substrate は directory inode `763585`、lock inode `763586` で作成され、capture終了後に holderなしを確認してcleanup PASS、lockとdirectoryは削除された。children remainingは空、process-group cleanupも完了した。
- service startは `13:07:18.965346`、activeは `13:07:19.364243`、application startup completeは `13:07:22.138284` だった。自動restartやGPU fault、KFD fault、OOMの記録はなかった。
- maintenance process counts は `systemctl_stop=1`、`systemctl_start=1`、`capture_tool=1`、`rocprof=1`、`launcher=1`、runnerは0だった。

### 正確な失敗原因

- rocprof自体は起動し、profile launcherを開始した。launcherは最初の trusted validator subprocess で失敗し、runnerを開始していない。
- launcher failure: stage `validator`、reason `trusted validator subprocess rejected root/B`。
- validatorは return code 0で `prepared_not_executed` をstdoutへ返したが、rocprofの計測環境を継承したため、validator stderrへrocprof初期化・終了ログ890 bytesが出た。launcherの厳格契約は validator stderr non-empty を拒否するため、ここで安全に停止した。
- launcher safety は `runner_started=false`、`gpu_command_executed=false`、`model_load_executed=false`。resident runner output、raw/summary、capture artifact、kernel/API/memory records、ROCTx exact 12 markersは生成されていない。
- rocprof stdoutは250 bytes、stderrは1093 bytes。capture failureは exit 1、timeout false、outer signal false、process group cleanup completeだった。

### restoreとpost-health

- outer restoreはserviceを正常に再起動したが、maintenanceの内部判定は `restored service epoch/NRestarts differs` でfalseになった。pre-stopの `NRestarts=1` に対し、明示的stop/start後のsystemd値が `NRestarts=0` になったためである。
- この不変条件は待機で変化しないが、harnessは最大120回のrecovery validationを続け、各回7.2 GiB package tree hashを再計算した。このため失敗判定後も全体で約48分を要した。介入や中断はしていない。
- 独立した正式post-healthはPASSした。service PID `1834403` は active/running、`Result=success`、`ExecMainCode=0`、`ExecMainStatus=0`、worker PID `1834494`、AMD/KFD owners `[1834494]`、lock busyだった。
- served manifest、worker、package manifest、package contentの4 hashはproduction値と一致した。formal container healthは9/6/6、gateway healthz/readyz/models/OpenWebUI healthはすべてHTTP 200だった。残存するmaintenance/capture/rocprof/launcher/runner processはなかった。

### 出力SHA-256

- maintenance `launcher-evidence.json`: `1f60fd29ce4a01bde0528265424893e67adf3bac6d499aec208ff2289f4250dc`
- launcher `launcher-evidence.json`: `b4ae7b8073e6a801f2d811cae49abd2b9a1e985730b1fdc6c4299fdea9066973`
- validator stdout/stderr: `7c463b16bab152c3554ee355938e1731b1ba65e3ea059adf22e0ccf329635c2a` / `92c62a2d0e5948f47b42441435c3f19843bd78301cd8c8ef9a1d6d6a7c1deaff`
- `capture-failure.json`: `47be5b03164daf967b1f2ec95e36983c2c734a3ffd8f16ee31ca84927221f58f`
- rocprof stdout/stderr: `ed40bf7212c23707ebb2ccfa439854eac27030d4385daed954c7d85991088d63` / `6cd7d37937dd4ffdac4891fd2439432155c1a20ebf456cdb7ff5dfdc0ac50782`
- operator result: `af3e33edfead192fd4bda5c93e5e25513841a8537bc98982959fc5a282a8a0ac`
- maintenance、launcher、capture、operator resultの各`SHA256SUMS`はPASSした。

## 次の行動

- このsingle-use authorizationは失敗証跡として閉じ、再実行しない。
- 次版ではrocprof環境をresident runnerだけに適用し、launcherのtrusted validator subprocessへ計測環境を継承させない。
- restore判定は明示的stop/startでresetされる`NRestarts`の絶対値一致ではなく、service結果、epoch変更、worker/lock/GPU owner、formal healthで判定する。失敗が決定的な場合は120回の全package再hashを繰り返さない。

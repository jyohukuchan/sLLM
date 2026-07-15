# P2 resident profile diagnostic actual attempt

## 前回の要点

- 通常の resident one-case actual v9 は commit `0246ba36c6fbcf9cee87ce3b843674e384ab9ff8` で成功し、1 model load、2 warmup、10 measured、全 digest 一致を記録した。
- profile diagnostic operator command は commit `5e7ecce84c54288453cf102405d6eaf845e6d501` の manifest を唯一の argv 源とした。

## 今回の変更点

### 実行契約

- manifest: `benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-profile-operator-command-v1/command-manifest.json`
- manifest SHA-256: `e4a989fb149d7dc7a583a537fdcef0c6e05257e14978ba6b496661090e45a7a4`
- canonical command SHA-256: `549a79b46f39898c82cd7598fb179f32bf941fed6bd8a2470182387c5d8f5cbb`
- exact argv count: 10
- `shell=false`
- same PTY で `sudo -v` を成功させた後、manifest SHA、command SHA、argv count、shell mode、7 fresh outputs の ABSENT を実行直前に再検証した。
- canonical start: `1784086743426960100` ns (`2026-07-15T12:39:03.426960+09:00`)
- canonical end: `1784086761466790255` ns (`2026-07-15T12:39:21.466790+09:00`)
- elapsed: `18039830155` ns
- return code: `1`
- invocation count: 1。再試行はしていない。

### 結果と失敗地点

- status: `failed`
- failure stage: `pre-stop-snapshot`
- exact reason: `[Errno 2] No such file or directory: '/run/ullm/r9700.lock'`
- `sudo-prevalidate` だけが実行され、return code は 0 だった。
- `launcher_started=false`。capture tool、rocprof、profile launcher、runner、resident driver は開始されていない。
- process counts は `sudo=1` で、`systemctl_stop=0`、`systemctl_start=0`、`capture_tool=0`、`rocprof=0`、`launcher=0`、Docker と container curl の各 count も 0 だった。
- safety は `service_touched=false`、`service_stopped=false`、`gpu_command_executed=false`、`model_load_executed=false` だった。
- service stop が未試行のため outer restore は `attempted=false`、`passed=true` だった。
- profile launcher evidence、runner output、capture output、capture artifact、`rocprof.stdout`、`rocprof.stderr` はすべて未生成だった。このため kernel/API/memory/capability と exact 12 ROCTx markers、2 warmup + 10 measured の profile 証跡は得られていない。

### 証跡と実行後の安全確認

- maintenance evidence の `SHA256SUMS` は PASS した。
- `launcher-evidence.json` SHA-256: `6ecf3c4414185627cf235547334b1bb34fcd274bae3bafda9f0ce3864bc871fc`
- `SHA256SUMS` SHA-256: `365e5d2fdb1582d3975271ed37ea81dd435230262f6ab71ee93a4d46efe0ca4f`
- 対象 harness、capture tool、launcher、profiler、Python、target manifest の profile trust は `before-start` と `finalize-before` の両方で PASS した。
- 対象プロセスの残存はなかった。
- read-only の systemd/journal 監査で、別 PTY `pts/5` から `systemctl stop ullm-openai.service` が `12:39:07.837412` に発行され、systemd が `12:39:07.844949` に停止を開始したことを確認した。これは canonical start の約4.41秒後であり、この actual の pre-stop snapshot と競合して production runtime directory と lock を消失させた原因である。同じ PTY から `systemctl start` が `12:39:08.573318` に発行された。
- 最初の再起動は `/run/ullm` が未生成のまま mount namespace を構成しようとして `status=226/NAMESPACE` で失敗し、systemd の自動再起動が `12:39:19.118185` に scheduling された。次の起動は成功し、`/run/ullm/r9700.lock` は canonical end の約1.075秒後、`12:39:22.541399194` に再作成された。
- 上記の stop/start はこの actual の service 操作ではない。actual evidence が `systemctl_stop/start=0` と `service_touched=false` を固定している。
- 実行後の正式 health gate は PASS した。`ullm-openai.service` は active/running、worker は実行中、R9700 identity と AMD/KFD owner は一致し、lock は busy、production の served manifest / worker / package manifest / package content の4 hash は一致した。
- container namespace の formal health は Docker 9、docker exec 6、curl 6（endpoint 5）で、gateway healthz、readyz、models、OpenWebUI health の4 endpoint はすべて HTTP 200 だった。
- 実行後の service epoch は main PID `1722227`、worker PID `1722613`、`NRestarts=1` だった。`ExecMainStartTimestamp=2026-07-15 12:39:19 JST`、`Result=success`、`ExecMainCode=0`、`ExecMainStatus=0`、active/running だった。preflight 時点からの epoch 変更は並行する別 PTY の stop/start とその後の自動再起動によるもので、この actual 自身は service を操作していない。

## 次の行動

- この失敗証跡をそのまま保持し、profile diagnostic actual は再実行しない。
- 次回の新しい single-use authorization を作る場合は、実行直前まで production service が保持する `/run/ullm/r9700.lock` の存在と inode を binding し、並行 maintenance/probe との排他を operator contract に含める。

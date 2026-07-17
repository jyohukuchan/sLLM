# AQ4 Phase 3c: RuntimeDirectoryPreserve 修正と再実行 v0.1

## 前回の要点

- 2026-07-17 の最初の Phase 3c は、本番 gateway が `/run/ullm/r9700.lock` を保持していたため nonblocking `flock` が失敗し、GPU trace を起動せずに終了した。
- 同日の service-stop window では `ullm-openai.service` の stop/start 自体と復旧確認は成功したが、`RuntimeDirectory=ullm` と `RuntimeDirectoryPreserve=no` により stop 直後に `/run/ullm` ごと lock が削除された。既存 regular file だけを使う lock 契約を守ったため、trace は未起動だった。
- H5（GPU kernel 固有バグ）と実負荷下の H9（ハードウェア要因）は、いずれも数値測定がないため判定不能のままである。07/16 に停止した P3 harness の lock/root/artifact/environment には触れない。

## 今回の変更点

### systemd 設定の調査

- live unit は `/etc/systemd/system/ullm-openai.service` で、`RuntimeDirectory=ullm`、`RuntimeDirectoryMode=0750`、`RuntimeDirectoryPreserve=no`（変更前）だった。既存 drop-in は `/etc/systemd/system/ullm-openai.service.d/10-served-model.conf` だけであり、これは変更していない。
- lock の作成は tmpfiles.d、`ExecStartPre`、または別 service ではない。gateway worker の `_acquire_singleton_lock()` が起動時に `O_RDWR|O_CREAT|O_CLOEXEC|O_NOFOLLOW`（`O_EXCL` なし）で `ULLM_GPU_LOCK_FILE=/run/ullm/r9700.lock` を開き、regular file・owner/mode を検査して nonblocking flock を取得する。
- したがって `RuntimeDirectoryPreserve=yes` により残った mode `0600`・owner `homelab1` の regular lock file は、次回 worker 起動時に冪等に開かれる。既存 file があること自体はエラー条件ではなく、実際の排他は flock によって決まる。

### 適用した最小変更

- 新規 drop-in `/etc/systemd/system/ullm-openai.service.d/20-runtime-directory-preserve.conf` を root:root / `0644` で追加した。内容は次の1点だけである。

```ini
[Service]
RuntimeDirectoryPreserve=yes
```

- `systemctl daemon-reload` は成功した。この操作では service の stop/restart を行っていない。
- reload 後の読み取り専用確認で `RuntimeDirectory=ullm`、`RuntimeDirectoryPreserve=yes`、drop-in paths に `10-served-model.conf` と新規 `20-runtime-directory-preserve.conf` が並び、`systemd-analyze verify /etc/systemd/system/ullm-openai.service` は成功した。

### window 前 baseline

- service は `active/running`、MainPID `889726`、`NRestarts=0`、`ExecMainStartTimestamp=Fri 2026-07-17 11:58:09 JST` のままであり、daemon-reload 前後で process の停止・再起動はない。
- active manifest SHA-256 は `feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44`。lock は regular file、mode `0600`、uid/gid `1000:1000`、device/inode `26:889811` だった。

### Phase 3c service-stop window v0.3 の実行結果

- この window は一回だけ実行した。trace や guard の再試行、追加の stop/start はしていない。07/16 に停止した P3 harness の lock/root/artifact/environment と V620 には一切触れていない。
- 「systemctl stop ullm-openai.service」は 2026-07-17T03:26:43+00:00 に一回だけ成功した。直後の 2026-07-17T03:26:43.489316+00:00 に、既存 lock を作成せず「O_RDWR|O_NOFOLLOW|O_CLOEXEC」と「LOCK_EX|LOCK_NB」で検査した。
- lock は regular file のまま存在し、mode 0600、device/inode 26:889811 が停止前と完全に同一だった。probe は取得・解放に成功し、create_flag_used=false である。したがって、今回の RuntimeDirectoryPreserve=yes は実効し、元の systemd lifecycle 欠陥は修正・実証された。

| UTC timestamp | step | result |
| --- | --- | --- |
| 2026-07-17T03:26:43+00:00 | systemctl stop invoked / returned | 成功（各一回） |
| 2026-07-17T03:26:43.489316+00:00 | post-stop existing-lock probe | valid、nonblocking flock 取得成功、同一 inode |
| 2026-07-17T03:26:43+00:00 | HIP architecture guard | 成功。filtered HIP ordinal 0 は gfx1201、AMD Radeon Graphics、PCI BDF 0000:47:00.0 |
| 2026-07-17T03:26:43+00:00 | 同一 BDF の ASIC cross-check | 未完了。runuser の default PATH が amd-smi を解決できず失敗（通常の実行環境では /opt/rocm/bin/amd-smi） |
| 2026-07-17T03:26:43+00:00 | systemctl start invoked | guard 契約不成立を受け、trace 前に一回だけ復旧開始 |
| 2026-07-17T03:26:44+00:00 | systemctl start returned | 成功 |
| 2026-07-17T03:28:34.040905+00:00 | post-restore full verification | valid |

HIP の gfx1201 判定自体は成功しており、誤った GPU を選んだことによる失敗ではない。しかし、runbook が要求する「HIP で返った同一 BDF を amd-smi でも gfx1201 / 0x7551 / non-empty name と照合する」という guard 全体は完了していない。V620 を問い合わせる command は発行していない。single-use 契約に従い、この command-path 欠陥を window 内で直したり再実行したりせず、直ちに復旧した。

### trace、telemetry、段階別比較

- full guard が成立しなかったため、GPU health telemetry と GPU trace は開始していない。retry は false、comparison record は 0 である。
- 今回の H9 telemetry は「guard failure 後は採取しない」という手順に従って未取得である。前回の non-trace telemetry では、ECC/UMC ECC は 0、bad page はなし、温度は 36--37°C、throttle の明白な兆候はなく、clock は idle 状態の変化だけだった。今回の実負荷下 telemetry との比較はできない。

| stage | relative L2 | cosine | max abs | threshold 判定 |
| --- | --- | --- | --- | --- |
| qkv_dequant_row_scale | 未測定 | 未測定 | 未測定 | 判定不能 |
| z_dequant_row_scale | 未測定 | 未測定 | 未測定 | 判定不能 |
| recurrent_gate | 未測定 | 未測定 | 未測定 | 判定不能 |
| recurrent_beta | 未測定 | 未測定 | 未測定 | 判定不能 |
| recurrent_state_after | 未測定 | 未測定 | 未測定 | 判定不能 |
| recurrent_output | 未測定 | 未測定 | 未測定 | 判定不能 |
| attention_residual | 未測定 | 未測定 | 未測定 | 判定不能 |
| post_norm | 未測定 | 未測定 | 未測定 | 判定不能 |
| mlp_activation | 未測定 | 未測定 | 未測定 | 判定不能 |
| layer_output | 未測定 | 未測定 | 未測定 | 判定不能 |

従って最初に有意な乖離を示した stage は特定できない。H5（GPU kernel 固有バグ）は未判定、H9（ハードウェア要因）は実負荷下で未判定のままである。どちらも支持・否定の結論は出していない。

### service 復旧確認と evidence

- 復旧後の service は active/running、MainPID 995096、worker PID 995202、NRestarts=0（停止前から増加なし）である。active manifest SHA-256 は期待値 feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44 と一致した。
- healthz={"status":"ok"}、readyz={"status":"ready"} を確認した。新 worker は /dev/kfd と render node FD を保持し、R9700 BDF だけに絞った post-restore amd-smi process は PID 995202 を owner として示した。KFD VRAM は 7351832576 bytes だった。
- lock は復旧後も regular file、mode 0600、uid/gid 1000:1000、device/inode 26:889811 であり、新 gateway PID 995096 が保持している。stop/start をまたぐ lock の存続は post-restore evidence でも true になった。
- service-control の停止から start command 成功までの間隔は約 1 秒（03:26:43→03:26:44）。health/ready/worker/GPU/KFD/manifest を含む完全な復旧確認は stop 開始の 111.040905 秒後に完了した。guard 失敗 branch では連続的な ready polling をしていないため、後者は「確認完了時刻」であって正確な ready 到達時刻ではない。
- evidence root は benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-phase3c-gpu-stage-trace-v0.1/service-stop-window-v0.3-runtime-directory-preserve/。主要な記録は service-window-pre-stop.json、service-window-lock-after-stop.json、r9700-hip-device-guard.json、r9700-amd-smi-identity.stderr、service-window-post-restore.json、service-window-final-summary.json、service-window-core-evidence-v2.sha256 である。

### 実行した検証

- systemctl cat ullm-openai.service、systemctl show ullm-openai.service、systemd-analyze verify /etc/systemd/system/ullm-openai.service: 新規 drop-in の反映と unit 妥当性を確認。
- systemctl daemon-reload: 成功。直後に service stop/restart はしていない。
- g++ による tools/query-hip-device-identity.cpp の host-only build、cargo build --release -p ullm-engine --bin ullm-aq4-differential-trace --bin ullm-aq4-layer0-family-isolation、python3 tools/verify-aq4-layer0-package-embedding-fixture.py: すべて成功（fixture は {"cases":3,"status":"valid"}）。
- window driver: 一回の systemctl stop、既存 lock の no-create probe、R9700 HIP guard、同一 BDF への ASIC cross-check、guard failure による trace skip、一回の systemctl start、post-restore snapshot。ASIC cross-check だけが runuser: failed to execute amd-smi: No such file or directory で失敗した。

## 次の行動

- この service window は消費済みであり、service は正常復旧済みである。今回の command-path 欠陥をこの turn で修正・再実行しない。
- 将来 Phase 3c を行うには、別途明示承認された新しい window が必要である。その前に runuser 下でも絶対 path /opt/rocm/bin/amd-smi を使うなど、ASIC cross-check の command resolution を CPU-only/read-only preflight で確認する必要がある。
- H5/H9 の結論、Phase 4 以降の fix 実装、P3 harness への操作には進まない。

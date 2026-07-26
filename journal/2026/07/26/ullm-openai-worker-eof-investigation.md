# `ullm-openai.service` worker stdout EOF の調査記録

## 前回の要点

- 依頼AYの隔離窓は 2026-07-26 19:12:23--19:59:57 JST に閉じた。
  窓外の 20:17:27 JST に gateway が `unexpected worker stdout EOF` を記録し、
  20:19:35 JST に start-limit を解除して復旧したため、当初は worker の自然停止と
  区別できていなかった。
- 調査対象の現行 manifest SHA-256 は
  `c57a2b6c5827b8ddd102560b3f5efd879711705cf4d8a36f4d7872821d05fca4`、
  現行 worker は `/opt/ullm/.../ullm-aq4-worker`
  (`1f93f21543af777adb0f00cc35d6857d0af432657ed74e7723636ace9dfca69b`)
  である。

## 今回の変更点

### 20:17:27 JST の結論

これは worker が独立に落ちたことを示す事象ではなく、**明示的な service 停止中の
teardown EOF** である。journal の順序は次の通りである。

| 時刻 (JST) | 記録された事実 |
| --- | --- |
| 20:12:56.964--20:12:58.339 | 最後の request (`req-721dec…`) は prompt 39 / completion 64 を `length` で正常 release した。 |
| 20:17:27.056549 | `homelab1` による `sudo systemctl stop ullm-openai.service` が journal に記録された。発行元の端末・agent は journal からは特定できない。 |
| 20:17:27.063328 | systemd が `Stopping ullm-openai.service` を記録した。 |
| 20:17:27.086505 | gateway が `Shutting down` を記録した。 |
| 20:17:27.106464 | `worker_fatal` / `unexpected worker stdout EOF`。`request_id` と `completion_id` はともに `null`。停止開始から 43.121 ms 後。 |
| 20:17:27.272316 | systemd は `Deactivated successfully`、続いて `Stopped` を記録した。該当 instance は CPU 19.767 s、memory peak 601.2 MiB、swap peak 0 B。 |

従って、EOF の原因として確認できるのは **service stop 要求**である。worker
子プロセスの終了コード・終了シグナルは残っておらず、例えば `SIGTERM` だったと
断定する証拠はない。gateway の worker stderr は
`services/openai-gateway/src/ullm_openai_gateway/worker.py` で pipe を drain して
捨てており、worker 専用 systemd unit / journal も存在しないためである。

`KillMode=control-group`、`KillSignal=15` の unit 設定と、worker が gateway の
同一 cgroup の子プロセスであることは確認した。gateway 側の `_stopping` はその
自身の shutdown coroutine で初めてセットされるため、control-group stop による
pipe close がそれより先に観測されると `worker_fatal` と記録される。このログ名は
意図的停止時には誤解を招くが、今回のログから worker の異常終了を結論付けることは
できない。

### gateway / start-limit の挙動

- unit は `Restart=on-failure`、`RestartSec=10s`、`StartLimitBurst=3`、
  `StartLimitIntervalSec=900` である。
- 今回の explicit stop は `Deactivated successfully` で終了したため、
  `Restart=on-failure` による自動再始動は行われなかった。
- 20:07:51--20:12:43 には別セッションの
  `tools/promote-served-model.py --semantic-self-test` と rollback が実行され、
  `restart_service()` の設計通りに 20:08:11、20:09:06、20:11:56、20:12:42 の
  service restart を発生させている。各 EOF は `Stopping` の 35--45 ms 後である。
- これに先立つ 20:05:57 の `systemctl stop` と 20:06:13 の `systemctl start` も
  journal に残る。これは AY が記録した 19:12--19:59 の隔離窓とは別の service 操作で、
  発行元の端末・agent は同じく特定できない。
- 20:19:18 の `start-limit-hit` は、20:17 の stop 後に行われた明示 `start` が
  直近 15 分の start-rate 上限に当たったものである。20:19:34 の
  `reset-failed` + `start` により 20:19:35 に復旧した。worker 自然故障に対する
  systemd 自動再始動が失敗した、という記録ではない。
- 復旧後は 20:20:01.159 に prompt 39 / max completion 64 を admit し、
  20:20:02.556 に completion 64 を `length` で正常 release した
  （admit-to-release 1.395409717 s）。

### 過去 journal の再現性確認

全保持 journal の `unexpected worker stdout EOF` は 245 件だった。

- 230 件は直前 10 秒以内に同一 service の `Stopping` があり、service teardown
  中の EOF である。今回の 20:17:27 もこの群に属する。
- 15 件は 2026-07-11 と 2026-07-13 にのみあり、`Stopping` なし、gateway main
  process の `status=1/FAILURE`、`Restart=on-failure` による restart job という
  本当の worker failure だった。Jul 13 の例では active request 中に EOF が発生し、
  StartLimitBurst=3 に到達している。
- この 15 件は現在とは異なる worker / manifest である。Jul 13 は
  `target/reasoning-v2/release/ullm-aq4-worker`
  (`177f3106414efc7cc4b08fa2d87bed6e147d4188e0a290f43b7a1ac591fae48d`) と
  manifest `e9875a…` であり、現行 `/opt/ullm` worker / manifest とは同一視できない。
  worker stderr と exit status が捨てられていたため、その旧 failure の内因は未特定である。
- Jul 14--Jul 26 20:17:27 までに、停止操作と対応しない同型 EOF はない。従って
  **現行 artifact については、本件を含む未説明の自然 EOF の再発は journal 上では
  確認されなかった**。旧 build に true failure の履歴があること自体は残る。

### OOM / GPU / 同時負荷の照合

- 20:17 の前後、および 2026-07-20--26 の `journalctl -k` と `dmesg` に、OOM killer、
  `Killed process`、memory cgroup OOM、AMD GPU ring timeout、page fault、reset の
  記録はなかった。したがって本件に OOM killer または GPU reset が関与した証拠はない。
- 20:12:42 に `kfd_process_wq_release [amdgpu] hogged CPU` が 1 件あるが、これは
  rollback が明示停止を開始した 67 ms 後の teardown 中であり、20:17 の EOF に
  対応する GPU reset/error ではない。20:07:59 の `failed to disable PTL` も別時刻で、
  reset / page fault の記録ではない。
- F32 参照コーパス parent (PID 1595579) は 14:43:30 JST に開始され、8 child は
  すべて 20:17 より前に起動し、その後も継続している。設定は
  `--processes 8 --threads 8 --nice 10` である。調査時点の 8 child RSS 合計は
  3.37 GiB だったが、これは 20:17 時点の snapshot ではない。
- 調査時点の host は available memory 73 GiB、現行 service cgroup の
  `memory.events` は `oom=0` / `oom_kill=0` だった。これは復旧後の値であり、
  過去 instance の counter ではない。上記 kernel journal と併せ、20:17 のメモリ枯渇を
  支持する証拠はない。過去時点の全 process RSS / build 状態は保存されておらず未確認である。
- 20:07:59--20:08:02 には decode-attention 用 `rocprofv3` が記録されるが、20:17 の
  瞬間には journal に GPU workload 起動・kernel fault はない。既に起動済みだった
  無記録の負荷の有無は journal だけでは未確認とする。

### bridge endpoint の確認

`172.20.0.1:8000` へホスト shell から接続できないのは**意図的な設定**である。

- `ullm-openai-firewall.service` は active で、`inet ullm_openai` の実効 rule は
  `iifname != "br-79bb7cfca31c"` の `172.20.0.1:{8000,8001}` を drop する。
  host の route は `local … dev lo` なのでこの drop 対象である。
- `deploy/nftables/ullm-openai.nft` と `deploy/README.md` は、host loopback / LAN access
  を意図的に拒否し、Docker network から health check する設計を明記している。
- 実測でも host `curl --max-time 3 http://172.20.0.1:8000/health` は 2.002 s で timeout、
  `open-webui` container (`172.20.0.2`) からの `/readyz` は HTTP 200
  `{"status":"ready"}` だった。iptables / nftables は変更していない。

## 次の行動

1. 本番 unit と active manifest は変更しない。今回確認した是正候補は unit の停止方式であり、
   maintenance window で shutdown EOF を `worker_fatal` と誤記録しない必要が生じた場合は、
   `KillMode=mixed` を unit 変更候補として検証する。最初に gateway だけを SIGTERM して
   既存の graceful worker shutdown を走らせ、TimeoutStop 後だけ cgroup 全体を kill する
   方式である。これは**提案のみ**で、今回適用しない。
2. 将来の真の worker failure を特定可能にするには、gateway の worker stderr tail と
   `Process.returncode` を fatal event に残す軽量な観測を別変更として追加する。現状は
   stderr drain が破棄するため、旧 15 件のような failure は原因を遡れない。
3. service 操作を伴う promotion / rollback は `StartLimitBurst=3/15min` を消費する。
   この上限を跨ぐ batch の前には、操作回数と `restart_service()` が記録する
   start-limit recovery を確認する。unit の値を変更する場合も別途提案・承認対象とする。
4. 調査終了時の確認値: `ullm-openai.service` は `active/running`、`NRestarts=0`、
   manifest SHA-256 は上記 `c57a2b6…05fca4` である。

## Evidence

- `journalctl -u ullm-openai.service --since '2026-07-26 20:12:35' --until '2026-07-26 20:19:40' -o short-iso-precise`
- `journalctl --since '2026-07-26 20:17:20' --until '2026-07-26 20:17:40' -o short-iso-precise`
- `journalctl -k` / `dmesg` の OOM・amdgpu error 検索
- `systemctl cat/show ullm-openai.service`, `systemctl cat ullm-openai-firewall.service`
- `deploy/nftables/ullm-openai.nft`, `deploy/README.md`, `services/openai-gateway/src/ullm_openai_gateway/worker.py`
- `docker network inspect open-webui-network`, host curl と `docker exec open-webui` `/readyz` probe

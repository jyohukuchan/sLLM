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

## 次の行動

- 新しい条件を確認する追加 stop/start は行わない。今回の Phase 3c service-stop window 一回だけで、stop 後の既存 lock の存続、nonblocking flock、R9700 guard、H9 telemetry、GPU trace、30 record 比較、service 復旧を順に実行する。
- lock が stop 後も regular file として残らなければ trace へ進まず、直ちに service を一回だけ start して復旧確認する。trace 内の失敗も同一 window で再試行しない。

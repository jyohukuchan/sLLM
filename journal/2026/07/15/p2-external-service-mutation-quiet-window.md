# P2 external service mutation quiet window

## 前回の要点

maintenance と diagnostic の実行が並行して production service epoch を変更しないことを確認するため、service、worker、GPU owner、lock、端末と service 操作 process を読み取り専用で監視する必要があった。

## 今回の変更点

主監視を 120 秒、5 秒間隔、25 samples で行った。開始時点の `ullm-openai.service` は MainPID `1722227`、worker PID `1722613`、NRestarts `1`、ExecMainStartTimestamp `Wed 2026-07-15 12:39:19 JST`、active/running だった。lock inode `762398` と holder PID `1722227`、AMD SMI/KFD owner PID `1722613` は全 samples で不変だった。NRestarts `1` は window 開始前の既存値であり、window 内では増加していない。

formal container health は開始・終了とも container identity、gateway `/healthz`、`/readyz`、認証済み `/v1/models`、OpenWebUI `/health` が一致して HTTP 200 だった。

最初の pts/5 抽出は監視コマンド自身の文字列を拾ったため証拠から除外し、`ps -t pts/5` を用いた端末単位の追加監視を別の 120 秒、5 秒間隔、25 samples で行った。pts/5 process は全 samples で 0、外部 `systemctl`/maintenance/launcher process の観測も 0、service epoch と lock inode は不変だった。両 window とも violation は 0 で、判定は GO である。

機械可読 evidence は `benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-quiet-window-v1/quiet-window.json` に保存した。

## 次の行動

後続 manifest は quiet-window evidence の SHA-256、`status=go`、service epoch、25 samples、formal health start/end equality、terminal pts/5 と external process observations 0 を検証してから使用する。

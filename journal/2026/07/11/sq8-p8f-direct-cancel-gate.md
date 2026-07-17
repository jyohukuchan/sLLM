# SQ8 P8-F direct cancellation gate

日付: 2026-07-11

状態: 4段階cancelと4回の回復requestをR9700実機で完了

## 前回の要点

OpenWebUI Stopは実button操作でformal gate済みである。残るdirect HTTPの4位相は、journald followerではなく低遅延AF_UNIX lifecycle observerで厳密にtriggerし、authoritative journalと後からbyte一致させる必要があった。

## 今回の変更点

- commit `9539d98`で`uLLM-project/tools/run-sq8-direct-cancel-gate.py`を追加した。位相は`after_started_before_progress`、`prefill_after_128`、`prefill_after_2048`、`decode_after_first_content`の固定順序である。
- 前半3件は固定`exact-p3584`、decodeは`exact-p0032`を使う。各targetは`client_disconnect`と`reset_complete=true`で5秒以内にreleaseし、その後に`exact-p0032` / `max_tokens=2`の回復requestを完走させる。
- 実HTTP clientのexplicit closeとfirst-content auto-closeを使い、raw request/response chunk/SSE boundaryを保持する。observerはkernel PID/UID/GIDを検査し、journalはboot/unit/PID/cursor/payloadを厳密に照合する。
- 独立review後にrelease duration算術、回復prefill完了後のfirst token、公開直前のraw/JSON再ハッシュを追加した。専用33 tests、関連5 suites 125 tests、Ruff check/format、strict mypy、`py_compile`、`git diff --check`は合格した。
- preflightは`/etc/ullm/openai-api-key`の0640をgroup-readableとして正しく拒否した。GPU requestは発行されず、失敗bundleも残らなかった。0600の一時snapshotで再実行し、終了後に削除した。

## R9700実機結果

- atomic bundleは`/home/homelab1/datapool/sq8-direct-cancel-formal-20260711-194417`。`summary.json` SHA-256は`60e6f49158ed2da54a54cc072c0b257547f668db71ffeca578486ed81c4dc646`である。
- 4 target + 4 recoveryの8 requestは全件成功し、maximum activeは1だった。observer/journal/correlationは55/55/55 recordsで完全一致した。
- `after_started_before_progress`: progress 0、completion 0、cancel→release 149.386 ms。
- `prefill_after_128`: progressは`[128]`だけ、completion 0、cancel→release 218.734 ms。
- `prefill_after_2048`: progressは128から2048まで16境界でovershootなし、completion 0、cancel→release 1.297 s。
- `decode_after_first_content`: progress `[32]`、completion 1、最初の実本文chunkでcloseし、cancel→release 32.069 ms。
- 4回の回復は全て`outcome=length`、completion 2、`reset_complete=true`だった。gateway PID `712139`、worker PID `712239`、`NRestarts=0`、OpenWebUI health=healthyのままである。observer socket、一時container、一時API keyも残っていない。

## 次の行動

OpenWebUI post-header worker failure gateの独立reviewを閉じ、実際のSIGKILLと1回だけのsystemd再起動、失敗表示、同一chat回復を実機で確認する。

# P2 resident one-case preflight NO-GO

## 前回の要点

base one-case actual は、HEAD `949d5c5`、maintenance harness `426290f`、再固定済み ready artifact を使い、同一 PTY の sudo cache と全 pre-stop gate が合格した場合だけ一度実行する契約だった。失敗時の再試行と profile 実行は禁止されている。

## 今回の変更点

ready artifact の checksum と mode 0444、harness/launcher の固定内容、対象パスの clean、maintenance/launcher/runner の三つの出力先が未使用であることを確認した。RAM available は約 80.4 GB、repository filesystem の空きは約 2.58 TB、datapool の空きは約 9.20 TB だった。

固定 harness と同じ `capture_running` で pre-stop gate を検査したところ、`http://172.20.0.1:8000/readyz` が 5 秒で timeout した。この時点で NO-GO とし、sudo prime、service stop、`--confirm-one-case`、actual launcher は一度も実行していない。

読み取り専用診断では、gateway は `172.20.0.1:8000` で LISTEN していた。host namespace からは `127.0.0.1:8000` が connection refused、`172.20.0.1:8000` の `/healthz` と `/readyz` が timeout した。一方、従来の有効経路である `open-webui` container `172.20.0.2/16` からは、同じ gateway の `/healthz` と `/readyz` が HTTP 200、認証付き `/v1/models` が HTTP 200 かつ model count 1 だった。OpenWebUI 自身の health も HTTP 200 だった。

原因は endpoint 障害や負荷ではなく、host-side harness probe と Docker network namespace の経路不一致である。診断後も `ullm-openai.service` は active/running、MainPID `3090367`、worker PID `3090924`、NRestarts `0`、対象 GPU の単一 owner と lock 保持を維持した。production の復旧操作は不要である。

構造化 evidence は `benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-preflight-no-go-v1/diagnosis.json` に保存した。

## 次の行動

actual の再実行とは分けて、maintenance harness の gateway health probe を OpenWebUI container network namespace の固定経路へ変更し、機密情報非露出、container identity/network binding、失敗時 fail-closed を QA する。その後、harness trust と ready artifact を再固定してから、新しい明示承認のもとで one-case を一度だけ実行する。

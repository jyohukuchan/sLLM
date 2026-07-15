# SQ8 resident capture failure receipt修正

## 前回の要点

独立監査 receipt `189ada29c116515782b8f7b153302b61fc3b316e0f3cefd3595db5f81fe38722` は、capture toolがworker stderrの完全性・終了信号・timeoutを固定エラー契約へ構造化せず、drain未完了をfail closedできないとしてNO-GOと判定した。

## 今回の変更点

- `capture-aq4-resident-executor-record.py` のworker stderr envelopeへ `complete` と `stream_error` を追加し、drain thread未完了・stream exceptionを不完全として扱うようにした。
- capture error envelopeを8キー固定 (`schema_version,status,stage,reason,timed_out,worker_returncode,worker_signal,worker_stderr`) にし、workerの負値returncodeから正のsignalを保持するようにした。
- request/shutdown timeoutを明示し、worker terminateからkillへの段階的cleanupとreapを行うようにした。
- stderrのnon-JSON、invalid UTF-8、secret、32 KiB超の入力を実subprocessで検証する専用テストと、JSON success pathの回帰テストを追加した。
- 監査で検出されたruff F841/F601と専用test EOF blankを修正した。

## 検証

- `pytest -q tests/test_capture_aq4_resident_executor_record.py` : 10 passed
- `python3 -m py_compile tools/capture-aq4-resident-executor-record.py tests/test_capture_aq4_resident_executor_record.py` : passed
- fake worker実行でsignal、timeout、non-JSON、invalid UTF-8、secret、40 KiB stderrを確認済み。

## 次の行動

outer runner側でこの固定envelopeを構造化保存し、failure receipt/SHA256SUMSへ結び付ける。

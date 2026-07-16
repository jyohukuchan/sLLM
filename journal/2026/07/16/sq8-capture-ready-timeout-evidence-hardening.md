# SQ8 capture ready/timeout evidence hardening

## 対象と制約

- source commit: `6d12f67e297571485c07d2b87ef5f8727a9652f2`
- 隔離 worktree: `/tmp/ullm-sq8-timeout-evidence-6d12f67e`
- GPU、service、sudo、新しい authorization、actual execution は実行していない。
- SQ8 actual request は 128 prompt tokens と 2 completion tokens のまま変更していない。

## RCA

従来の capture は worker を起動した直後に generate request を stdin へ書き、同時に 240 秒の deadline を開始していた。worker は resident model の load と backend operation trace の出力を完了して ready を flush してから command reader を開始するため、model load 時間が request 実行時間として消費されていた。外側の 300 秒上限より内側の 240 秒上限が先に発火し、ready 直後または request 開始前に request timeout として終了し得る構造だった。

## 変更

- worker 起動後は strict ready identity を最大 900 秒待ち、ready 検証後にだけ generate request を書く。
- request を flush した時点から既存の 240 秒 request deadline を開始する。
- shutdown は 30 秒、worker terminate/reap と pipe drain を明示的に上限化し、runner の outer timeout は 1350 秒とした。ready + request + shutdown + terminate/reap + bounded drain + 60 秒 packaging margin の不等式をテストで固定した。
- cleanup の最後に残っていた無期限 `wait()` を除去した。
- failure stage を `ready_timeout`、`ready_protocol`、`request_timeout`、`request_protocol`、`shutdown_timeout`、`capture_outer_timeout` として区別した。
- Gate actual request と prepared/actual/failure receipt に ready=900、request=240、shutdown=30、outer=1350 の契約を固定し、runner と receipt writer で完全一致を検証する。
- Gate の trusted_components を4要素の exact schema とし、approved tools root、canonical path、regular/non-symlink、nlink=1、SHA-256を実行前に検証する。capture、receipt writer、served-model generator は検証済みパスだけを実行時に使用し、maintenance/success/failure evidenceへ同じpath+SHAを束縛する。
- lifecycle は last_event、events_truncated、ready先頭、request_sent offset順序を検証し、ready_protocol で request_sent を禁止する。runner のSQ8 telemetryはキー、整数非bool、範囲、閾値を厳密に検証する。
- stderr evidence を v2 に更新し、raw total bytes/SHA-256、各 16 KiB の head/tail、record count、schema counts、最大 512 records/4 MiB、許可フィールドだけの last complete record を保持する。
- stdout lifecycle evidence を最大 64 events に制限し、process start からの monotonic offset、request ID 一致、token index/count だけを保持する。token ID、prompt、生成内容は保持しない。
- failure envelope v4 は request ID、timeouts、worker return code/signal、stderr/lifecycle evidence を結び、runner で shape、値、stage/terminal 対応、秘密情報、改ざんを fail closed で検証する。

## 検証

以下を jobs=1 で実行した。

```text
python3 -m compileall -q tools tests
PYTHONPATH=. pytest -q \
  tests/test_capture_aq4_resident_executor_record.py \
  tests/test_qwen35_aq4_sq8_overlay_promotion_receipt.py \
  tests/test_prepare_qwen35_aq4_sq8_overlay_gpu_promotion.py \
  tests/test_run_qwen35_aq4_sq8_overlay_gpu_promotion.py
```

結果は `152 passed`。delayed-ready success、ready timeout before request、request timeout、outer timeout margin、partial token lifecycle、stderr tail truncation、secret rejection、Gate component/path/symlink/nlink/unknown/missing tamper、request/timeouts/lifecycle/schema-count/receipt/telemetry tamper rejectionを含む。

## lineage への影響

この変更は source commit/tree/archive を変更するため、以前の independent audit、authorization lineage、prepared/actual/failure receipt、request ID、output、sentinel は新しい source identity を authorization しない。actual execution の前に、新しい materialization、独立 runtime audit、新しい固定 request ID、新しい output/sentinel、新しい authorization lineage が必要である。

# T1 linear-attn request state owner

## 前回の要点

- full mixed layer kind inventoryで、Qwen3.5-9B packageは32層連続、self-attn 8層、linear-attn 24層だと確認した。
- 次のT1 blockerはlinear-attnのrecurrent stateとConv1d historyをrequestごとに分けることだった。

## 今回の変更点

- `PackageLinearAttnResidentStepBatchLayer` を追加した。
- `RequestId` からlinear-attn resident state slotへ解決するownerにした。
- request slot index helperを追加し、空request listと重複request idをrejectするunit testを追加した。
- 結果レポートを `uLLM-project/benchmarks/results/2026-07-09/package-batch-throughput/phase-t1-linear-attn-request-state-owner-v1.md` に保存した。

## 次の行動

1. full mixed-attention runnerのlayer enumにこのownerを接続する。
2. small B=2 smokeでfull mixed pathのreal prefill/decode flagを立てる。
3. その後、weights共有とthroughput改善へ進む。

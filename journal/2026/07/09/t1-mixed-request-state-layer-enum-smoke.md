# T1 mixed request-state layer enum smoke

## 前回の要点

- linear-attention request state ownerは実package smokeでrequestごとのstate分離を確認済み。
- self-attention request state ownerはunit testでrequest-id dispatch境界を追加済み。
- 次は両方を同じmixed layer enumへ接続する必要があった。

## 今回の変更点

- `PackageMixedRequestStateLayer` を追加した。
- `package-token-ids-mixed-request-state-smoke` を追加した。
- R9700で `layers=0,3`、`batch=2`、`prompt=2`、`generated=1` の小さいmixed dispatch smokeを実行し、`verified=true` を確認した。
- 結果は `uLLM-project/benchmarks/results/2026-07-09/package-batch-throughput/phase-t1-mixed-request-state-layer-enum-smoke-v1.md` に保存した。

## 次の行動

1. `manifest-all` へ広げ、full mixed layer orderで壊れないか確認する。
2. request slotごとのresident weight複製をやめ、shared resident weights + per-request state bufferへ寄せる。
3. full packageのreal prefill/decode/end-to-end throughput rowを作る。

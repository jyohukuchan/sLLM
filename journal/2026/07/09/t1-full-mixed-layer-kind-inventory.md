# T1 full mixed layer kind inventory

## 前回の要点

- self-attn stack real-batch small gridは通ったが、full packageにはlinear-attn層が24層ある。
- full package logical batch rowsはあるが、`prefill_real_batch=false` / `decode_real_batch=false` のためSQ throughput比較には使えない。
- 次はfull mixed-attention runnerの実装targetを曖昧にしない必要があった。

## 今回の変更点

- `package-layer-kind-inventory-smoke` を追加した。
- `manifest-all` aliasを追加し、manifestからsupported layer indexを昇順に抽出できるようにした。
- 実AQ4 packageで、layers `0..31` が連続し、self-attnが `3,7,11,15,19,23,27,31`、linear-attnが残り24層だと確認した。
- 結果レポートを `uLLM-project/benchmarks/results/2026-07-09/package-batch-throughput/phase-t1-full-mixed-layer-kind-inventory-v1.md` に保存した。

## 次の行動

1. manifest orderを使ってfull mixed-attention runnerを組む。
2. linear-attn層のper-request recurrent stateとConv1d history ownerを実装する。
3. full packageでreal request-batch prefill/decode/end-to-end rowを保存する。

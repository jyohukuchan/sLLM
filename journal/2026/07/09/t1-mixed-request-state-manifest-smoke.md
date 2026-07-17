# T1 mixed request-state manifest smoke

## 前回の要点

- `PackageMixedRequestStateLayer` は `layers=0,3` の小さいmixed guardで通った。
- full mixed-attention packageのmanifest order `0..31` が同じdispatch境界で通るかは未確認だった。

## 今回の変更点

- `package-token-ids-mixed-request-state-smoke` を `manifest-all` で実行した。
- R9700でAQ4 full packageの32層、linear-attention 24層、self-attention 8層を通し、`verified=true` を確認した。
- final top1 tokensは `44370,5446`。
- 結果は `uLLM-project/benchmarks/results/2026-07-09/package-batch-throughput/phase-t1-mixed-request-state-manifest-smoke-v1.md` に保存した。

## 次の行動

1. shared resident weights + per-request state/cache bufferへ寄せる。
2. full packageで `prefill_real_batch=true` / `decode_real_batch=true` のAQ4 baseline rowを作る。
3. 同じworkload gridへSQ FP8候補を接続する。

# T1 linear-attn request state smoke

## 前回の要点

- full mixed-attention packageのlayer orderは `0..31` で固定済み。
- `PackageLinearAttnResidentStepBatchLayer` は追加済みだったが、実package上での実行証拠はまだなかった。
- full mixed runnerへ進むには、linear-attention層のrecurrent stateとcausal Conv1d historyがrequestごとに分離されることを確認する必要があった。

## 今回の変更点

- `package-linear-attn-request-state-smoke` を追加した。
- R9700でAQ4 package layer `0` を `request_count=2`、`sequence_len=2` でinterleaved実行した。
- batch owner出力をrequestごとのserial resident layer referenceと比較し、`serial_reference_max_abs_diff=0` を確認した。
- 結果を `benchmarks/results/2026-07-09/package-batch-throughput/phase-t1-linear-attn-request-state-smoke-v1.md`、計画書、state freezeへ記録した。

## 次の行動

1. full mixed-attention runnerのlayer enumへself-attention resident step layerとlinear-attn request-state ownerを並べる。
2. 小さいB=2 / prompt=2 / generated=1でfull mixed path smokeを作る。
3. その後にweights共有とactual throughput改善へ進む。

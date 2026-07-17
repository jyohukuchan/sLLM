# T1 self-attn request state owner

## 前回の要点

- linear-attn request-state ownerは実package smokeまで通った。
- full mixed-attention runnerではself-attention層とlinear-attention層を同じrequest-id dispatch形へ揃える必要がある。
- 既存の `PackageSelfAttnResidentStepLayer` はsingle request向けにpaged KV cache、written_len、block tableを内部保持していた。

## 今回の変更点

- `PackageSelfAttnResidentStepBatchLayer` を追加した。
- `RequestId` からself-attn resident layer state slotへ解決し、requestごとのpaged KV cache、written_len、block tableを分離するownerにした。
- request slot helperをlinear/selfで共通化し、self-attn側の空request list、重複request id、順序維持のunit testを追加した。
- 結果を `benchmarks/results/2026-07-09/package-batch-throughput/phase-t1-self-attn-request-state-owner-v1.md`、計画書、state freezeへ記録した。

## 次の行動

1. full mixed-attention runnerのlayer enumにself-attn/linear-attn両方のrequest-state ownerを並べる。
2. 小さいB=2 / prompt=2 / generated=1でmanifest orderのfull mixed path smokeを作る。
3. その後にweights共有とactual throughput改善へ進む。

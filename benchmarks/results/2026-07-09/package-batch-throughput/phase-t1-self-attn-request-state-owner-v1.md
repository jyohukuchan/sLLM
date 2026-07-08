# T1 self-attn request state owner v1

## 前回の要点

- linear-attention側では `PackageLinearAttnResidentStepBatchLayer` を実package smokeまで通し、requestごとのrecurrent stateとConv1d history分離を確認した。
- full mixed-attention runnerではself-attention層とlinear-attention層をmanifest orderで同じrequest-id dispatch形へ並べる必要がある。
- 既存の `PackageSelfAttnResidentStepLayer` はsingle request向けにpaged KV cache、written_len、block tableを内部保持していた。

## 今回の変更点

- `PackageSelfAttnResidentStepBatchLayer` を追加した。
- このownerは `RequestId` からself-attn resident layer state slotへ解決する。
- 各slotは `PackageSelfAttnResidentStepLayer` を持ち、requestごとのpaged KV cache、written_len、block tableを分離する。
- `step_from_host_to_device`、`step_from_device_to_device`、`output_buffer`、`read_output`、component timing取得をrequest id付きで呼べる境界にした。
- request id slot index helperをlinear/selfで共通化し、self-attn側の空request listと重複request idを拒否するunit testを追加した。

## Validation

| check | result |
| --- | --- |
| self-attn request slot index rejects empty list | pass |
| self-attn request slot index rejects duplicate request id | pass |
| self-attn request slot index preserves request order | pass |
| linear-attn request slot index regressions | pass |

Command:

```text
cargo test -p ullm-engine request_slot_index -- --test-threads=1
cargo build -p ullm-engine
```

## 判断

- これはfull package throughput rowではない。
- これはまだshared-weight real batch kernelではなく、requestごとにresident stateを分離するためのrunner-side ownerである。
- self-attention側とlinear-attention側のrequest-id dispatch形が揃ったため、次の段階ではfull mixed layer enumへ両ownerを接続できる。

## 次の行動

1. full mixed-attention runnerのlayer enumに `PackageSelfAttnResidentStepBatchLayer` と `PackageLinearAttnResidentStepBatchLayer` を並べる。
2. 小さいB=2 / prompt=2 / generated=1で、manifest orderのfull mixed path smokeを作る。
3. その後、weights共有と実throughput改善に進む。

# T1 linear-attn request state owner v1

## 前回の要点

- `phase-t1-full-mixed-layer-kind-inventory-v1` で、full mixed-attention packageはlayers `0..31`、self-attention 8層、linear-attention 24層だと確認した。
- full package real-batch runnerでは、self-attentionのpaged KV stateだけでなく、linear-attentionのrecurrent stateとcausal Conv1d historyもrequestごとに分離する必要がある。
- 既存の `PackageLinearAttnResidentStepLayer` はsingle request向けにstateを内部保持していた。

## 今回の変更点

- `PackageLinearAttnResidentStepBatchLayer` を追加した。
- このownerは `RequestId` からlinear-attn resident layer state slotへ解決する。
- 各slotは `PackageLinearAttnResidentStepLayer` を持ち、requestごとのrecurrent stateとConv1d historyを分離する。
- `step_from_host_to_device`、`step_from_device_to_device`、`output_buffer`、`read_output`、component timing取得をrequest id付きで呼べる境界にした。
- request id slot index helperを追加し、空request listと重複request idを拒否するunit testを追加した。

## Validation

| check | result |
| --- | --- |
| request slot index rejects empty list | pass |
| request slot index rejects duplicate request id | pass |
| request slot index preserves request order | pass |

Command:

```text
cargo test -p ullm-engine linear_attn_request_slot_index -- --test-threads=1
```

## 判断

- これはfull package throughput rowではない。
- これはまだshared-weight real batch kernelではなく、requestごとにresident stateを分離するためのrunner-side ownerである。
- 次の段階では、このownerをmanifest order runnerのlinear-attention layer側に接続する。
- 性能最終形では、weightsをrequestごとに複製しないshared resident weight + per-request state bufferへ寄せる必要がある。

## 次の行動

1. full mixed-attention runnerのlayer enumにlinear-attn request-batch ownerを接続する。
2. self-attention層は既存のpaged KV request stateを使い、linear-attention層はこのowner経由でrequest stateを引く。
3. まず小さいB=2 / prompt=2 / generated=1で、`prefill_real_batch=true` と `decode_real_batch=true` のfull mixed path smokeを作る。
4. その後、weights共有と実際のthroughput改善に進む。

# T1 full mixed layer kind inventory v1

## 前回の要点

- `phase-t1-self-attn-stack-real-batch-small-grid-v1` では、manifest self-attention層 `3,7,11,15,19,23,27,31` だけをreal-batchで測った。
- ただしQwen3.5-9B full packageにはself-attention層とlinear-attention層が混在しているため、この結果はfull mixed-attention LM throughputではなかった。
- full package real-batch runnerを実装する前に、manifest上のlayer orderとkindを機械的に固定する必要があった。

## 今回の変更点

- `package-layer-kind-inventory-smoke` を追加した。
- `manifest-all` aliasを追加し、`.ullm.d` manifestからsupported layer indexを昇順に抽出できるようにした。
- `package-token-ids-logits-smoke`、`sq-fp8-token-ids-logits-smoke`、`package-token-ids-generate-smoke`、`package-batch-throughput-bench` も `manifest-all` を受け取れるようにした。
- R9700 AQ4 packageで、full mixed-attention targetが32層連続、self-attention 8層、linear-attention 24層であることを確認した。

## Command

```text
target/debug/ullm-engine package-layer-kind-inventory-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d manifest-all
```

## Inventory

| field | value |
| --- | --- |
| status | `ok` |
| schema version | `package-layer-kind-inventory-smoke-v0.1` |
| package | `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d` |
| layer count | 32 |
| layer indices | `0..31` |
| contiguous layer indices | true |
| self-attention count | 8 |
| linear-attention count | 24 |
| mixed attention | true |
| self-attention layers | `3,7,11,15,19,23,27,31` |
| linear-attention layers | `0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30` |
| verified | true |

Layer kind order:

```text
0:linear_attention
1:linear_attention
2:linear_attention
3:self_attention
4:linear_attention
5:linear_attention
6:linear_attention
7:self_attention
8:linear_attention
9:linear_attention
10:linear_attention
11:self_attention
12:linear_attention
13:linear_attention
14:linear_attention
15:self_attention
16:linear_attention
17:linear_attention
18:linear_attention
19:self_attention
20:linear_attention
21:linear_attention
22:linear_attention
23:self_attention
24:linear_attention
25:linear_attention
26:linear_attention
27:self_attention
28:linear_attention
29:linear_attention
30:linear_attention
31:self_attention
```

## 判断

- このrowはthroughput計測ではなく、T1 full mixed-attention runnerの実装targetを固定するinventoryである。
- `manifest-all` はfull package layer orderを扱うための入口として使える。
- ただし現時点のfull package throughput rowはまだlogical batchであり、SQ throughput比較へ昇格できない。
- 次の実装上の本体は、self-attentionのper-request paged KV stateに加えて、linear-attentionのper-request recurrent stateとcausal Conv1d historyを同じrequest-batch runnerで保持することである。

## 次の行動

1. `PackageLinearAttnResidentStepLayer` 相当のstateをrequestごとに持つbatch ownerを実装する。
2. manifest order `0..31` に従ってlinear-attention層とself-attention層を混在実行する。
3. full packageで `batching.mode=real`、`prefill_real_batch=true`、`decode_real_batch=true` のAQ4 baseline rowを保存する。
4. その後に同じschemaでFP8 SQ候補を比較する。

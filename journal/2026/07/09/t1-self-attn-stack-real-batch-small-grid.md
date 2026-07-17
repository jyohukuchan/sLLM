# T1 self-attn stack real-batch small grid

## 前回の要点

- T1 token-id model-loop bridgeはselected-layerで動いていたが、full packageにはself-attention層とlinear-attention層が混在していた。
- full mixed-attention runnerはまだなく、`all` をそのままfull layer orderとして扱うことはできなかった。

## 今回の変更点

- `package-token-ids-model-loop-smoke` と `sq-fp8-token-ids-model-loop-smoke` に `all-self-attn` / `manifest-self-attn` aliasを追加した。
- manifestのself-attention `q_norm` / `k_norm` passthrough tensor集合からlayer indexを抽出し、layers `3,7,11,15,19,23,27,31` の中間stackを実行できるようにした。
- R9700 AQ4で `batch=1/4/8`、`prompt=4`、`generated=1` を測定した。

## 実測

| batch | batching | prefill_real_batch | decode_real_batch | prefill tok/s | decode tok/s | end-to-end tok/s |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 1 | `hybrid` | false | false | 74.066673 | 70.654751 | 73.358179 |
| 4 | `real` | true | true | 73.537780 | 71.298348 | 73.078709 |
| 8 | `real` | true | true | 73.326934 | 71.010893 | 72.851718 |

## 次の行動

1. このrowはmanifest self-attention stackの中間guardとして扱う。
2. 次はlinear-attention層を含むfull mixed-attention package real-batch prefill/decode/end-to-end runnerへ進める。
3. SQ throughput decisionには、full mixed-attention runnerの `batch=1/4/8` AQ4/FP8 rowsが揃ってから使う。

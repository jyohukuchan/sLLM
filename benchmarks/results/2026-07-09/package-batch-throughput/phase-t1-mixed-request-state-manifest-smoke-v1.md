# T1 mixed request-state manifest smoke v1

## 前回の要点

- `PackageMixedRequestStateLayer` は `layers=0,3` の小さいlinear-attn to self-attn guardで動作した。
- ただし、full mixed-attention packageのmanifest order `0..31` が同じrequest-state dispatchで通るかは未確認だった。
- SQ throughput比較へ進むには、まずfull mixed layer orderが壊れないことを確認する必要があった。

## 今回の変更点

- `package-token-ids-mixed-request-state-smoke` を `manifest-all` で実行した。
- R9700上でAQ4 full packageの32層をmanifest order通りに通した。
- linear-attention 24層とself-attention 8層を同じrequest-id dispatch境界でinterleaved実行した。
- final RMSNormとlm_head top1 guardまで到達することを確認した。

## R9700 smoke

Command:

```text
target/debug/ullm-engine package-token-ids-mixed-request-state-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d 2 1048576 manifest-all len:2x2 1 1 1024 32 10000000 0
```

Result:

| field | value |
| --- | ---: |
| backend | `hip` |
| device index | 2 |
| layers | `0..31` |
| layer count | 32 |
| linear-attention layers | 24 |
| self-attention layers | 8 |
| requests | 2 |
| prompt tokens/request | 2 |
| generated tokens/request | 1 |
| batching mode | `request_state_interleaved` |
| throughput row | `false` |
| prefill real batch | `false` |
| decode real batch | `false` |
| per-request cache buffers | `true` |
| shared paged cache | `false` |
| prefill request counts | `2,2` |
| decode request counts | `2` |
| final top1 tokens | `44370,5446` |
| prefill total input tok/s | 37.834213 |
| decode total generated tok/s | 81.417793 |
| end-to-end tok/s | 0.314875 |
| layer load ms | 18416.054962 |
| final logits wall ms | 236.056085 |
| total wall ms | 19055.161428 |
| verified | `true` |

## 判断

- full mixed layer order `0..31` はrequest-state dispatchで通った。
- このrowはthroughput rowではない。
- 現在はrequest slotごとにresident weightsを複製しており、`layer_load_ms` が支配的で、real package throughputとは扱わない。
- `prefill_real_batch=false` / `decode_real_batch=false` のままなので、SQ/vLLM throughput比較には使わない。

## 次の行動

1. request slotごとのresident weight複製をやめ、shared resident weights + per-request state/cache bufferへ寄せる。
2. full package pathで `batching.mode=real`、`prefill_real_batch=true`、`decode_real_batch=true` のAQ4 baseline rowを保存する。
3. その後、SQ FP8候補を同じworkload gridへ接続する。

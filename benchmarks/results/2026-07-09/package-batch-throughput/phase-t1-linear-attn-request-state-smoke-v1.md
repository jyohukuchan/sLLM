# T1 linear-attn request state smoke v1

## 前回の要点

- `PackageLinearAttnResidentStepBatchLayer` は `RequestId` からlinear-attn resident layer state slotへ解決するownerとして追加済みだった。
- ただし前段の `phase-t1-linear-attn-request-state-owner-v1` はunit test中心で、実package上でrequest state ownerを実行した証拠はまだなかった。
- full mixed-attention runnerへ進む前に、linear-attention層のrecurrent stateとcausal Conv1d historyがrequest間で混ざらないことを小さく確認する必要があった。

## 今回の変更点

- `package-linear-attn-request-state-smoke` を追加した。
- R9700上で実packageのlinear-attention layer `0` を `request_count=2`、`sequence_len=2` でinterleaved実行した。
- batch ownerの出力を保存した後、ownerをdropし、requestごとに単体 `PackageLinearAttnResidentStepLayer` をロードし直してserial referenceと比較した。
- unknown request idが拒否されることも同じsmokeで確認した。

## R9700 smoke

Command:

```text
target/debug/ullm-engine package-linear-attn-request-state-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d 2 1048576 0 2 2
```

Result:

| field | value |
| --- | ---: |
| backend | `hip` |
| device index | 2 |
| layer | 0 |
| requests | 2 |
| sequence len | 2 |
| interleaved steps | 4 |
| hidden | 4096 |
| interleaved wall ms | 49.964805 |
| serial reference wall ms | 655.483195 |
| interleaved step/s | 80.056352 |
| output max abs | 12.886699677 |
| serial reference max abs diff | 0.000000000 |
| nonfinite count | 0 |
| unknown request rejected | `true` |
| verified | `true` |

## 判断

- このrowはthroughput rowではない。
- `PackageLinearAttnResidentStepBatchLayer` は実package上でもrequestごとのstate分離guardとして動いた。
- 現在の実装はrequest slotごとにresident layerを複製するため、full package throughputの最終形ではない。
- 次のT1では、full mixed-attention runnerのlayer enumへこのlinear-attn request-state ownerを接続し、小さいfull mixed path smokeへ進む。

## 次の行動

1. full mixed-attention runnerのlayer enumにself-attention resident step layerとlinear-attn request-state ownerを並べる。
2. manifest order `0..31` のうち、まず小さいB=2 / prompt=2 / generated=1のfull mixed path smokeを作る。
3. smokeが通った後、weights共有と実throughput計測へ進む。

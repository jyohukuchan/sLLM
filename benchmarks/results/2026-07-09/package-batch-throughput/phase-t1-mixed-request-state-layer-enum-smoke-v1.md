# T1 mixed request-state layer enum smoke v1

## 前回の要点

- `PackageLinearAttnResidentStepBatchLayer` は実package smokeまで通っていた。
- `PackageSelfAttnResidentStepBatchLayer` はunit testでrequest-id dispatch境界を追加済みだった。
- 次のblockerは、両ownerを同じlayer enumに並べ、linear-attn層からself-attn層へdevice bufferを渡せることの確認だった。

## 今回の変更点

- `PackageMixedRequestStateLayer` を追加した。
- `package-token-ids-mixed-request-state-smoke` を追加した。
- token IDからembedding rowを読み、requestごとのlinear-attn recurrent stateとself-attn paged KV stateへinterleavedに流すsmokeにした。
- final hiddenにfinal RMSNormとlm_head top-kをかけ、requestごとのfinal top1 tokenを確認した。

## R9700 smoke

Command:

```text
target/debug/ullm-engine package-token-ids-mixed-request-state-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d 2 1048576 0,3 len:2x2 1 1 1024 32 10000000 0
```

Result:

| field | value |
| --- | ---: |
| backend | `hip` |
| device index | 2 |
| layers | `0,3` |
| layer kinds | `linear_attention,self_attention` |
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
| final top1 tokens | `151353,151353` |
| prefill total input tok/s | 56.881809 |
| decode total generated tok/s | 1089.598799 |
| end-to-end tok/s | 3.035110 |
| layer load ms | 1387.090175 |
| final logits wall ms | 248.150567 |
| verified | `true` |

## 判断

- このrowはthroughput rowではない。
- `PackageMixedRequestStateLayer` はlinear-attn ownerとself-attn ownerを同じrequest-id dispatch境界で実行できた。
- 現時点ではrequest slotごとにresident weightsとcache bufferを持つため、shared-weight real batch runnerではない。
- `prefill_real_batch=false` / `decode_real_batch=false` のままなので、SQ/vLLM throughput比較には使わない。

## 次の行動

1. `0,3` の小さいguardから `manifest-all` へ広げ、full mixed layer orderで壊れないことを確認する。
2. full manifest smoke後に、shared resident weights + per-request state bufferへ寄せる。
3. その後、full packageで `batching.mode=real`、`prefill_real_batch=true`、`decode_real_batch=true` のAQ4 baseline rowを作る。

# T2 SQ FP8 mixed request-state resident throughput small grid v1

## 前回の要点

- T1ではfull mixed `manifest-all` AQ4 pathで `throughput_row=true`、`load_excluded_from_total=true`、`prefill_real_batch=true`、`decode_real_batch=true` のresident baselineを保存した。
- 次のT2要件は、SQ FP8 candidateを同じfull mixed resident schemaへ接続し、AQ4/SQのqualityとthroughputを同じ条件で比較することだった。

## 今回の変更点

- `sq-fp8-token-ids-mixed-request-state-smoke` を追加した。
- full mixed request-state loaderへ `Qwen3PackageSqOverlay` を渡し、artifactに存在するtensorだけSQ FP8からF32 resident bufferへmaterializeするようにした。
- artifactに存在しないtensorは従来どおりAQ4 resident matvecへfallbackする。
- `PackageAq4ResidentMatvec` はAQ4 storageとSQ/F32 materialized storageを持てるようになった。
- SQ/F32 storageが混じる場合、融合AQ4 kernelは使わず、F32 `matvec_f32` と小さいhost-side vector fallbackで接続する。
- stdoutに `sq_execution_mode=materialized_f32_fallback` を追加した。
- `run-external-benchmark.py` は `sq_execution_mode` をworkload metadataとして保持する。

このrowはSQ overlay接続とquality guard確認のprobeであり、native SQ kernelの速度代表値ではない。

## R9700 small grid

Common command shape:

```text
target/debug/ullm-engine sq-fp8-token-ids-mixed-request-state-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d /tmp/ullm-sq-fp8-kup6-gate5-down5-policy-v0.1-artifact 2 1048576 manifest-all len:2xB 1 1 1024 32 10000000 0
```

SQ artifact:

| field | value |
| --- | --- |
| candidate | `sq-fp8-w8a16-r9700-v0` |
| artifact | `/tmp/ullm-sq-fp8-kup6-gate5-down5-policy-v0.1-artifact` |
| schema | `sq-fp8-artifact-v0.1` |
| FP8 tensors | 22 |
| passthrough tensors | 753 |
| row chunk | 256 |

Result:

| batch | mode | prefill real | decode real | AQ4 end-to-end tok/s | SQ end-to-end tok/s | SQ prefill tok/s | SQ decode tok/s | SQ total wall ms | SQ outer wall ms | AQ4 final top1 | SQ final top1 | top1 match |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | `single` | `false` | `false` | 8.926325 | 7.260317 | 13.047930 | 24.609341 | 413.205112 | 94226.557775 | `44370` | `44370` | `true` |
| 4 | `real` | `true` | `true` | 24.096096 | 14.585119 | 20.661755 | 24.890457 | 822.756416 | 100598.061257 | `44370,5446,10701,25411` | `44370,1622,10701,25411` | `false` |
| 8 | `real` | `true` | `true` | 34.577530 | 17.961835 | 22.932277 | 25.066788 | 1336.166394 | 100614.600986 | `44370,5446,10701,25411,21901,685,279,27973` | `44370,1622,10701,25411,21901,685,279,27973` | `false` |

All SQ rows:

- `status=ok`
- `verified=true`
- `sq_overlay=true`
- `throughput_row=true`
- `load_excluded_from_total=true`
- `final_logits_in_total=true`
- `request_batch_executor=true`
- `fused_request_batch=false`
- `prefill_mode=token_id_full_mixed_request_state`
- `layers_csv=0..31`

## 判断

- SQ FP8 candidateをfull mixed resident pathへ接続できた。
- B=4/B=8では2番目requestのfinal top1がAQ4 baselineと一致しないため、この `kup6_gate5_down5` candidateはfull mixed quality guardを通過していない。
- 現在のSQ速度はF32 materialized fallbackを含むため、SQ format本来の速度評価ではなく、T2接続probeとして扱う。
- 次の速度評価には、SQ nativeまたは少なくともFP8 decodeをAQ4同等のresident matvec境界へ入れる実装が必要である。

## 次の行動

1. top1 driftが出ない保守的SQ candidateをfull mixed pathで再評価する。
2. resident F32 fallbackを超えて、SQ FP8を直接読むmatvec kernelまたは低遅延dequant matvecを実装する。
3. native SQ rowができたら、同じB=1/4/8 schemaでAQ4/SQ/vLLM比較へ戻る。

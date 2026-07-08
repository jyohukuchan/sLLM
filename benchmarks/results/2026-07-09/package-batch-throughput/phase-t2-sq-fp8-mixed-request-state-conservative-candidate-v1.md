# T2 SQ FP8 mixed request-state conservative candidate v1

## 前回の要点

- `kup6_gate5_down5` SQ FP8 candidateはfull mixed `manifest-all` request-batch pathへ接続できた。
- ただしB=4/B=8で2番目requestのfinal top1がAQ4 baselineからずれたため、full mixed quality guardは通過していない。
- 速度はまだ `sq_execution_mode=materialized_f32_fallback` であり、native SQ kernelの速度代表値ではない。

## 今回の変更点

- full mixed pathでstrict top1が崩れない最小候補を再評価した。
- `/tmp/ullm-sq-fp8-k-layer3-rb16-model-loop-candidate-v0.1-artifact` をB=1/4/8で実行し、AQ4 baselineのfinal top1と比較した。
- 1 tensor拡張候補として `up-layer3`、2 tensor拡張候補として `kup1-layer3-k16-up32` をB=4で確認した。

このrowもSQ overlay接続とquality boundary確認のprobeであり、native SQ kernelの速度代表値ではない。

## R9700 candidate grid

Common command shape:

```text
target/debug/ullm-engine sq-fp8-token-ids-mixed-request-state-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d ARTIFACT 2 1048576 manifest-all len:2xB 1 1 1024 32 10000000 0
```

Passing artifact:

| field | value |
| --- | --- |
| candidate | `sq-fp8-w8a16-r9700-v0-k-layer3-rb16` |
| artifact | `/tmp/ullm-sq-fp8-k-layer3-rb16-model-loop-candidate-v0.1-artifact` |
| tensor | `model.language_model.layers.3.self_attn.k_proj.weight` |
| schema | `sq-fp8-artifact-v0.1` |
| FP8 tensors | 1 |
| passthrough tensors | 774 |
| row chunk | 256 |

Result:

| candidate | batch | mode | prefill real | decode real | AQ4 end-to-end tok/s | SQ end-to-end tok/s | SQ prefill tok/s | SQ decode tok/s | SQ total wall ms | SQ outer wall ms | AQ4 final top1 | SQ final top1 | top1 match |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `k-layer3-rb16` | 1 | `single` | `false` | `false` | 8.926325 | 8.301199 | 17.180309 | 80.069802 | 361.393565 | 10438.252231 | `44370` | `44370` | `true` |
| `k-layer3-rb16` | 4 | `real` | `true` | `true` | 24.096096 | 25.144005 | 48.691111 | 81.134326 | 477.250937 | 12263.241359 | `44370,5446,10701,25411` | `44370,5446,10701,25411` | `true` |
| `k-layer3-rb16` | 8 | `real` | `true` | `true` | 34.577530 | 35.820128 | 63.272579 | 81.162501 | 670.014365 | 16353.182277 | `44370,5446,10701,25411,21901,685,279,27973` | `44370,5446,10701,25411,21901,685,279,27973` | `true` |

Rejected expansion checks:

| candidate | artifact | batch | FP8 tensors | SQ end-to-end tok/s | AQ4 final top1 | SQ final top1 | top1 match |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `up-layer3` | `/tmp/ullm-sq-fp8-up-layer3-model-loop-candidate-v0.1-artifact` | 4 | 1 | 22.785325 | `44370,5446,10701,25411` | `44370,1622,10701,25411` | `false` |
| `kup1-layer3-k16-up32` | `/tmp/ullm-sq-fp8-kup1-layer3-k16-up32-policy-v0.1-artifact` | 4 | 2 | 20.726920 | `44370,5446,10701,25411` | `44370,1622,10701,25411` | `false` |

All rows:

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
- `sq_execution_mode=materialized_f32_fallback`

## 判断

- full mixed request-batch pathで現在promoteできる保守的SQ候補は、layer3 `k_proj` row-block16の1 tensorだけである。
- layer3 `up_proj` は単体でもB=4の2番目request top1を `5446` から `1622` へ変えるため、`k+up` へ広げるcandidateはpromoteしない。
- B=4/B=8は `prefill_real_batch=true` / `decode_real_batch=true` で通っているため、selected-layerだけでなくfull mixed request-batch guardとして扱える。
- ただし速度はmaterialized F32 fallbackの影響を受けるため、SQ format本来のthroughput評価ではなくquality boundary確認として扱う。

## 次の行動

1. `k-layer3-rb16` をfull mixed strict-top1 regression subsetとして固定する。
2. 次はSQ FP8 direct matvec、または低遅延dequant matvecを実装し、materialized F32 fallbackから外す。
3. native SQ rowができたら、B=1/4/8だけでなく長いprefill/prefix gridにも同じcandidateを通す。

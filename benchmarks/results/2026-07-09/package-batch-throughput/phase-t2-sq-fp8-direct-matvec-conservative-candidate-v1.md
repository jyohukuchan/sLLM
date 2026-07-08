# T2 SQ FP8 direct matvec conservative candidate v1

## 前回の要点

- full mixed strict-top1 regression subsetとしてpromoteできる保守候補は、layer3 `k_proj` row-block16の1 tensorだけだった。
- 直前の結果は `sq_execution_mode=materialized_f32_fallback` であり、SQ payloadをF32 resident bufferへ全展開していた。

## 今回の変更点

- runtimeへ `ullm_runtime_sq_fp8_matvec_f32` を追加した。
- HIPRTC kernel `ullm_sq_fp8_matvec_f32_kernel` は、FP8 E4M3 payload byteとF32 scaleを直接読み、matvec内でdequantしてF32 outputへaccumulateする。
- `PackageAq4ResidentMatvec` に `SqFp8` storageを追加し、SQ overlay tensorはF32 materializationではなくpayload/scale resident bufferとして保持する。
- `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_KERNEL=1` でR9700上のfull mixed B=1/4/8を実行し、HIP direct kernelが使われることを確認した。

このrowは、materialized F32 fallbackを外す最初のdirect-dequant pathである。ただしまだ単体SGEMV形であり、AQ4のpair/triple/fused kernelと同等の融合は行っていない。

## R9700 direct matvec grid

Common command shape:

```text
ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_KERNEL=1 target/debug/ullm-engine sq-fp8-token-ids-mixed-request-state-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d /tmp/ullm-sq-fp8-k-layer3-rb16-model-loop-candidate-v0.1-artifact 2 1048576 manifest-all len:2xB 1 1 1024 32 10000000 0
```

Result:

| batch | mode | prefill real | decode real | direct prefill tok/s | direct decode tok/s | direct end-to-end tok/s | previous F32 fallback end-to-end tok/s | AQ4 end-to-end tok/s | final top1 | top1 match |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | `single` | `false` | `false` | 20.373389 | 79.999546 | 9.083299 | 8.301199 | 8.926325 | `44370` | `true` |
| 4 | `real` | `true` | `true` | 44.467487 | 81.012681 | 23.470577 | 25.144005 | 24.096096 | `44370,5446,10701,25411` | `true` |
| 8 | `real` | `true` | `true` | 63.262438 | 81.320426 | 35.598166 | 35.820128 | 34.577530 | `44370,5446,10701,25411,21901,685,279,27973` | `true` |

All rows:

- `status=ok`
- `verified=true`
- `sq_overlay=true`
- `sq_execution_mode=direct_fp8_dequant_matvec`
- `throughput_row=true`
- `load_excluded_from_total=true`
- `final_logits_in_total=true`
- `request_batch_executor=true`
- `fused_request_batch=false`
- `prefill_mode=token_id_full_mixed_request_state`
- `layers_csv=0..31`
- `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_KERNEL=1`

## 判断

- SQ FP8 overlayの実行はF32 materialized bufferではなく、payload/scale resident bufferからのdirect dequant matvecへ移った。
- 保守候補のB=1/4/8 final top1はAQ4 baselineと一致した。
- 単一tensorだけの候補では、direct pathのend-to-end tok/sはmaterialized F32 fallbackとほぼ同等で、速度差はまだformat性能の判断材料として弱い。
- 速度改善には、SQ tensor数を増やしてもqualityが崩れない候補探索、またはSQ FP8 pair/triple/fused matvecへの拡張が必要である。

## 次の行動

1. `SqFp8` storageをpair/triple/fused fallbackの中でさらに無駄なく使えるようにする。
2. top1 driftしやすい `up_proj` 系は、row-block幅、scale粒度、W8A8/activation scaleを変えて再探索する。
3. qualityが通るcandidate数が増えた段階で、B=1/4/8と長いprefill/prefix gridを再計測する。

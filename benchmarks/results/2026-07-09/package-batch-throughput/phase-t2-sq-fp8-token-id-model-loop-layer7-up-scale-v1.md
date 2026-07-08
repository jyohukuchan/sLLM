# T2 SQ FP8 token-id model-loop layer7 up scale v1

## 前回の要点

- `kup1-layer3-k16-up32` はR9700 token-id model-loop prompt bundleで `3 / 3` strict top1を通った。
- layer7 isolationでは、`layer7-k16-up32` と `layer3-kup-plus-layer7-k16` は通ったが、`layer3-kup-plus-layer7-up32` は `case_a` で崩れた。
- したがって直近の境界は、layer3 k16/up32 passing probeにlayer7 `up_proj` を足したときの相互作用だった。

## 今回の変更点

- layer3 `k_proj` row-block16 + layer3 `up_proj` row-block32 + layer7 `k_proj` row-block16を固定した。
- layer7 `up_proj` について、fallback、row-block16、row-block64の3条件をpolicy artifact化して同じR9700 prompt bundleで測った。
- すべて `batching.mode=real`、`prefill_real_batch=true`、`decode_real_batch=true` のselected-layer model-loop行である。

## R9700 result

| row | layer7 up policy | strict top1 pass | final top1 | AQ4 top1 rank in SQ top8 | prefill tok/s | decode tok/s | VRAM consumed bytes |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| `layer7-up-fallback` | layer3 k16/up32 + layer7 k16; layer7 up fallback | 3 / 3 | `110784,237950,182949` | `1,1,1` | 33.074688 | 32.789651 | 5885886464 |
| `layer7-up16` | layer3 k16/up32 + layer7 k16/up16 | 2 / 3 | `110784,193706,182949` | `1,2,1` | 33.296695 | 32.967699 | 5252481024 |
| `layer7-up64` | layer3 k16/up32 + layer7 k16/up64 | 2 / 3 | `110784,193706,182949` | `1,2,1` | 32.987160 | 32.575694 | 5757931520 |

AQ4 baseline top1: `110784,237950,182949`

## 判断

- layer7 `up_proj` fallbackは `3 / 3` strict top1を維持した。
- layer7 `up_proj` row-block16とrow-block64は、どちらも `case_a` で `237950` から `193706` へ入れ替わった。
- どちらの失敗でもAQ4 top1はSQ top8のrank 2に残っているため、壊滅的崩壊ではなくranking driftとして扱う。
- 現在のT2境界では、layer3 k16/up32 + layer7 k16をpassing subsetとして保持し、layer7 `up_proj` はfallbackに残す。

## Artifacts

- `results.jsonl`
- `comparison.json`
- `sq-layer7-up-fallback/raw.json`
- `sq-layer7-up16/raw.json`
- `sq-layer7-up64/raw.json`
- `benchmarks/results/2026-07-09/sq-fp8-layer7-up-fallback-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-layer7-up16-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-layer7-up64-policy-v0.1.json`

## 次の行動

1. layer7 `up_proj` はfallback維持でT2 policy boundaryへ反映する。
2. 次はこのpassing subsetを基準に、追加family/layerを1つずつ戻してcase_a driftが再発する境界を探す。
3. throughput評価では、引き続きこのselected-layer tok/sをfull LM tok/sとは扱わない。

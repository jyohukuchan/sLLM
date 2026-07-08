# T2 SQ FP8 token-id model-loop layer7 add family v1

## 前回の要点

- layer7 `up_proj` scale probeでは、layer3 k16/up32 + layer7 k16のpassing subsetを維持するにはlayer7 `up_proj` fallbackが必要だった。
- 次のT2対象は、このpassing boundaryにlayer7の追加familyを1つずつ戻し、`case_a` driftが再発する境界を探すことだった。

## 今回の変更点

- layer3 `k_proj` row-block16 + layer3 `up_proj` row-block32 + layer7 `k_proj` row-block16を固定した。
- layer7 `up_proj` はfallbackのまま、layer7 `o_proj` row-block32、`gate_proj` row-block32、`down_proj` row-block64を個別に追加した。
- `o32` と `gate32` が個別に通ったため、追加で `o32+gate32` の組み合わせも測った。

## R9700 result

| row | coverage | FP8 tensors | strict top1 pass | final top1 | case_a AQ4 rank in SQ top8 | prefill tok/s | decode tok/s |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| `layer7-plus-o32` | layer3 k16/up32 + layer7 k16/o32; layer7 up fallback | 4 | 3 / 3 | `110784,237950,182949` | 1 | 33.065052 | 32.826436 |
| `layer7-plus-gate32` | layer3 k16/up32 + layer7 k16/gate32; layer7 up fallback | 4 | 3 / 3 | `110784,237950,182949` | 1 | 33.016730 | 32.651026 |
| `layer7-plus-down64` | layer3 k16/up32 + layer7 k16/down64; layer7 up fallback | 4 | 2 / 3 | `110784,111791,182949` | 2 | 33.213301 | 32.884681 |
| `layer7-plus-o32-gate32` | layer3 k16/up32 + layer7 k16/o32/gate32; layer7 up/down fallback | 5 | 2 / 3 | `110784,193706,182949` | 2 | 33.341784 | 33.048955 |

AQ4 baseline top1: `110784,237950,182949`

## 判断

- layer7 `o_proj` row-block32は単独追加で `3 / 3` strict top1を維持した。
- layer7 `gate_proj` row-block32も単独追加で `3 / 3` strict top1を維持した。
- layer7 `down_proj` row-block64は `case_a` で `237950` から `111791` へ入れ替わった。
- layer7 `o32+gate32` は、単独pass同士の組み合わせだが `case_a` で `193706` へ入れ替わった。
- したがって現在のboundaryでは、layer7 `up_proj` と `down_proj` はfallback維持、`o_proj` と `gate_proj` は片方ずつならpassing、同時追加はfailure guardである。

## Artifacts

- `results.jsonl`
- `comparison.json`
- `sq-layer7-plus-o32/raw.json`
- `sq-layer7-plus-gate32/raw.json`
- `sq-layer7-plus-down64/raw.json`
- `sq-layer7-plus-o32-gate32/raw.json`
- `benchmarks/results/2026-07-09/sq-fp8-layer7-up-fallback-plus-layer7-o32-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-layer7-up-fallback-plus-layer7-gate32-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-layer7-up-fallback-plus-layer7-down64-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-layer7-up-fallback-plus-layer7-o32-gate32-policy-v0.1.json`

## 次の行動

1. `layer7-plus-o32` と `layer7-plus-gate32` はpassing probesとして保持する。
2. `layer7-plus-down64` と `layer7-plus-o32-gate32` はfailure guardsとして残す。
3. 次は `o32+gate32` の組み合わせをより強いscale/layoutで回復できるか試すか、`o32` または `gate32` の片側branchでcoverageを広げる。
4. selected-layer tok/sは引き続きfull LM throughputとは扱わない。

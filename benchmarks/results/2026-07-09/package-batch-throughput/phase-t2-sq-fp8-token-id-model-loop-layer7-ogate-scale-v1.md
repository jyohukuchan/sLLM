# T2 SQ FP8 token-id model-loop layer7 o/gate scale v1

## 前回の要点

- layer7 add-family probeでは、layer7 `o_proj` row-block32と`gate_proj` row-block32は個別に `3 / 3` passした。
- ただし `o32+gate32` の同時追加は `case_a` で崩れた。
- 次の確認は、`o/gate` の同時追加がrow-block幅の強化で回復するかを見ることだった。

## 今回の変更点

- layer3 `k_proj` row-block16 + layer3 `up_proj` row-block32 + layer7 `k_proj` row-block16を固定した。
- layer7 `up_proj` と `down_proj` はfallbackのまま、`o/gate` の組み合わせを `o16+gate32`、`o32+gate16`、`o16+gate16` で評価した。
- すべて同じR9700 six-layer token-id model-loop prompt bundleで測った。

## R9700 result

| row | coverage | FP8 tensors | strict top1 pass | final top1 | case_a AQ4 rank in SQ top8 | prefill tok/s | decode tok/s |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| `layer7-ogate-o16-gate32` | layer3 k16/up32 + layer7 k16/o16/gate32; layer7 up/down fallback | 5 | 2 / 3 | `110784,193706,182949` | 2 | 33.189080 | 32.763964 |
| `layer7-ogate-o32-gate16` | layer3 k16/up32 + layer7 k16/o32/gate16; layer7 up/down fallback | 5 | 2 / 3 | `110784,193706,182949` | 2 | 33.008031 | 32.641597 |
| `layer7-ogate-o16-gate16` | layer3 k16/up32 + layer7 k16/o16/gate16; layer7 up/down fallback | 5 | 2 / 3 | `110784,193706,182949` | 2 | 32.706634 | 33.212571 |

AQ4 baseline top1: `110784,237950,182949`

## 判断

- `o16+gate32`、`o32+gate16`、`o16+gate16` はすべて `case_a` が `193706` へ入れ替わった。
- 失敗時もAQ4 top1 `237950` はSQ top8 rank `2` に残るため、壊滅的崩壊ではなくranking driftである。
- row-block16化だけでは、layer7 `o_proj` と `gate_proj` の同時追加は回復しない。
- 現在のT2境界では、`o_proj` と `gate_proj` は片方ずつのbranch候補として扱い、同時追加はfailure guardに残す。

## Artifacts

- `results.jsonl`
- `comparison.json`
- `sq-layer7-ogate-o16-gate32/raw.json`
- `sq-layer7-ogate-o32-gate16/raw.json`
- `sq-layer7-ogate-o16-gate16/raw.json`
- `benchmarks/results/2026-07-09/sq-fp8-layer7-up-fallback-plus-layer7-o16-gate32-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-layer7-up-fallback-plus-layer7-o32-gate16-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-layer7-up-fallback-plus-layer7-o16-gate16-policy-v0.1.json`

## 次の行動

1. `o+gate` 同時追加は現行W8A16/F32 row-block scaleではfailure guardとして保持する。
2. 次は `o32` branchまたは `gate32` branchのどちらかを選び、coverageを広げる。
3. `o+gate` 同時追加を回復する場合は、row-block幅ではなく別scale layout、別dtype、またはtext-level acceptance guardの導入後に再評価する。
4. selected-layer tok/sは引き続きfull LM throughputとは扱わない。

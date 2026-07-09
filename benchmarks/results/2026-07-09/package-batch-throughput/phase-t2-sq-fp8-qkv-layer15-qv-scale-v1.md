# T2 SQ FP8 qkv layer15 q/v scale prompt bundle v1

## 前回の要点

- layer15 `q16+v16` は`case_a`で `4105 -> 5582` に反転した。
- layer15 `q16+k16` と `k16+v16` はstrict top1を維持したため、driftはQ/V interactionに寄っていると判断した。

## 今回の変更点

- layer15 `q+v` interactionに対して、`q8/v16`、`q16/v8`、`q8/v8` を試した。
- baseはlayer3+7+11 `q16/k16/v16` のまま維持した。
- `scale.overrides[]` でlayer15 `q_proj` または `v_proj` だけrow-block8にした。
- 実行telemetryは前回と同じ `single+triple` 境界だった。

## R9700 result

| row | FP8 tensors | single count | pair count | triple count | prefill tok/s | decode tok/s | end-to-end tok/s | VRAM consumed bytes | final top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| AQ4 baseline reused | 0 | 0 | 0 | 0 | 66.811351 | 80.883171 | 35.370726 | 4443144192 | `24218,4105,329` |
| SQ `layers3/7/11 + layer15 q8-v16` | 11 | 46 | 0 | 69 | 59.384805 | 75.428031 | 32.777400 | 5200433152 | `24218,4105,329` |
| SQ `layers3/7/11 + layer15 q16-v8` | 11 | 46 | 0 | 69 | 59.351654 | 75.197563 | 32.678281 | 5194121216 | `24218,5582,329` |
| SQ `layers3/7/11 + layer15 q8-v8` | 11 | 46 | 0 | 69 | 55.622420 | 75.185622 | 30.515812 | 5202513920 | `24218,4105,329` |

## Quality comparison

### layer15 q8-v16

| prompt | AQ4 top1 | SQ top1 | strict top1 | top8 overlap | AQ4 top1 rank in SQ top8 | SQ top1 margin over rank2 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `len4` | 24218 | 24218 | pass | 7 / 8 | 1 | 0.278604508 |
| `case_a` | 4105 | 4105 | pass | 8 / 8 | 1 | 0.000858784 |
| `case_b` | 329 | 329 | pass | 8 / 8 | 1 | 0.190518379 |

### layer15 q16-v8

| prompt | AQ4 top1 | SQ top1 | strict top1 | top8 overlap | AQ4 top1 rank in SQ top8 | SQ top1 margin over rank2 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `len4` | 24218 | 24218 | pass | 7 / 8 | 1 | 0.278304577 |
| `case_a` | 4105 | 5582 | fail | 8 / 8 | 2 | 0.000601769 |
| `case_b` | 329 | 329 | pass | 8 / 8 | 1 | 0.189933777 |

### layer15 q8-v8

| prompt | AQ4 top1 | SQ top1 | strict top1 | top8 overlap | AQ4 top1 rank in SQ top8 | SQ top1 margin over rank2 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `len4` | 24218 | 24218 | pass | 7 / 8 | 1 | 0.278797150 |
| `case_a` | 4105 | 4105 | pass | 8 / 8 | 1 | 0.000812530 |
| `case_b` | 329 | 329 | pass | 8 / 8 | 1 | 0.191033840 |

## 判断

- layer15 `q8/v16` と `q8/v8` はstrict top1 `3 / 3` を維持した。
- layer15 `q16/v8` は`case_a`で `4105 -> 5582` のまま失敗した。
- したがって、回復に効いているのは主にlayer15 `q_proj` のrow-block16からrow-block8への細粒度化であり、`v_proj` だけ細かくしても足りない。
- 最小のpassing q/v scale refinementは `q8/v16` として扱う。
- このrunもtelemetry上はpair kernelではなく `single+triple` 境界なので、品質切り分けでありpair kernel速度評価ではない。

Artifacts:

- `benchmarks/results/2026-07-09/sq-fp8-qkv-layers3-7-11-plus-layer15-q8-v16-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-qkv-layers3-7-11-plus-layer15-q16-v8-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-qkv-layers3-7-11-plus-layer15-q8-v8-policy-v0.1.json`
- `results.schema.jsonl`
- `results.jsonl`
- `comparison.json`

## 次の行動

1. layer15 `q8/v16` を最小passing q/v refinementとして保存する。
2. 次はlayer15 `q8/k16/v16` を試し、full layer15 QKV相当をstrict top1に戻せるか確認する。
3. passした場合もfull SQ policyではなく、より広いprompt/text guardへ進める前の診断候補として扱う。

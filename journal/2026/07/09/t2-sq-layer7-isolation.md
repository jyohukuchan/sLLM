# T2 SQ layer7 isolation

## 前回の要点

- `kup1-layer3-k16-up32` はstrict top1を `3 / 3` 維持した。
- `kup2-k16-up32` はlayers `3,7` へ広げると `case_a` が崩れた。
- 次はlayer7単体と、layer3 passing probeへlayer7の片側だけを足す切り分けが必要だった。

## 今回の変更点

- `layer7 k16/up32`、`layer3 k16/up32 + layer7 k16`、`layer3 k16/up32 + layer7 up32` の3条件を作った。
- R9700 six-layer token-id model-loop prompt bundleで評価した。
- 結果を `uLLM-project/benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-layer7-isolation-v1.md` に保存した。

## 結果

| variant | pass | case_a SQ top1 | case_a AQ4 rank in SQ top8 |
| --- | ---: | ---: | ---: |
| `layer7-k16-up32` | 3 / 3 | 237950 | 1 |
| `layer3-kup-plus-layer7-k16` | 3 / 3 | 237950 | 1 |
| `layer3-kup-plus-layer7-up32` | 2 / 3 | 193706 | 3 |

## 次の行動

- `layer7-k16-up32` と `layer3-kup-plus-layer7-k16` はpassing probesとして保持する。
- `layer3-kup-plus-layer7-up32` は現在のfailure guardとして残す。
- 次はlayer7 `up_proj` のrow-block16、row-block64、またはfallbackを、layer3 k16/up32 + layer7 k16固定で試す。

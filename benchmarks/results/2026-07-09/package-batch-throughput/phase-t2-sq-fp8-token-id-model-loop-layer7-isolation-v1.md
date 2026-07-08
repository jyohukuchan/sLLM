# T2 SQ FP8 Model-Loop Layer7 Isolation v1

## 前回の要点

- `kup1-layer3-k16-up32` はstrict top1を `3 / 3` 維持した。
- 同じmixed scaleをlayers `3,7` へ広げた `kup2-k16-up32` は `case_a` が崩れた。
- 次はlayer7単体と、layer3 passing probeへlayer7の片側だけを足す切り分けが必要だった。

## 今回の変更点

- `layer7 k16/up32`、`layer3 k16/up32 + layer7 k16`、`layer3 k16/up32 + layer7 up32` の3条件を作った。
- すべて同じR9700 six-layer token-id model-loop prompt bundleで評価した。
- 比較は直前のAQ4 prompt-bundle baselineを参照し、top1 match、AQ4 top1 rank in SQ top8、top8 commonを保存した。

## Summary

| variant | coverage | FP8 tensors | pass | len4 SQ top1 | case_a SQ top1 | case_a AQ4 rank in SQ top8 | case_b SQ top1 | prefill tok/s | decode tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `layer7-k16-up32` | layer7 k row-block16 + up row-block32 | 2 | 3 / 3 | 110784 | 237950 | 1 | 182949 | 32.955513 | 32.577636 |
| `layer3-kup-plus-layer7-k16` | layer3 k16/up32 + layer7 k row-block16 | 3 | 3 / 3 | 110784 | 237950 | 1 | 182949 | 33.143258 | 32.135649 |
| `layer3-kup-plus-layer7-up32` | layer3 k16/up32 + layer7 up row-block32 | 3 | 2 / 3 | 110784 | 193706 | 3 | 182949 | 31.107852 | 32.971147 |

## Interpretation

`layer7-k16-up32` alone passes, and adding only layer7 `k_proj` to the layer3 passing probe also passes. Adding layer7 `up_proj` to the layer3 passing probe fails `case_a`. This narrows the immediate T2 boundary to layer7 `up_proj` interaction with the layer3 k/up mixed-scale probe.

The failure is still close: AQ4 top1 remains in SQ top8 at rank 3 for `case_a`. Under the current strict top1 promotion rule this remains a failure guard, not an accepted SQ policy.

These timings are selected-layer model-loop diagnostics. They are not final full-package SQ throughput, and wrapper elapsed includes artifact read/materialization.

## Artifacts

- `results.jsonl`
- `comparison.json`
- `sq-kup1-layer7-k16-up32/*`
- `sq-kup1-layer3-k16-up32-plus-layer7-k16/*`
- `sq-kup1-layer3-k16-up32-plus-layer7-up32/*`

## 次の行動

1. Keep `layer7-k16-up32` and `layer3-kup-plus-layer7-k16` as passing probes, not promoted policies.
2. Keep `layer3-kup-plus-layer7-up32` as the current failure guard.
3. Next probe should target layer7 `up_proj`: row-block16, row-block64, or fallback while keeping layer3 k16/up32 and layer7 k16 fixed.

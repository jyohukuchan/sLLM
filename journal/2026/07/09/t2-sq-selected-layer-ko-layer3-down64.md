# T2 SQ selected-layer k/o layer3 down64

## 前回の要点

- `selected-layer-ko-plus-layer3-o32` は `3 / 3` strict top1 passだった。
- layer3 `gate_proj` row-block32/16はどちらもlen4でstrict top1を壊した。

## 今回の変更点

- current 13 tensor branchにlayer3 `down_proj` row-block64を追加した。
- R9700 six-layer token-id model-loop prompt bundleで測った。

## 結果

| variant | pass | final top1 | AQ4 rank in SQ top8 | prefill tok/s | decode tok/s |
| --- | ---: | --- | --- | ---: | ---: |
| `selected-layer-ko-plus-layer3-o32-down64` | `3 / 3` | `110784,237950,182949` | `1,1,1` | `33.091248` | `32.876952` |

AQ4 baseline top1: `110784,237950,182949`

## 判断

- layer3 `down_proj` row-block64はcurrent branchへ追加できる。
- current passing branchは layer3 `k16/o32/up32/down64` + layers 7/11/15/19/23 `k16/o32`。
- layer3 `gate_proj` row-block32/16はfailure guardとして残す。

## 次の行動

1. 14 tensor版 `selected-layer-ko-plus-layer3-o32-down64` をcurrent passing branchとして保持する。
2. 次はlayer11 `up_proj` row-block32を足せるかを見る。

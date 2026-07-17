# T2 SQ selected-layer k/o layer3 gate scale

## 前回の要点

- `selected-layer-ko-plus-layer3-o32` は `3 / 3` strict top1 passだった。
- current boundaryは layer3 `k16/o32/up32` + layers 7/11/15/19/23 `k16/o32`。

## 今回の変更点

- layer3 `gate_proj` row-block32を追加した `selected-layer-ko-plus-layer3-o32-gate32` を作った。
- gate32がlen4で失敗したので、layer3 `gate_proj` だけrow-block16にした `gate16` も作った。
- R9700 six-layer token-id model-loop prompt bundleで両方を測った。

## 結果

| variant | pass | final top1 | len4 AQ4 rank in SQ top8 | prefill tok/s | decode tok/s |
| --- | ---: | --- | ---: | ---: | ---: |
| `gate32` | `2 / 3` | `102446,237950,182949` | 3 | `28.610392` | `28.109230` |
| `gate16` | `2 / 3` | `102446,237950,182949` | 3 | `33.121674` | `32.765115` |

AQ4 baseline top1: `110784,237950,182949`

## 判断

- layer3 `gate_proj` はrow-block32でもrow-block16でもlen4のstrict top1を壊す。
- AQ4 top1はSQ top8に残るが、T2 promotion ruleはstrict top1なのでpromoteしない。
- current passing branchは13 tensor版 `selected-layer-ko-plus-layer3-o32` のまま。

## 次の行動

1. layer3 `gate_proj` row-block32/16をfailure guardとして残す。
2. 次はlayer3 `down_proj` row-block64をcurrent branchへ足せるかを見る。

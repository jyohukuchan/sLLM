# T2 SQ FP8 QKV layer15 pair

## 前回の要点

- layer15 single splitはすべてstrict top1 `3 / 3` だった。
- layer15 full QKVは`case_a`だけfailした。

## 今回の変更点

- layer15 `q+k`、`q+v`、`k+v` pair候補を作り、R9700 full mixed prompt bundleで測定した。

## 結果

| variant | prefill tok/s | decode tok/s | end-to-end tok/s | strict top1 | final top1 |
| --- | ---: | ---: | ---: | --- | --- |
| `q16-k16` | 59.257498 | 75.399241 | 33.018293 | 3 / 3 | `24218,4105,329` |
| `q16-v16` | 59.700485 | 75.017792 | 33.389594 | 2 / 3 | `24218,5582,329` |
| `k16-v16` | 55.666617 | 75.741372 | 30.225450 | 3 / 3 | `24218,4105,329` |

## 判断

- `q+k` と `k+v` はpass。
- `q+v` は`case_a`でfail。
- layer15 driftはQ/V interactionに寄っている。

## 次の行動

- layer15 `q+v` のscale再調整を試す。

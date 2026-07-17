# T2 SQ FP8 QKV layer15 q/v scale

## 前回の要点

- layer15 `q16+v16` は`case_a`でfailした。
- `q+k` と `k+v` はpassしたため、Q/V interactionに絞った。

## 今回の変更点

- `q8/v16`、`q16/v8`、`q8/v8` をR9700 full mixed prompt bundleで測定した。

## 結果

| variant | prefill tok/s | decode tok/s | end-to-end tok/s | strict top1 | final top1 |
| --- | ---: | ---: | ---: | --- | --- |
| `q8-v16` | 59.384805 | 75.428031 | 32.777400 | 3 / 3 | `24218,4105,329` |
| `q16-v8` | 59.351654 | 75.197563 | 32.678281 | 2 / 3 | `24218,5582,329` |
| `q8-v8` | 55.622420 | 75.185622 | 30.515812 | 3 / 3 | `24218,4105,329` |

## 判断

- `q8/v16` と `q8/v8` はpass。
- `q16/v8` はfail。
- 回復に効いているのはlayer15 `q_proj` のrow-block8化。

## 次の行動

- layer15 `q8/k16/v16` を試し、full layer15 QKV相当を戻せるか見る。

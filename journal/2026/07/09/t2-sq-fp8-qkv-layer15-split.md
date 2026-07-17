# T2 SQ FP8 QKV layer15 split

## 前回の要点

- layer15 Q/K/V同時追加では`case_a`だけstrict top1が崩れた。

## 今回の変更点

- layer3+7+11 `q16/k16/v16` をbaseに、layer15 `q16`、`k16`、`v16` を単独追加して比較した。

## 結果

| variant | prefill tok/s | decode tok/s | end-to-end tok/s | strict top1 |
| --- | ---: | ---: | ---: | --- |
| `q16` | 59.071645 | 75.280931 | 33.180809 | 3 / 3 |
| `k16` | 56.575977 | 75.925142 | 31.492802 | 3 / 3 |
| `v16` | 59.481707 | 75.949459 | 32.897182 | 3 / 3 |

## 判断

- layer15 `q16/k16/v16` は単独ならすべてstrict top1 `3 / 3` を維持した。
- 同時追加時だけ崩れるため、単独projectionのhard failureではなく累積driftとして扱う。

## 次の行動

- layer3+7+11をcurrent passing boundaryとして維持する。
- layer15はpair split、scale再調整、または広いtext guardを追加してから再挑戦する。

# T2 SQ FP8 q16/k16/v16 short batch guard

## 前回の要点

- layer3 `q16/k16/v16` はfull mixed prompt bundleでstrict top1 `3 / 3`。
- layer7以降へ広げる前に、短いB=1/4/8 guardで再確認する必要があった。

## 今回の変更点

- B=1/4/8、prompt `len:2xB`、generated 1、top_k 1でAQ4/SQを比較した。
- SQはtriple direct kernel必須で実行した。

## 結果

- B=1/4/8すべてtop1一致。
- SQ triple countはB=1/4/8で `3/12/24`。
- SQ decode tok/sは `79.245965 / 79.959365 / 80.030092`。

## 次の行動

1. `q16/k16/v16` をlayer3 QKV triple pass boundaryとして固定する。
2. 次はlayer7 QKVへ同じrow-block16を足す。

# T2 SQ FP8 QKV layer15 q8 full QKV

## 前回の要点

- `q8/v16` と `q8/v8` はpassし、`q16/v8` はfailした。
- 回復に効いているのはlayer15 `q_proj` のrow-block8化だった。

## 今回の変更点

- layer15 Q/K/V同時追加に戻し、layer15 `q_proj` だけrow-block8、`k_proj`/`v_proj` はrow-block16にした。

## 結果

- strict top1: `3 / 3`
- final top1: `24218,4105,329`
- SQ prefill tok/s: `62.888070`
- SQ decode tok/s: `74.903309`
- SQ end-to-end tok/s: `33.650360`
- case_a margin: `0.000087261`
- triple count: `92`

## 判断

- layer15 QKV同時追加は `q8/k16/v16` で回復した。
- ただしcase_a marginが非常に薄いため、まだfull SQ policyにはしない。

## 次の行動

- B=1/4/8 short guardまたは広いprompt/text guardで安定性を確認する。

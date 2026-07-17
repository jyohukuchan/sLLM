# T2 SQ FP8 full mixed qk pair prompt bundle

## 前回の要点

- layer3 `q/k/v` tripleはfull mixed prompt bundleで `case_a` のtop1を入れ替え、strict top1 `2 / 3` だった。
- 原因を狭めるには、`v_proj` をAQ4に戻して `q/k` pairだけを試す必要があった。

## 今回の変更点

- layer3 `q/k` pair候補を同じ `len4`、`case_a`、`case_b` prompt bundleで測定した。
- `ULLM_DISABLE_AQ4_MATVEC_TRIPLE_SELF_ATTN_QKV=1` と `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_PAIR_KERNEL=1` でpair境界を強制した。
- 比較結果を `comparison.json` と `results.jsonl` に保存した。

## 結果

- AQ4 final top1: `24218,4105,329`
- SQ final top1: `24218,4105,329`
- strict top1: `3 / 3`
- SQ telemetry: `sq_projection_boundary=pair`, `sq_fp8_pair_matvec_count=23`
- `case_a` はSQ top1 marginが約 `0.000080586` と非常に薄い。

## 次の行動

1. `q/k` pairはfull mixed prompt-bundle pass境界として保存する。
2. `v_proj` 単体または `q/k + v` の別scale粒度を試し、`q/k/v` failureの原因をさらに絞る。

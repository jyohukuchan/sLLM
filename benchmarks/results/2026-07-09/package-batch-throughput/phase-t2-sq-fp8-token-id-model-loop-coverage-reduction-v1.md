# T2 SQ FP8 Model-Loop Coverage Reduction v1

## 前回の要点

- `kup6_gate5_down5` はdirect logits prompt bundleではstrict top1を維持したが、six-layer token-id model-loop prompt bundleでは `len4` と `case_a` が崩れた。
- 現行のpromotion ruleはstrict top1なので、model-loop guardで崩れる候補はSQ quality policyへ昇格しない。

## 今回の変更点

- k/up row-block32のcoverageを `6 -> 5 -> 4 -> 3 -> 2 -> 1` layerへ順に削り、同じR9700 token-id model-loop prompt bundleで評価した。
- layer3だけについて `k_proj` 単体、`up_proj` 単体、`k_proj` row-block16、`k/up` row-block16を追加評価した。
- 比較は直前のAQ4 prompt-bundle baselineを参照し、top1 match、AQ4 top1 rank in SQ top8、top8 commonを保存した。

## Summary

| variant | coverage | FP8 tensors | pass | len4 SQ top1 | case_a SQ top1 | case_a AQ4 rank in SQ top8 | case_b SQ top1 | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `kup6-rowblock32` | k/up layers 3,7,11,15,19,23 row-block32 | 12 | 2 / 3 | 110784 | 193706 | 4 | 182949 | case_a drift |
| `kup5-rowblock32` | k/up layers 3,7,11,15,19 row-block32 | 10 | 2 / 3 | 110784 | 193706 | 5 | 182949 | case_a drift |
| `kup4-rowblock32` | k/up layers 3,7,11,15 row-block32 | 8 | 2 / 3 | 110784 | 193706 | 5 | 182949 | case_a drift |
| `kup3-rowblock32` | k/up layers 3,7,11 row-block32 | 6 | 2 / 3 | 110784 | 193706 | 4 | 182949 | case_a drift |
| `kup2-rowblock32` | k/up layers 3,7 row-block32 | 4 | 2 / 3 | 110784 | 193706 | 3 | 182949 | case_a drift |
| `kup1-layer3-rowblock32` | k/up layer 3 row-block32 | 2 | 2 / 3 | 110784 | 124170 | 4 | 182949 | case_a drift |
| `k-layer3-rowblock32` | k layer 3 row-block32 | 1 | 2 / 3 | 110784 | 111791 | 2 | 182949 | case_a drift |
| `up-layer3-rowblock32` | up layer 3 row-block32 | 1 | 3 / 3 | 110784 | 237950 | 1 | 182949 | passes strict top1 |
| `k-layer3-rowblock16` | k layer 3 row-block16 | 1 | 3 / 3 | 110784 | 237950 | 1 | 182949 | passes strict top1 |
| `kup1-layer3-rowblock16` | k/up layer 3 row-block16 | 2 | 2 / 3 | 110784 | 193706 | 3 | 182949 | case_a drift |

## Interpretation

The model-loop guard shows a stricter boundary than the direct logits guard. k/up row-block32 fails `case_a` even when reduced to layer 3 only. `up_proj` layer 3 alone preserves strict top1, while `k_proj` layer 3 needs row-block16 to preserve strict top1. However, combining layer3 k/up with row-block16 still fails `case_a`, so the interaction between selected tensors matters and cannot be judged by single-tensor pass/fail alone.

The immediate T2 direction should be to treat `k_proj` as a higher-risk family in model-loop quality guards, test finer scale layouts per family, and avoid promoting k/up coverage from direct logits evidence alone.

Throughput and VRAM values in these rows are selected-layer model-loop diagnostics. They are useful for same-path sanity checks, but they are not final full-package SQ throughput and wrapper elapsed includes artifact read/materialization for generated SQ artifacts.

## Artifacts

- `results.jsonl`
- `comparison.json`
- `sq-*/stdout.log`, `raw.json`, `memory.jsonl`, `stderr.log`

## 次の行動

1. `up_proj` layer3 row-block32 and `k_proj` layer3 row-block16 are passing single-tensor probes, not combined SQ policies.
2. Add per-family/per-tensor scale-layout support before testing mixed scale policies such as k row-block16 + up row-block32 in one artifact.
3. Continue T2 with model-loop prompt-bundle guards as the promotion gate; direct logits guards remain diagnostic only.

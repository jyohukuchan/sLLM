# T2 SQ FP8 overlay acceptance rule

## 前回の要点

- SQ FP8 mixed row-block candidateは、`v` fallback + `q/k/o/gate/up/down` row-block32 FP8として、4-5層までは有望だった。
- 6層bundleとall self-attention probeではstrict top1が崩れた。
- top-k overlap、AQ4 top1 rank、logit gapは保存されていたが、T2昇格条件としての扱いが曖昧だった。

## 今回の変更点

- `tools/evaluate-sq-fp8-overlay-acceptance.py` を追加した。
- `tests/test_sq_fp8_overlay_acceptance.py` を追加した。
- T2 promotion rule v0.1を `strict_top1` にした。
- 診断専用ruleとして `topk_common >= 5`、`baseline_top1_rank_in_sq_topk <= 2`、`abs(gap) <= 0.15` を保存するようにした。
- 診断ruleはstrict top1 failureを上書きしない。
- mixed candidate guard bundleとlayer 23 fallback probeを評価し、10ケース中strict top1 passは `5 / 10`、accepted for T2 promotionはfalseだった。
- 結果を `benchmarks/results/2026-07-08/sq-fp8-mixed-candidate-acceptance-v0.1.md/json` に保存した。
- plan、state freeze、SQ FP8 artifact specにacceptance ruleを反映した。

## 次の行動

1. text-level guardを正式採用するまでは、T2 promotionにstrict top1一致を要求する。
2. 6層bundleのstrict top1 failureを、per-layer/family fallbackまたはstronger scale/layoutで潰す。
3. mixed candidateはまだT5 throughput比較用のpromoted SQ policyとして扱わない。

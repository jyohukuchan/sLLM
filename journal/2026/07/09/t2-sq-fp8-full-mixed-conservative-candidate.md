# T2 SQ FP8 full mixed conservative candidate

## 前回の要点

- `kup6_gate5_down5` はfull mixed B=4/B=8で2番目requestのtop1がAQ4 baselineからずれた。
- SQ mixed pathは `materialized_f32_fallback` で接続済みだが、native SQ throughput評価ではない。

## 今回の変更点

- R9700で `sq-fp8-w8a16-r9700-v0-k-layer3-rb16` をfull mixed `manifest-all` B=1/4/8で再計測した。
- B=1/4/8すべてでAQ4 final top1と一致した。
- `up-layer3` と `kup1-layer3-k16-up32` はB=4で2番目requestが `5446` から `1622` へずれた。
- 結果を `uLLM-project/benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-mixed-request-state-conservative-candidate-v1.md` と `results.jsonl` に保存した。

## 次の行動

1. `k-layer3-rb16` をfull mixed strict-top1 regression subsetとして扱う。
2. 次はSQ FP8 direct matvecまたは低遅延dequant matvecへ進み、materialized F32 fallbackから外す。
3. native SQ rowができたら、同じcandidateをB=1/4/8と長いprefill/prefix gridへ流す。

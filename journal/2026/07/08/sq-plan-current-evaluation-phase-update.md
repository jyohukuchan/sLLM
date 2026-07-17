# SQ plan current evaluation phase update

## 前回の要点

- FlashAttention2-style cached-prefix/cold-prefill componentは、R9700でSQ候補評価を始める前提速度として一旦十分と判断した。
- SQ FP8候補は、row-block32とfallbackの組み合わせで品質境界を狭め、次の短期候補が `kup6_gate5_down5` になった。
- T1 JSONL/schema preservationは完了扱いだが、real batch runnerは未完了のまま残っている。

## 今回の変更点

- `docs/plans/fp8-sq-r9700-batch-throughput-prefill-plan-v0.1.md` を更新した。
- 追加のFlashAttention2-like最適化を主タスクから外し、SQ比較で不足が見えたcaseだけに限定する方針にした。
- SQ策定フェーズの主順序を、T2品質境界固定、T1 real batch throughput runner、T5 AQ4/FP8比較、T6/T7 vLLM比較に整理した。
- `benchmarks/results/2026-07-08/sq-r9700-state-freeze-v0.1.md` と `.json` のNext Actionを同じ方針に同期した。

## 次の行動

1. `kup6_gate5_down5` をcase_a/case_bで確認し、6層prompt-bundleのstrict top1 guardを作る。
2. T1 real batch runnerをfull package pathへ接続し、`batch=1/4/8` のprefill/decode/end-to-end total throughputを保存する。
3. FP8 SQ候補1とAQ4 latest baselineを同じworkload gridで比較する。
4. uLLM側のR9700結果が揃った後、vLLMを同じgridで測る。

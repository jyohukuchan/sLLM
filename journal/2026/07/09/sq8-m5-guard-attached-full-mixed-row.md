# SQ8_0 M5 Guard-Attached Full Mixed Row

## 前回の要点

- M5のparser/schema整備により、artifact timingとprompt-suite regression statusを`inference-benchmark-result-v0.1`へ保存できるようになった。
- 既存の`sq-layer23-k16` behavioral guard bundleは、`acceptance_mode=behavioral`でpassしている。
- package、SQ artifact、`target/debug/ullm-engine`はいずれも再実行可能な状態だった。

## 今回の変更点

- `sq-fp8-token-ids-mixed-request-state-smoke`をfull mixed `manifest-all` prompt bundleで再実行した。
- `--prompt-guard-bundle-json`でbehavioral guard bundleを添付した。
- 結果を`benchmarks/results/2026-07-09/package-batch-throughput/phase-m5-sq8-guard-attached-full-mixed-v1/`へ保存した。
- `summary.md`を追加し、M5の最低フィールドが揃っていることを要約した。

## 次の行動

- M10へ進む前に、vLLM + FP8と同じprompt length / generated length / concurrencyのuLLM SQ8_0 rowを追加する。

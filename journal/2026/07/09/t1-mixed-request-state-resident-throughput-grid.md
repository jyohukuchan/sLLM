# T1 mixed request-state resident throughput grid

## 前回の要点

- full mixed `manifest-all` は request-batch executor 境界まで進んだ。
- 前回rowは `throughput_row=false` で、`total_wall_ms` が layer load 込みだった。

## 今回の変更点

- `package-token-ids-mixed-request-state-smoke` の `total_wall_ms` を `prefill + decode + final_logits` に変更した。
- load込みの時間は `outer_wall_ms` に分離した。
- stdout/JSONLで `throughput_row=true`、`load_excluded_from_total=true`、`final_logits_in_total=true` を保存するようにした。
- parserは単一値CSVを1要素リストとして扱い、mixed pathの `max_new_tokens_csv` を generated token listとして扱うようにした。
- R9700 AQ4 full mixed `manifest-all` の batch `1/4/8` を測定した。

## 結果

- Report: `uLLM-project/benchmarks/results/2026-07-09/package-batch-throughput/phase-t1-mixed-request-state-resident-throughput-small-grid-v1.md`
- JSONL: `uLLM-project/benchmarks/results/2026-07-09/package-batch-throughput/phase-t1-mixed-request-state-resident-throughput-small-grid-v1/results.jsonl`
- B=1: end-to-end `8.926325 tok/s`, final top1 `[44370]`
- B=4: end-to-end `24.096096 tok/s`, final top1 `[44370,5446,10701,25411]`
- B=8: end-to-end `34.577530 tok/s`, final top1 `[44370,5446,10701,25411,21901,685,279,27973]`
- All rows: `status=ok`, `verified_all=true`, `request_batch_executor=true`, `throughput_row=true`, `load_excluded_from_total=true`, `final_logits_in_total=true`.

## 次の行動

- 次はSQ FP8 candidateを同じfull mixed resident pathに接続する。
- `fused_request_batch=false` のままなので、GPU fused batch速度の証拠とは分けて扱う。

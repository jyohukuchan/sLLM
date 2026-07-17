# T1 mixed request-state layer-batch executor

## 前回の要点

- full mixed `manifest-all` pathはrequest-state dispatchとweight bundle sharingまで進んでいた。
- ただし実行は `request_state_interleaved` で、`prefill_real_batch=false` / `decode_real_batch=false` のままだった。

## 今回の変更点

- `package-token-ids-mixed-request-state-smoke` に `mixed_request_state_layer_batch_step` を追加した。
- prefill/decodeのactive requestを層単位でまとめて実行し、`batching_mode=real`、`prefill_real_batch=true`、`decode_real_batch=true` を出すようにした。
- `request_batch_executor=true` と `fused_request_batch=false` をstdoutとJSONL parserで保存するようにした。
- `prefill_mode=token_id_full_mixed_request_state` を出力し、selected-layer bridgeとの区別を明確にした。
- R9700 AQ4 `manifest-all` smokeを `run-external-benchmark.py --parse ullm-model-loop-throughput` で保存した。

## 結果

- Report: `uLLM-project/benchmarks/results/2026-07-09/package-batch-throughput/phase-t1-mixed-request-state-layer-batch-executor-v1.md`
- JSONL: `uLLM-project/benchmarks/results/2026-07-09/package-batch-throughput/phase-t1-mixed-request-state-layer-batch-executor-v1/results.jsonl`
- R9700 result:
  - `status=ok`
  - `batching.mode=real`
  - `request_batch_executor=true`
  - `fused_request_batch=false`
  - `throughput_row=false`
  - `prefill_real_batch=true`
  - `decode_real_batch=true`
  - `final_top1_tokens=[44370,5446]`
  - `prefill_total_input_tokens_per_second=36.164606`
  - `decode_total_generated_tokens_per_second=81.064816`

## 次の行動

- 次はresident loadと測定区間を分離し、AQ4 batch `1/4/8` のfull mixed request-batch throughput rowを作る。
- この行はfused GPU batchではなく、layer loadも含むため、SQ/AQ4正式速度比較にはまだ使わない。

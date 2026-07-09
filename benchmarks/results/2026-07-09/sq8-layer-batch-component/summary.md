# SQ8_0 Layer Batch Component Result

## 前回の要点

- Qwen3.5 layer3の全projection SQ8_0 artifactは、raw smokeで `sq_fp8_batch_matvec_count=14` and `sq_fp8_expected_all_batch_matvec_count=14` を確認済み。
- この結果はlayer componentの証拠であり、full-package serving throughputではない。

## 今回の変更点

- `tools/run-external-benchmark.py --parse ullm-component-prefill` 経由で、同じsmokeを `inference-benchmark-result-v0.1` JSONLへ保存した。
- Result: `results.jsonl`
- Case: `sq8-qwen35-layer3-full-proj-batch-smoke`
- Status: `ok`
- `batching.mode=real`
- `batching.prefill_real_batch=true`
- `batching.prefill_executor_token_parallelism=2`
- `workload.sq_projection_boundary=batch`
- `workload.sq_projection_implementation_ids=batch=sq8_0_matvec_batch_r9700_direct`
- `workload.sq_fp8_batch_matvec_count=14`
- `workload.sq_fp8_expected_all_batch_matvec_count=14`
- `metrics.prefill_total_input_tokens_per_second=703.558458`
- `memory.vram_consumed_bytes=470949888`

## 次の行動

- full-package real-batch/server-style uLLM runnerへ、このlayer-level direct batch projection boundaryを接続する。
- M10のvLLM+FP8比較では、このcomponent rowをkernel/package connectivity evidenceとして扱い、full-package serving rowとは分ける。

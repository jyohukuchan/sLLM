# SQ8_0 Qwen3-14B Full Mixed Real-Batch No-Final-Logits Smoke

Date: 2026-07-09

Command:

```bash
ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_BATCH_KERNEL=1 target/debug/ullm-engine sq-fp8-token-ids-mixed-request-state-smoke /tmp/ullm-qwen3-14b-fp8-bf16-thin.ullm.d /tmp/ullm-qwen3-14b-fp8-full-sq8-artifact 2 1048576 manifest-all len:16x2 8 0 1024 128 1000000 0
```

Result row: `results.jsonl`

Key fields:

- `status=ok`
- `engine.commit=ce89ec2-dirty-no-final-logits`
- `model.name=Qwen3-14B-FP8`
- `format_id=SQ8_0`
- `batching.mode=real`
- `prefill_real_batch=true`
- `decode_real_batch=true`
- `final_logits_in_total=false`
- `sq_projection_boundary=batch`
- `sq_fp8_batch_matvec_count=6720`
- `sq_fp8_expected_all_batch_matvec_count=6720`
- `prefill_sq_fp8_batch_matvec_count=4480`
- `decode_sq_fp8_batch_matvec_count=2240`
- `sq_diagnostic_host_staging_read_count=0`
- `sq_diagnostic_host_staging_write_count=72`
- `sq_diagnostic_host_staging_read_bytes=0`
- `sq_diagnostic_host_staging_write_bytes=1966080`
- `prefill_total_input_tps=15.417194`
- `decode_total_generated_tps=15.709506`
- `end_to_end_total_tps=15.513415`

Interpretation:

This row confirms that the full 40-layer Qwen3-14B-FP8 mixed request-state path can run the
`pp16/tg8`, two-request real-batch shape with direct SQ8_0 batch matvec coverage for every selected
self-attention projection. `TOP_K=0` skips the final lm_head guard, so `total_wall_ms` excludes final
logits and records only prefill plus decode model-loop work.

It remains a diagnostic model-loop row, not a final vLLM serving comparison. It does, however, remove
the previous final-logits latency caveat from this real-batch SQ8_0 diagnostic class.

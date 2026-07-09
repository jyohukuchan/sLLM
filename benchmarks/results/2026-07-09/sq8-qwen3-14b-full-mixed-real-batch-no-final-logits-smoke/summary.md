# SQ8_0 Qwen3-14B Full Mixed Real-Batch No-Final-Logits Smoke

> **Quarantined:** the uLLM rows used an invalid v0.1 sidecar that omitted source
> `weight_scale_inv`. Keep them as connection diagnostics only. See the
> [quarantine record](../../2026-07-10/sq8-qwen3-14b-invalid-sidecar-quarantine.md).

Date: 2026-07-09

## b2 Command

```bash
ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_BATCH_KERNEL=1 target/debug/ullm-engine sq-fp8-token-ids-mixed-request-state-smoke /tmp/ullm-qwen3-14b-fp8-bf16-thin.ullm.d /tmp/ullm-qwen3-14b-fp8-full-sq8-artifact 2 1048576 manifest-all len:16x2 8 0 1024 128 1000000 0
```

## b4 Command

```bash
ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_BATCH_KERNEL=1 target/debug/ullm-engine sq-fp8-token-ids-mixed-request-state-smoke /tmp/ullm-qwen3-14b-fp8-bf16-thin.ullm.d /tmp/ullm-qwen3-14b-fp8-full-sq8-artifact 2 1048576 manifest-all len:16x4 8 0 1024 128 1000000 0
```

## b8 Command

```bash
ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_BATCH_KERNEL=1 target/debug/ullm-engine sq-fp8-token-ids-mixed-request-state-smoke /tmp/ullm-qwen3-14b-fp8-bf16-thin.ullm.d /tmp/ullm-qwen3-14b-fp8-full-sq8-artifact 2 1048576 manifest-all len:16x8 8 0 1024 128 1000000 0
```

Result rows: `results.jsonl`

## Result

| Case | Requests | Prompt tokens | Generated tokens | Prefill tok/s | Decode tok/s | End-to-end tok/s | Consumed GiB | SQ8 batch matvec | Host read/write |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `qwen3-14b-fp8-sq8-full-mixed-real-batch-no-final-pp16-tg8-b2` | 2 | 32 | 16 | 15.417194 | 15.709506 | 15.513415 | 12.49 | `6720/6720` | `0/72` |
| `qwen3-14b-fp8-sq8-full-mixed-real-batch-no-final-pp16-tg8-b4` | 4 | 64 | 32 | 16.220953 | 16.766274 | 16.398742 | 13.06 | `6720/6720` | `0/120` |
| `qwen3-14b-fp8-sq8-full-mixed-real-batch-no-final-pp16-tg8-b8` | 8 | 128 | 64 | 16.477829 | 16.747149 | 16.566635 | 13.57 | `6720/6720` | `0/216` |

b2 key fields:

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

b4 key fields:

- `status=ok`
- `engine.commit=ee3d9a8`
- `model.name=Qwen3-14B-FP8`
- `format_id=SQ8_0`
- `batching.mode=real`
- `prefill_real_batch=true`
- `decode_real_batch=true`
- `final_logits_in_total=false`
- `sq_projection_boundary=batch`
- `sq_fp8_batch_matvec_count=6720`
- `sq_fp8_expected_all_batch_matvec_count=6720`
- `sq_diagnostic_host_staging_read_count=0`
- `sq_diagnostic_host_staging_write_count=120`
- `prefill_total_input_tps=16.220953`
- `decode_total_generated_tps=16.766274`
- `end_to_end_total_tps=16.398742`

b8 key fields:

- `status=ok`
- `engine.commit=ee3d9a8`
- `model.name=Qwen3-14B-FP8`
- `format_id=SQ8_0`
- `batching.mode=real`
- `prefill_real_batch=true`
- `decode_real_batch=true`
- `final_logits_in_total=false`
- `sq_projection_boundary=batch`
- `sq_fp8_batch_matvec_count=6720`
- `sq_fp8_expected_all_batch_matvec_count=6720`
- `sq_diagnostic_host_staging_read_count=0`
- `sq_diagnostic_host_staging_write_count=216`
- `prefill_total_input_tps=16.477829`
- `decode_total_generated_tps=16.747149`
- `end_to_end_total_tps=16.566635`

Interpretation:

These rows confirm that the full 40-layer Qwen3-14B-FP8 mixed request-state path can run the
`pp16/tg8`, two-request, four-request, and eight-request real-batch shapes with direct SQ8_0 batch
matvec coverage for every selected self-attention projection. `TOP_K=0` skips the final lm_head guard, so
`total_wall_ms` excludes final logits and records only prefill plus decode model-loop work.

They remain diagnostic model-loop rows, not final vLLM serving comparisons. They do, however,
remove the previous final-logits latency caveat from this real-batch SQ8_0 diagnostic class and
advance the planned batch grid through `b2`, `b4`, and `b8`.

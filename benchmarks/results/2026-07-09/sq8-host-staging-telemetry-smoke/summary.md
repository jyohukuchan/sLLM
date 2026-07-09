# SQ8_0 Host Staging Telemetry Smoke

Date: 2026-07-09

Command:

```bash
ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_BATCH_KERNEL=1 target/debug/ullm-engine sq-fp8-token-ids-mixed-request-state-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d /tmp/ullm-sq8-layer3-full-projections-artifact 2 1048576 3 len:2x2 1 1 1024 32 10000000 0
```

Result row: `results.jsonl`

Key fields:

- `status=ok`
- `engine.commit=c7a7673-dirty-host-staging`
- `model.name=Qwen3.5-9B`
- `format_id=SQ8_0`
- `batching.mode=real`
- `prefill_real_batch=true`
- `decode_real_batch=true`
- `sq_projection_boundary=batch`
- `sq_projection_implementation_ids=batch=sq8_0_matvec_batch_r9700_direct`
- `sq_fp8_batch_matvec_count=21`
- `sq_fp8_expected_all_batch_matvec_count=21`
- `prefill_sq_fp8_batch_matvec_count=14`
- `decode_sq_fp8_batch_matvec_count=7`
- `sq_diagnostic_host_staging_read_count=39`
- `sq_diagnostic_host_staging_write_count=48`
- `sq_diagnostic_host_staging_read_bytes=1327104`
- `sq_diagnostic_host_staging_write_bytes=1130496`

Interpretation:

This row proves the new diagnostic host-staging counters are emitted by the SQ8_0 mixed request-state
real-batch path and preserved by `tools/run-external-benchmark.py`.

It is a selected-layer telemetry smoke, not a serving-comparison row. Nonzero staging counts record
the known diagnostic resident path copies that still need to be reduced before treating this class
as serving-equivalent throughput.

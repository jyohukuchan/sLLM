# SQ8_0 Host Staging Reduced Smoke

Date: 2026-07-09

Command:

```bash
ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_BATCH_KERNEL=1 target/debug/ullm-engine sq-fp8-token-ids-mixed-request-state-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d /tmp/ullm-sq8-layer3-full-projections-artifact 2 1048576 3 len:2x2 1 1 1024 32 10000000 0
```

Result row: `results.jsonl`

Key fields:

- `status=ok`
- `engine.commit=984c0cd-dirty-host-staging-reduced`
- `model.name=Qwen3.5-9B`
- `format_id=SQ8_0`
- `batching.mode=real`
- `prefill_real_batch=true`
- `decode_real_batch=true`
- `sq_projection_boundary=batch`
- `sq_fp8_batch_matvec_count=21`
- `sq_fp8_expected_all_batch_matvec_count=21`
- `prefill_sq_fp8_batch_matvec_count=14`
- `decode_sq_fp8_batch_matvec_count=7`
- `sq_diagnostic_host_staging_read_count=33`
- `sq_diagnostic_host_staging_write_count=42`
- `sq_diagnostic_host_staging_read_bytes=1228800`
- `sq_diagnostic_host_staging_write_bytes=1032192`

Interpretation:

This row keeps the selected layer3 SQ8_0 real-batch projection coverage at `21/21` while reducing
diagnostic host staging from the previous telemetry smoke's `39` reads / `48` writes to `33` reads /
`42` writes. The reduction comes from moving the o-projection residual add and post-RMSNorm segment
onto batch device buffers.

It remains a selected-layer diagnostic row. Additional staging remains around batch packing,
per-request projection handoff, MLP activation, and final per-request output handoff.

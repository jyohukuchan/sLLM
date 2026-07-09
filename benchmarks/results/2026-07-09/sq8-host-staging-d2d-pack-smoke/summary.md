# SQ8_0 Host Staging D2D Pack Smoke

Date: 2026-07-09

Command:

```bash
ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_BATCH_KERNEL=1 target/debug/ullm-engine sq-fp8-token-ids-mixed-request-state-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d /tmp/ullm-sq8-layer3-full-projections-artifact 2 1048576 3 len:2x2 1 1 1024 32 10000000 0
```

Result row: `results.jsonl`

Key fields:

- `status=ok`
- `engine.commit=eb96c35-dirty-d2d-pack`
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
- `sq_diagnostic_host_staging_read_count=0`
- `sq_diagnostic_host_staging_write_count=9`
- `sq_diagnostic_host_staging_read_bytes=0`
- `sq_diagnostic_host_staging_write_bytes=196608`

Interpretation:

This row keeps the selected layer3 SQ8_0 real-batch projection coverage at `21/21` while reducing
diagnostic host staging from the previous MLP device smoke's `24` reads / `39` writes to `0` reads /
`9` writes. The reduction comes from replacing host-mediated batch packing and per-request unpacking
with runtime buffer-to-buffer copies.

It remains a selected-layer diagnostic row. The remaining counted writes are residual host inputs
and the batch residual upload for the host-driven smoke path.

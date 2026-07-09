# SQ8_0 Qwen3-14B Full Mixed Real-Batch Device Handoff Smoke

> **Quarantined:** the uLLM row used an invalid v0.1 sidecar that omitted source
> `weight_scale_inv`. Keep this as a connection diagnostic only. See the
> [quarantine record](../../2026-07-10/sq8-qwen3-14b-invalid-sidecar-quarantine.md).

Date: 2026-07-09

Command:

```bash
ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_BATCH_KERNEL=1 target/debug/ullm-engine sq-fp8-token-ids-mixed-request-state-smoke /tmp/ullm-qwen3-14b-fp8-bf16-thin.ullm.d /tmp/ullm-qwen3-14b-fp8-full-sq8-artifact 2 1048576 manifest-all len:1x2 1 1 1024 128 1000000 0
```

Result row: `results.jsonl`

Key fields:

- `status=ok`
- `engine.commit=f116928-dirty-device-handoff`
- `model.name=Qwen3-14B-FP8`
- `format_id=SQ8_0`
- `batching.mode=real`
- `prefill_real_batch=true`
- `decode_real_batch=true`
- `sq_projection_boundary=batch`
- `sq_fp8_batch_matvec_count=560`
- `sq_fp8_expected_all_batch_matvec_count=560`
- `prefill_sq_fp8_batch_matvec_count=280`
- `decode_sq_fp8_batch_matvec_count=280`
- `sq_diagnostic_host_staging_read_count=0`
- `sq_diagnostic_host_staging_write_count=6`
- `sq_diagnostic_host_staging_read_bytes=0`
- `sq_diagnostic_host_staging_write_bytes=163840`
- `final_logits_in_total=true`

Interpretation:

This row confirms that the full 40-layer Qwen3-14B-FP8 mixed request-state path still reaches
`560/560` direct SQ8_0 batch matvec coverage after the D2D device residual handoff change. It removes
the measured layer-to-layer residual host read staging that remained in the earlier full D2D-pack
diagnostic row.

It remains a diagnostic model-loop row, not a final serving-comparison row. Final logits still
dominate total latency, and the remaining counted host writes come from the smoke path's initial
host-side residual inputs.

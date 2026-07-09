# SQ8_0 Qwen3-14B Full Mixed Request-State Real-Batch Smoke

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
- `engine.commit=3fa2bf3`
- `model.name=Qwen3-14B-FP8`
- `batching.mode=real`
- `prefill_real_batch=true`
- `decode_real_batch=true`
- `sq_execution_mode=direct_fp8_dequant_matvec`
- `sq_projection_boundary=batch`
- `sq_projection_implementation_ids=batch=sq8_0_matvec_batch_r9700_direct`
- `sq_fp8_tensor_count=281`
- `sq_fp8_batch_matvec_count=560`
- `sq_fp8_expected_all_batch_matvec_count=560`
- `prefill_sq_fp8_batch_matvec_count=280`
- `decode_sq_fp8_batch_matvec_count=280`
- `sq_fp8_single_matvec_count=0`
- `layers_csv=0..39`

Interpretation:

This is the first saved Qwen3-14B-FP8 full 40-layer mixed request-state SQ8_0 row with real
request-batch projection counters. It uses the BF16-only thin package plus full SQ8_0 sidecar
artifact, and it proves the full model-loop path can execute all selected self-attention projections
through direct SQ8_0 batch matvec boundaries.

It is not yet a final serving-comparison row. Final logits are included and dominate total latency,
and the current resident batch path still uses diagnostic host staging around some batch projection
boundaries.

# SQ8_0 Resident Stack All-Projection Batch Diagnostic

Date: 2026-07-09

Command:

```bash
ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_BATCH_KERNEL=1 target/debug/ullm-engine sq-fp8-package-self-attn-stack-batch-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d /tmp/ullm-sq8-layer3-full-projections-artifact 2 1048576 3 len:2x2 1 1 1024 32 10000000 0
```

Result row: `results.jsonl`

Key fields:

- `status=ok`
- `engine.commit=28c2bf0`
- `batching.mode=real`
- `prefill_real_batch=true`
- `decode_real_batch=true`
- `mixed_request_state_real_batch_projection_used=true`
- `sq_execution_mode=direct_fp8_dequant_matvec`
- `sq_projection_boundary=batch`
- `sq_projection_implementation_ids=batch=sq8_0_matvec_batch_r9700_direct`
- `sq_fp8_batch_matvec_count=21`
- `sq_fp8_expected_all_batch_matvec_count=21`
- `prefill_sq_fp8_batch_matvec_count=14`
- `decode_sq_fp8_batch_matvec_count=7`
- `sq_fp8_single_matvec_count=0`
- `sq_fp8_triple_matvec_count=0`

Interpretation:

The resident mixed request-state stack now routes all seven selected self-attention projections
through direct SQ8_0 batch matvec boundaries for both prefill and decode. This closes the selected
layer3 projection-boundary diagnostic. It is still not a final full-package or server-style
throughput row because the current implementation uses host staging around the batch projection
boundaries and only exercises the selected layer3 artifact.

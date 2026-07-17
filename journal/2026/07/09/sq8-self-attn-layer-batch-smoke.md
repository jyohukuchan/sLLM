# SQ8_0 self-attn layer batch smoke

- Added `sq-fp8-package-self-attn-layer-batch-smoke` as a thin SQ8_0 wrapper around the existing
  self-attention layer batch smoke.
- Added SQ8_0 resident `row_f32` support so sampled batch projection verification can read SQ8_0
  rows without materializing the full matrix.
- Added phase-local mixed-request-state batch projection counters:
  `prefill_sq_fp8_batch_matvec_count`, `decode_sq_fp8_batch_matvec_count`, and
  `mixed_request_state_real_batch_projection_used`.
- R9700 smoke:
  `target/debug/ullm-engine sq-fp8-package-self-attn-layer-batch-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d /tmp/ullm-sq-fp8-qkv-layers3-7-11-15-19-q8-k16-v16-plus-layer23-k16-policy-v0.1-artifact 2 1048576 3 len:2 1 256 32 10000000 0`
- Result: `verified=true`, `real_batch=true`, `sq_projection_boundary=batch`,
  `sq_fp8_batch_matvec_count=6`, `sq_fp8_expected_all_batch_matvec_count=14`, and
  `sq_projection_implementation_ids=batch=sq8_0_matvec_batch_r9700_direct`.
- Interpretation: this proves the SQ8_0 batch matvec runtime path is callable through a layer batch
  smoke. It is not a final M10 serving row because the artifact is a partial q/k/v selected-layer
  overlay and not a full-package all-projection SQ8_0 artifact.

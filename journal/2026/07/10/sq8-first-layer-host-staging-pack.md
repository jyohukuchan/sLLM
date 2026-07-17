# SQ8 First-Layer Host Staging Pack

- Reviewed the subagent change in `PackageSelfAttnResidentStepBatchLayer::step_batch_from_host_to_device`.
- For `items.len() > 1`, the mixed request-state path now packs the per-request residuals into `batch_residual_buffer` with one host-to-device copy per timestep.
- Each request layer input is then populated with `copy_from_buffer` from the packed residual buffer, so the previous per-request host staging writes are removed.
- Removed the later duplicate residual upload before the o-projection residual add; that stage reuses the already packed batch residual buffer.
- Verified with `cargo build -p ullm-engine`.
- Verified with `cargo test -p ullm-engine -- package_model_loop_cli_tail_tests::infer_mixed_request_state_real_batch_flags_enables_both_phases_when_batch_matvec_used`.
- Verified with `git diff --check -- ':!README.md'`.
- Verified the short layer3 smoke:
  - command: `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_BATCH_KERNEL=1 target/debug/ullm-engine sq-fp8-token-ids-mixed-request-state-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d /tmp/ullm-sq8-layer3-full-projections-artifact 2 1048576 3 len:2x2 1 1 1024 32 10000000 0`
  - result: `sq_diagnostic_host_staging_read_count=0`, `sq_diagnostic_host_staging_write_count=3`, `sq_diagnostic_host_staging_write_bytes=98304`

# SQ8_0 resident stack q/k/v batch

## Summary

- Split the resident self-attention step into input RMSNorm, q/k/v projection, and post-q/k/v tail.
- Added mixed request-state self-attention batch helpers for host-input and device-input paths.
- Routed multi-request self-attention steps through `PackageAq4ResidentMatvec::matvec_batch` for q/k/v.
- Kept o/gate/up/down on existing direct single projection boundaries for now.

## Verification

- `cargo fmt --check -p ullm-engine`
- `cargo check -p ullm-engine`
- `cargo build -p ullm-engine`
- `python3 -m unittest tests.test_external_benchmark_batch_parser`
- `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_BATCH_KERNEL=1 target/debug/ullm-engine sq-fp8-package-self-attn-stack-batch-smoke ...`

## Result

- Commit for implementation benchmark: `9f5fe83`
- Saved row: `benchmarks/results/2026-07-09/sq8-stack-resident-qkv-batch/results.jsonl`
- `batching_mode=real`
- `prefill_real_batch=true`
- `decode_real_batch=true`
- `sq_projection_boundary=single+batch`
- `sq_fp8_batch_matvec_count=9`
- `sq_fp8_expected_all_batch_matvec_count=21`
- `prefill_sq_fp8_batch_matvec_count=6`
- `decode_sq_fp8_batch_matvec_count=3`

## Remaining

- Batch o/gate/up/down projection boundaries for the resident self-attention stack.
- Reduce host staging in the current q/k/v batch helper once a runtime device-to-device slice copy or packed RMSNorm path is available.
- Move from selected-layer stack diagnostics to full-package or server-style rows before vLLM FP8 comparison.

## Follow-up: all selected projections batched

- Commit `28c2bf0` extends the resident self-attention batch path to o/gate/up/down as well.
- Saved row: `benchmarks/results/2026-07-09/sq8-stack-resident-all-batch/results.jsonl`
- `batching_mode=real`
- `prefill_real_batch=true`
- `decode_real_batch=true`
- `sq_projection_boundary=batch`
- `sq_fp8_batch_matvec_count=21`
- `sq_fp8_expected_all_batch_matvec_count=21`
- `prefill_sq_fp8_batch_matvec_count=14`
- `decode_sq_fp8_batch_matvec_count=7`
- `sq_fp8_single_matvec_count=0`

Remaining after this follow-up:

- Reduce host staging around resident batch projection boundaries.
- Promote the same execution mode into full-package or server-style rows before treating it as vLLM-comparable throughput.

## Follow-up: full 40-layer Qwen3-14B mixed request-state row

- Saved row:
  `benchmarks/results/2026-07-09/sq8-qwen3-14b-full-mixed-real-batch-smoke/results.jsonl`
- Model/package: `/tmp/ullm-qwen3-14b-fp8-bf16-thin.ullm.d` plus
  `/tmp/ullm-qwen3-14b-fp8-full-sq8-artifact`
- `batching_mode=real`
- `prefill_real_batch=true`
- `decode_real_batch=true`
- `sq_projection_boundary=batch`
- `sq_fp8_batch_matvec_count=560`
- `sq_fp8_expected_all_batch_matvec_count=560`
- `prefill_sq_fp8_batch_matvec_count=280`
- `decode_sq_fp8_batch_matvec_count=280`
- `sq_fp8_single_matvec_count=0`

This proves the full 40-layer model-loop mixed request-state path can use direct SQ8_0 batch matvec
for every selected self-attention projection. It remains separate from final serving comparison
because final logits dominate the saved row and the resident batch path still uses host staging.

## Follow-up: host staging telemetry

- Added SQ8_0 diagnostic host-staging counters to the mixed request-state stdout and external
  benchmark parser.
- Saved row:
  `benchmarks/results/2026-07-09/sq8-host-staging-telemetry-smoke/results.jsonl`
- Short layer3 telemetry smoke:
  - `sq_fp8_batch_matvec_count=21`
  - `sq_fp8_expected_all_batch_matvec_count=21`
  - `sq_diagnostic_host_staging_read_count=39`
  - `sq_diagnostic_host_staging_write_count=48`
  - `sq_diagnostic_host_staging_read_bytes=1327104`
  - `sq_diagnostic_host_staging_write_bytes=1130496`

The counters make the host-staging caveat machine-readable. They do not close the serving
comparison blocker; the next step is to reduce these counted copies or add a separate
serving-equivalent path.

## Follow-up: first host staging reduction

- Added a batch residual buffer to `PackageSelfAttnResidentStepBatchLayer`.
- Moved the o-projection residual add to `add_f32` on batch device buffers.
- Moved post-RMSNorm to `segmented_rmsnorm_f32` on the batch attention block output.
- Saved row:
  `benchmarks/results/2026-07-09/sq8-host-staging-reduced-smoke/results.jsonl`
- The same layer3 selected-layer shape stayed at `sq_fp8_batch_matvec_count=21/21`.
- Host staging counters moved from:
  - read count `39` to `33`
  - write count `48` to `42`
  - read bytes `1327104` to `1228800`
  - write bytes `1130496` to `1032192`

Remaining host staging is still present around batch packing, per-request projection handoff, MLP
activation, and final per-request output handoff.

## Follow-up: MLP device-side host staging reduction

- Added a separate batch MLP up buffer so gate and up projections can both stay resident before
  activation.
- Moved MLP gate/up SiLU-mul to `silu_mul_f32` on batch device buffers.
- Moved the MLP down residual add to `add_f32` on batch device buffers.
- Saved row:
  `benchmarks/results/2026-07-09/sq8-host-staging-mlp-residual-device-smoke/results.jsonl`
- The same layer3 selected-layer shape stayed at `sq_fp8_batch_matvec_count=21/21`.
- Host staging counters moved from the first reduced row:
  - read count `33` to `24`
  - write count `42` to `39`
  - read bytes `1228800` to `540672`
  - write bytes `1032192` to `737280`

Remaining host staging is now concentrated around batch packing, per-request projection handoff, and
final per-request layer output handoff. The next larger reduction likely needs runtime support for
device-to-device slice copy, gather, or scatter rather than more CPU-side glue removal.

## Follow-up: D2D batch pack host staging reduction

- Added runtime buffer-to-buffer copy support through `ullm_runtime_buffer_copy` and
  `RuntimeBuffer::copy_from_buffer`.
- Switched the SQ8_0 resident batch path to device-to-device copies for:
  - per-request input-normed slices into the batch input buffer
  - batch q/k/v projection slices into per-request buffers
  - per-request attention projection inputs into the batch o-projection input buffer
  - final batch layer output slices into per-request layer output buffers
- Saved row:
  `benchmarks/results/2026-07-09/sq8-host-staging-d2d-pack-smoke/results.jsonl`
- The same layer3 selected-layer shape stayed at `sq_fp8_batch_matvec_count=21/21`.
- Host staging counters moved from the MLP device row:
  - read count `24` to `0`
  - write count `39` to `9`
  - read bytes `540672` to `0`
  - write bytes `737280` to `196608`

Remaining counted host writes are the host residual inputs and the batch residual upload in the
host-driven smoke path. The next comparison blocker is less about this selected-layer staging
diagnostic and more about promoting the same resident direct-batch path into full-package or
server-style measurements for the planned vLLM+FP8 comparison.

## Follow-up: full 40-layer D2D pack diagnostic

- Ran the same D2D batch-pack code on the full Qwen3-14B-FP8 mixed request-state path.
- Saved row:
  `benchmarks/results/2026-07-09/sq8-qwen3-14b-full-mixed-real-batch-d2d-pack-smoke/results.jsonl`
- The full 40-layer row stayed at `sq_fp8_batch_matvec_count=560/560`.
- Host staging counters were:
  - read count `156`
  - write count `240`
  - read bytes `3194880`
  - write bytes `6553600`
- Interpretation: the selected-layer batch pack/unpack host staging is effectively removed, but
  the full stack still uses `step_batch_from_device_to_device`, which reads previous-layer residuals
  back to host before calling the host-driven batch helper. The next implementation step is a true
  device-to-device stack handoff path.

## Follow-up: full 40-layer device residual handoff diagnostic

- Changed `PackageSelfAttnResidentStepBatchLayer::step_batch_from_device_to_device` to keep
  previous-layer residuals on device for multi-request self-attention batches.
- The single-item path now delegates directly to `step_from_device_to_device`.
- The multi-item path copies residual, input-normed, q/k/v projected, attention projection input,
  and final layer-output slices with runtime buffer-to-buffer copies instead of host staging.
- Removed the now-unused SQ diagnostic host read staging helpers.
- Saved row:
  `benchmarks/results/2026-07-09/sq8-qwen3-14b-full-mixed-real-batch-device-handoff-smoke/results.jsonl`
- The full 40-layer row stayed at `sq_fp8_batch_matvec_count=560/560`.
- Host staging counters moved from the full D2D-pack row:
  - read count `156` to `0`
  - write count `240` to `6`
  - read bytes `3194880` to `0`
  - write bytes `6553600` to `163840`
- The remaining counted writes are the smoke path's initial host-side residual inputs. This makes
  the full mixed request-state row a better diagnostic baseline for the later vLLM+FP8 comparison.

## Follow-up: no-final-logits real-batch diagnostic

- Allowed `TOP_K=0` for the mixed request-state CLI family to skip the final lm_head guard.
- The output now reports `final_logits_in_total=false`, `final_lm_head_guard=false`,
  `lm_head_top_k=0`, and `final_logits_wall_ms=0.000000` for that mode.
- Saved row:
  `benchmarks/results/2026-07-09/sq8-qwen3-14b-full-mixed-real-batch-no-final-logits-smoke/results.jsonl`
- Workload: Qwen3-14B-FP8, `manifest-all`, `prompt_tokens=16x2`, `generated_tokens=8x2`,
  `rotary_dim=128`, `rope_base=1000000`, `TOP_K=0`.
- The row records `batching_mode=real`, `prefill_real_batch=true`, `decode_real_batch=true`, and
  `sq_fp8_batch_matvec_count=6720/6720`.
- Host staging counters are read `0`, write `72`, read bytes `0`, write bytes `1966080`.
- Throughput without final logits: prefill `15.417194` tok/s, decode `15.709506` tok/s,
  end-to-end `15.513415` tok/s.
- This is still a CLI model-loop diagnostic, but it removes the final-logits latency caveat from
  the real-batch SQ8_0 row class.

## Follow-up: matched vLLM b2 baseline

- Added a vLLM FP8 row with the same smoke shape as the uLLM real-batch no-final-logits diagnostic:
  `prompt_tokens=16x2`, `generated_tokens=8x2`, `concurrent_requests=2`.
- Saved row:
  `benchmarks/results/2026-07-09/sq8-vllm-fp8-comparison/results.jsonl`
- `case_id=vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b2-tp1-rocr`
- vLLM metrics: prefill `34.41438620647337` tok/s, decode `17.21` tok/s, total `51.62` tok/s.
- Consumed VRAM: `21007855616` bytes.
- This gives a same-shape external baseline for the uLLM b2 row, but the harnesses still differ:
  vLLM uses `vllm bench throughput`; uLLM uses the current CLI model-loop diagnostic.

## Follow-up: b4/b8 batch-grid smoke pairs

- Added a second uLLM no-final-logits real-batch row:
  `qwen3-14b-fp8-sq8-full-mixed-real-batch-no-final-pp16-tg8-b4`.
- Workload: Qwen3-14B-FP8, `manifest-all`, `prompt_tokens=16x4`, `generated_tokens=8x4`,
  `rotary_dim=128`, `rope_base=1000000`, `TOP_K=0`.
- The row records `batching_mode=real`, `prefill_real_batch=true`, `decode_real_batch=true`, and
  `sq_fp8_batch_matvec_count=6720/6720`.
- Host staging counters are read `0`, write `120`, read bytes `0`, write bytes `3932160`.
- uLLM throughput without final logits: prefill `16.220953` tok/s, decode `16.766274` tok/s,
  end-to-end `16.398742` tok/s.
- Added the matching vLLM FP8 row:
  `vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b4-tp1-rocr`.
- vLLM metrics: prefill `135.04146895989985` tok/s, decode `67.52` tok/s,
  total `202.56` tok/s.
- Consumed VRAM: `30121553920` bytes.
- Added a third uLLM no-final-logits real-batch row:
  `qwen3-14b-fp8-sq8-full-mixed-real-batch-no-final-pp16-tg8-b8`.
- Workload: Qwen3-14B-FP8, `manifest-all`, `prompt_tokens=16x8`, `generated_tokens=8x8`,
  `rotary_dim=128`, `rope_base=1000000`, `TOP_K=0`.
- The b8 row records `batching_mode=real`, `prefill_real_batch=true`, `decode_real_batch=true`,
  and `sq_fp8_batch_matvec_count=6720/6720`.
- Host staging counters are read `0`, write `216`, read bytes `0`, write bytes `7864320`.
- uLLM b8 throughput without final logits: prefill `16.477829` tok/s, decode `16.747149` tok/s,
  end-to-end `16.566635` tok/s.
- Added the matching vLLM FP8 row:
  `vllm-r9700-qwen3-14b-fp8-smoke-pp16-tg8-b8-tp1-rocr`.
- vLLM b8 metrics: prefill `236.01404374447745` tok/s, decode `118.01` tok/s,
  total `354.02` tok/s.
- Consumed VRAM: `30121566208` bytes.
- The batch grid now has b2, b4, and b8 shapes for both uLLM and vLLM, still as separate harness
  classes rather than final serving parity.

## Follow-up: batch-grid summary helper

- Added `tools/summarize-sq8-vllm-batch-grid.py` to regenerate a compact Markdown table from one
  or more JSONL result files.
- The helper supports `--workload-prefix pp16-tg8` and `--requests 2,4,8`, which reproduces the
  current uLLM/vLLM b2/b4/b8 rows without carrying the historical b1 connectivity rows into the
  compact batch-grid table.
- Added `tests/test_summarize_sq8_vllm_batch_grid.py` for filtering, requests parsing, and invalid
  JSON line-number reporting.

## Follow-up: harness normalization metadata

- Added a `harness` object to new `tools/run-external-benchmark.py` rows.
- `vllm-throughput` is classified as `serving_throughput_benchmark` with
  `harness_type=vllm_bench_throughput_cli`, `serving_parity_candidate=true`, and
  `includes_http_server=false`.
- `ullm-model-loop-throughput` is classified as `cli_model_loop_diagnostic` with
  `serving_parity_candidate=false`.
- `ullm-package-batch-throughput` is classified as `cli_logical_batch_diagnostic` with
  `serving_parity_candidate=false`.
- The compact SQ8/vLLM batch-grid helper now emits a `Harness` column and infers the legacy b2/b4/b8
  row classes when the older rows do not yet contain the new `harness` object.

## Follow-up: serving parity gate

- Added `--require-serving-parity` to `tools/summarize-sq8-vllm-batch-grid.py`.
- The gate exits with status `2` when the selected table has no rows, includes
  `serving_parity_candidate=false`, or mixes different `harness.class` values.
- The current `pp16/tg8 --requests 2,4,8` compact table fails this gate by design because uLLM rows
  are `cli_model_loop_diagnostic` and vLLM rows are `serving_throughput_benchmark`.

## Follow-up: harness-class filter

- Added `--harness-class` to `tools/summarize-sq8-vllm-batch-grid.py`.
- The filter is applied before `--require-serving-parity`, so a serving-only slice can be checked
  without carrying CLI diagnostic rows into the gate.
- The current serving-only slice passes the gate only because it contains the vLLM
  `serving_throughput_benchmark` rows; it is a sanity check, not a uLLM-vs-vLLM final comparison.

## Follow-up: required-engine gate

- Added `--require-engines` to `tools/summarize-sq8-vllm-batch-grid.py`.
- Final comparison gate commands can now require both `uLLM` and `vLLM` after all filters are
  applied.
- The current serving-only vLLM slice now fails the stricter final-comparison gate with missing
  `uLLM`, which is expected until a uLLM serving-parity row exists.

## Follow-up: per-request engine-grid gate

- Added `--require-engine-grid` to `tools/summarize-sq8-vllm-batch-grid.py`.
- When combined with `--require-engines uLLM,vLLM`, each requested concurrency bucket must contain
  both engines after filtering.
- The current serving-only vLLM slice now reports missing `uLLM` independently for b2, b4, and b8.

## Follow-up: projection kernel family telemetry

- Added `sq_projection_kernel_families` next to `sq_projection_implementation_ids` in SQ8_0 stdout
  rows.
- `tools/run-external-benchmark.py` now preserves the field in `workload`.
- Current rows report `direct` for executed single/batch/pair/triple matvec boundaries; future
  non-direct or fused C++ kernel families can be distinguished without parsing implementation IDs.

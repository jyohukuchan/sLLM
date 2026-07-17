# 6h20m golden prefix drift localization plan

## Progress

- Created `uLLM-project/docs/plans/6h20m-golden-prefix-drift-localization-plan-v0.1.md`.
- The plan uses the previous measured completion time, `1139s`, as the baseline. `1139s * 20 = 22780s`, approximately `6h19m40s`.
- Confirmed `uLLM-project/journal/` is absent and the outer `ultimateLLM/journal/` tree is the active journal location.
- Reviewed the previous prefix plan, current `prefix0-8` fixture metadata, and `0..8` result JSONL.
- A subagent independently recommended narrowing the next task to layer 4 origin isolation through single-layer and window validation.
- Started executing the plan.
- Confirmed `uLLM-project/journal/` is absent before implementation.
- Baseline checks passed:
  - `cargo fmt --all --check`
  - `cargo check -p ullm-engine`
- Confirmed `qwen35-9b-prefix0-8-seq8` fixture is `prefix`, range `0..8`, `sequence_len=8`, `hidden_size=4096`, with before/after payloads for layers `0..7`.
- Spawned two `gpt-5.3-codex-spark` medium workers:
  - Worker A: Rust `package-golden-prefix-smoke` drift localization mode and input drift JSONL fields.
  - Worker B: `tools/analyze-golden-prefix-drift.py` summary script.
- Worker A completed the Rust implementation:
  - added optional `RUN_MODE` positional argument after `REPORT_PATH`;
  - default mode is `actual_prefix`;
  - added `golden_before_each_layer`;
  - added per-layer input drift metrics and previews to JSONL;
  - preserved existing output metric fields.
- Parent review added structured `runtime_metrics` extraction from linear-attention `runtime_line`, keeping the original string field.
- Worker B added `tools/analyze-golden-prefix-drift.py`.
- Parent review adjusted backend comparison to use `device/backend`, so R9700 and V620 remain distinct if both are analyzed.
- Static checks after integration passed:
  - `cargo fmt --all --check`
  - `cargo check -p ullm-engine`
  - `python3 -m py_compile tools/analyze-golden-prefix-drift.py`
- Targeted checks passed:
  - `cargo test -p ullm-engine golden -- --test-threads=1`
  - `cargo build -p ullm-engine`
  - `cargo test -p ullm-engine -- --test-threads=1`
  - `git diff --check`
- Final broad check passed:
  - `cargo test --workspace -- --test-threads=1`
- Required result artifact audit passed:
  - 14 `.txt` smoke outputs
  - 14 `.jsonl` smoke reports
  - 1 summary JSON
  - 1 summary Markdown
- Journal placement audit passed: `uLLM-project/journal/` is absent and this outer journal file exists.
- Backward compatibility smoke without explicit `RUN_MODE` passed and defaulted to `run_mode=actual_prefix`.
- Required `seq8` validation matrix completed for CPU `0` and R9700 `2`.
- Analysis artifacts:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/golden-prefix-drift-summary-seq8.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/golden-prefix-drift-summary-seq8.md`

## Summary

The next useful task is not another broad prefix run. It is to add drift localization around `package-golden-prefix-smoke` so each layer row records both:

- input drift: current input vs fixture `layer_N_before`
- output drift: actual layer output vs fixture `layer_N_after`

The required comparison is:

- `actual_prefix`: pass actual output from one layer to the next.
- `golden_before_each_layer`: reset each layer input to the fixture before hidden.

This separates prefix accumulation from layer-local drift.

## Current Evidence

- `0..4 seq8` stayed in `possible_quantization_error` on CPU/R9700/V620.
- `0..8 seq8` changed to `numeric_drift` from layer 4 onward on CPU/R9700.
- CPU and R9700 metrics stayed closely aligned, so backend divergence is currently unlikely.

## Results

Range-level summary:

| run | device | max MSE | min cosine | classes |
| --- | --- | ---: | ---: | --- |
| `actual_prefix 0..8` | CPU `0` | 0.264669271684 | -0.083518201 | `numeric_drift`, `possible_quantization_error` |
| `actual_prefix 0..8` | R9700 `2` | 0.264669283836 | -0.083517839 | `numeric_drift`, `possible_quantization_error` |
| `actual_prefix 4..8` | CPU `0` | 0.156752182543 | 0.631421521 | `numeric_drift`, `possible_quantization_error` |
| `actual_prefix 4..8` | R9700 `2` | 0.156752184204 | 0.631421580 | `numeric_drift`, `possible_quantization_error` |
| `golden_before_each_layer 4..8` | CPU `0` | 0.087692939574 | 0.818975685 | `possible_quantization_error` |
| `golden_before_each_layer 4..8` | R9700 `2` | 0.087692939801 | 0.818975686 | `possible_quantization_error` |

Single-layer windows from fixture before hidden:

| layer | CPU MSE | CPU cosine | R9700 MSE | R9700 cosine | class |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4 | 0.029671504005 | 0.826768764 | 0.029671497036 | 0.826768820 | `possible_quantization_error` |
| 5 | 0.021098504164 | 0.909955437 | 0.021098508846 | 0.909955410 | `possible_quantization_error` |
| 6 | 0.087692939574 | 0.818975685 | 0.087692939801 | 0.818975686 | `possible_quantization_error` |
| 7 | 0.022232840408 | 0.943961009 | 0.022232836801 | 0.943961018 | `possible_quantization_error` |

Analysis summary:

- first bad layer: layer 4 in `actual_prefix 0..8`.
- largest output MSE: layer 6 in `actual_prefix 0..8`.
- largest CPU/R9700 delta in the required matrix: output MSE delta `3.627e-08` at layer 4, `actual_prefix 0..8`.
- `actual_prefix 4..8` vs `golden_before_each_layer 4..8` CPU output MSE deltas:
  - layer 4: `0.0`
  - layer 5: `0.025127534867`
  - layer 6: `0.069059242969`
  - layer 7: `0.109231856202`

Structured linear-attention detail is now present as `runtime_metrics` beside `runtime_line`. For layer 4 and 6, internal runtime self-check diffs remain around `1e-6` or lower while HF golden output drift is much larger.

## Interpretation

- The drift is not backend-specific. CPU and R9700 remain closely aligned across every required run.
- Layer 4 is the first layer where `actual_prefix 0..8` becomes `numeric_drift`, but layer 4 by itself from fixture before hidden is only `possible_quantization_error`.
- Resetting every layer to fixture before hidden turns all layers `4..7` back into `possible_quantization_error`.
- The strongest current explanation is prefix input drift accumulation before and through layer 4, not a single isolated backend/runtime failure at layer 4.
- The next useful work is to inspect how layer `0..3` output drift changes the hidden distribution handed into layer 4, then decide whether AQ policy or prefix execution needs adjustment.

Optional seq16 spot check was not run. The required seq8 matrix already separates prefix accumulation from layer-local drift, and running seq16 would require an additional HF export/load cycle without changing the immediate next decision.

## Next

Next likely plan: add a compact hidden-distribution report for layer `0..4` boundaries, especially norm, mean, variance, top absolute coordinates, and per-token drift, then compare that against AQ family/tensor contribution around linear-attention layer 4.

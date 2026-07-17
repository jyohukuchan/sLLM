# 0900 project progress report

## Goal

日本時間 2026-07-05 09:00 までプロジェクトをできるだけ進め、09:30 までに報告書を作成する。

## Current Executive Summary

Status as of `2026-07-05 07:58 JST`:

- The issue is still worth debugging, but the target is now narrower than at the start.
- The remaining drift is not backend-specific noise:
  - CPU/R9700/V620 had already aligned earlier.
  - Current work repeatedly localizes the error to hidden `3994` row/cell paths.
- Best full-prefix result on the original token ids `1..16`:
  - baseline with existing manifest layer6/layer10 row-scale: `0.645338058`
  - current best smoke: `0.610977173`
  - current best ingredients: layer6 `mlp.down_proj.weight[3994]` row-scale + layer8 QKV V845 single-cell source-restore.
- The strongest local compensation candidate is layer6 `mlp.down_proj.weight[3994]` row-scale:
  - original fixture all-token scale: `1.026471714`, row-dot RMSE `0.117735388 -> 0.063680278`
  - tokens101-116 fixture all-token scale: `1.023383096`, row-dot RMSE `0.131756300 -> 0.061972585`
  - it partially generalizes to a genuinely different token fixture: `1.080525398 -> 1.043153763`
  - tokens201-216 shows mixed behavior: layer6/layer7 improve strongly, but final max worsens slightly `1.140727997 -> 1.145284653`
  - it has been validated as package manifest metadata, not only as a CLI smoke override.
- The layer8 QKV V845 cell is useful on the original fixture but should remain smoke-only for now:
  - original fixture single-cell qkv source-restore improved `0.645338058 -> 0.627647400`
  - combined with layer6 row-scale it reached `0.610977173`
  - tokens101-116 did not improve overall max and worsened layer11 versus layer6 row-scale alone.
- Layer10 MLP remains a real local-error area, but the two concrete probes tested so far are rejected:
  - gate single-cell source-restore worsened `0.610977173 -> 0.625913620`
  - `mlp.down_proj.weight[3994]` row-scale worsened `0.610977173 -> 0.616283417`
- Tooling improved:
  - `tools/export-qwen-layer-module-trace.py` now emits `row_dot.<projection>.scale_fit` under schema `qwen-layer-module-trace-v0.10`.
- Multi-fixture summary artifact:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-row-scale-multi-fixture-summary.md`

## Starting State

- Start check time: `2026-07-05 01:07:36 JST`.
- Repo state at start: `main...origin/main [ahead 7]`, clean.
- Journal placement check passed:
  - `uLLM-project/journal/` is absent.
  - outer `ultimateLLM/journal/` is active.
- Latest committed work:
  - `1175835 Add golden prefix drift localization`
  - `486fc28 Add golden prefix drift localization plan`

## Work Plan

- Continue debugging the golden prefix drift.
- Add hidden distribution diagnostics around layer `0..4` and `0..8` boundaries.
- Save machine-readable results under `uLLM-project/benchmarks/results/2026-07-05/engine/`.
- Produce a final report here by 09:30 JST.

## Progress

- Started hidden distribution diagnostic work.
- Spawned a `gpt-5.3-codex-spark` medium worker for the Python hidden-stats analyzer.
- Parent is implementing Rust-side distribution fields in `package-golden-prefix-smoke` JSONL.
- Added Rust-side hidden distribution fields to `package-golden-prefix-smoke` JSONL:
  - `input_distribution`
  - `output_distribution`
  - actual/expected/diff stats
  - per-token metrics
  - max and top-8 absolute-diff locations
- Added `tools/analyze-golden-prefix-hidden-stats.py`.
- Verified:
  - `cargo fmt --all --check`
  - `cargo check -p ullm-engine`
  - `cargo build -p ullm-engine`
  - `python3 -m py_compile tools/analyze-golden-prefix-hidden-stats.py`
- Re-ran hidden-stats validation for CPU `0` and R9700 `2`:
  - `actual_prefix 0..8`
  - `actual_prefix 4..8`
  - `golden_before_each_layer 4..8`
- Wrote artifacts:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/golden-prefix-hidden-stats-summary-seq8.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/golden-prefix-hidden-stats-summary-seq8.md`

## Hidden Stats Findings

- Distribution-based worst layer is layer 6 in `actual_prefix 0..8`.
  - R9700 output diff RMS: `0.514460186833`
  - output max abs diff: `44.618328094482`
  - per-token max MSE: `1.038341557630`
- CPU/R9700 remain aligned; the backend is still unlikely to be the primary cause.
- The top absolute-diff hidden coordinate is strongly concentrated:
  - hidden `3994`: `218` appearances in top-8 locations
  - hidden `3842`: `10`
  - hidden `3456`: `10`
- CPU `actual_prefix 0..8` output diff RMS progression:
  - layer 0: `0.085192724442`
  - layer 1: `0.112214351788`
  - layer 2: `0.161177817002`
  - layer 3: `0.189996750761`
  - layer 4: `0.292772739509`
  - layer 5: `0.331036812938`
  - layer 6: `0.514460175022`
  - layer 7: `0.488121008826`
- The max absolute diff is already at hidden `3994` from layer 0, then remains dominated by that coordinate through most of the prefix.

## Current Interpretation

- The problem is still worth debugging before full generation.
- It now looks less like broad random quantization noise and more like a concentrated hidden-channel outlier that gets amplified through the prefix.
- The next useful debug target is hidden channel `3994`, especially how layer 0 creates that channel drift and how layers 4-6 amplify it.

## Hot Channel Row Error Check

- User asked whether this is still worth debugging. Answer: yes, but narrowly.
- Added `tools/analyze-package-row-quant-error.py` to compare selected package AQ4 rows against source safetensors rows without materializing full tensors.
- Ran hidden row `3994` against `linear_attn.out_proj.weight` and `mlp.down_proj.weight` for language layers `0..31`.
- Wrote artifacts:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/golden-prefix-hot-channel-row-error-summary-layers0-31.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/golden-prefix-hot-channel-row-error-summary-layers0-31.md`
- Result:
  - inspected rows: `56`
  - row `3994` matched the tensor-wide manifest max absolute error in `55/56` rows
  - all `24/24` available `linear_attn.out_proj.weight` rows matched
  - `31/32` available `mlp.down_proj.weight` rows matched
  - worst row RMS: layer 0 `linear_attn.out_proj.weight`, `0.012744330947`
  - worst row max abs error: layer 0 `linear_attn.out_proj.weight`, `0.062869846821`

## Updated Interpretation

- Row `3994` is not just where activation drift appears; it is also almost always the row that contains the largest per-tensor AQ4 reconstruction error for the direct output projections.
- This does not look like package corruption: reconstructed row max errors match manifest metrics.
- The likely issue is an outlier hidden channel that is individually quantized correctly but poorly tolerated by the end-to-end prefix computation.
- Next debug target should be module-level contribution around hidden `3994`, not more backend CPU/R9700 comparison.

## Module Contribution Check

- Added `module_contribution` to `package-golden-prefix-smoke` JSONL rows.
  - Compares `actual_delta = actual_after - actual_before` with `expected_delta = golden_after - golden_before`.
  - Records attention output, attention block output, post norm, MLP output, residual identity error, and per-token trace for the max output-diff hidden coordinate.
- Added `tools/analyze-golden-prefix-module-contribution.py`.
- Ran CPU module contribution checks:
  - `actual_prefix 0..8`
  - `golden_before_each_layer 4..8`
- Wrote artifacts:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-module-contribution-cpu-actual-prefix0-8-seq8.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-module-contribution-cpu-actual-prefix0-8-seq8.txt`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-module-contribution-cpu-golden-before4-8-seq8.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-module-contribution-cpu-golden-before4-8-seq8.txt`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/golden-prefix-module-contribution-summary-seq8.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/golden-prefix-module-contribution-summary-seq8.md`
- Summary:
  - rows: `12`
  - hot hidden counts: `3994 = 12/12`
  - dominant actual module: `mlp = 9`, `attention = 3`
  - failure shape counts:
    - `missing_expected_delta = 5`
    - `spurious_actual_delta = 5`
    - `opposite_delta = 1`
    - `mixed_delta = 1`
- Worst row:
  - layer 6 `actual_prefix`, hidden `3994`, token `0`
  - output diff: `-44.618324279785`
  - input diff: `-18.225698471069`
  - delta diff: `-26.392627716064`
  - expected delta: `25.25`
  - actual delta: `-1.142626762390`
  - actual attention output: `0.007871641777`
  - actual MLP output: `-1.150498151779`

## Revised Debug Interpretation

- This is definitely still a real debug target.
- The failure is not primarily CPU vs R9700 and not a generic random drift.
- Hidden `3994` is the stable hot coordinate across row reconstruction, output diff, and module contribution.
- Layer 4/5 often produce a large negative MLP-side update where golden expects a small positive update.
- Layer 6 is more severe: golden expects a large positive update at hidden `3994`, while the package path produces almost no positive update.
- The next technical target should be layer 6 MLP/full-precision comparison for hidden `3994`, ideally by comparing the full-precision MLP down-row dot product against the AQ4 package path for the same post-norm/activation vector.

## Verification

- `cargo fmt --all --check`: passed
- `cargo check -p ullm-engine`: passed
- `cargo test -p ullm-engine golden -- --test-threads=1`: passed
- `python3 -m py_compile tools/analyze-golden-prefix-module-contribution.py tools/analyze-package-row-quant-error.py tools/analyze-golden-prefix-hidden-stats.py`: passed
- `git diff --check`: passed

## Full-Reference Layer Trace

- Added `tools/export-qwen-layer-module-trace.py`.
  - Instantiates the Qwen model on `meta`.
  - Loads only selected decoder-layer safetensors.
  - Runs golden `before` tensors through the selected layer.
  - Captures full-reference `linear_attn`, `post_attention_layernorm`, and `mlp` outputs.
- Added `tools/compare-qwen-module-trace.py`.
- Ran full-reference traces for linear-attention layers `4,5,6`, hidden `3994`.
- Wrote artifacts:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer-module-trace-fullref-layers4-6-hidden3994.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer-module-trace-fullref-layers4-6-hidden3994.md`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-module-trace-comparison-golden-before4-6-hidden3994.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-module-trace-comparison-golden-before4-6-hidden3994.md`
- Fixture match quality:
  - layer 4 max abs diff: `0.00390625`
  - layer 5 max abs diff: `0.0078125`
  - layer 6 max abs diff: `0.0078125`
- Full-reference vs package, golden-before mode:
  - layer 4 token 7 hidden `3994`:
    - full attention: `2.34375`, package attention: `-1.360840559`
    - full MLP: `-0.5234375`, package MLP: `-6.447978020`
    - delta error: `-9.683818817`
  - layer 5 token 6 hidden `3994`:
    - full attention: `2.8125`, package attention: `0.110342495`
    - full MLP: `-1.109375`, package MLP: `-5.617842197`
    - delta error: `-7.194999695`
  - layer 6 token 0 hidden `3994`:
    - full attention: `10.875`, package attention: `-0.013602152`
    - full MLP: `14.5`, package MLP: `0.116954692`
    - delta error: `-25.146647453`

## Current Root-Cause Shape

- Layer 6 is now the sharpest target.
- The missing full-reference update is not isolated to just MLP or attention:
  - full-reference attention contributes `+10.875`
  - full-reference MLP contributes `+14.5`
  - package contributes nearly zero for both at the hot coordinate
- This points upstream of only `down_proj` row reconstruction. The likely problem is that the AQ4 path changes the internal activation vector(s) feeding both `linear_attn.out_proj` and MLP enough that the hidden `3994` outlier update disappears.
- Next best test: compare package internal full vectors against full-reference internals for layer 6, especially attention projection input/core output and MLP activated vector, or add targeted row-dot diagnostics for `out_proj[3994]` and `down_proj[3994]`.

## Row-Dot Isolation

- Extended `tools/export-qwen-layer-module-trace.py` with `--package-dir`.
  - Captures the exact full-reference input to `linear_attn.out_proj` and `mlp.down_proj`.
  - Computes source-row dot and package-AQ4-row dot for hidden row `3994`.
- Extended `tools/compare-qwen-module-trace.py` with row-only and activation-path error columns.
- Updated artifacts:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer-module-trace-fullref-layers4-6-hidden3994.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer-module-trace-fullref-layers4-6-hidden3994.md`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-module-trace-comparison-golden-before4-6-hidden3994.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-module-trace-comparison-golden-before4-6-hidden3994.md`
- Key row-dot result in golden-before mode:
  - layer 4:
    - attention row-only error: `+0.0349335`
    - attention activation-path error: `-3.7410085`
    - MLP row-only error: `-0.0048215`
    - MLP activation-path error: `-5.9208342`
  - layer 5:
    - attention row-only error: `-0.0064717`
    - attention activation-path error: `-2.6932795`
    - MLP row-only error: `+0.0154356`
    - MLP activation-path error: `-4.5272307`
  - layer 6:
    - attention row-only error: `+0.0258382`
    - attention activation-path error: `-10.9017638`
    - MLP row-only error: `-0.3727856`

## Coordinate Chain Analyzer

- Added `tools/analyze-package-golden-prefix-coordinate-chain.py`.
  - Extracts one token/hidden coordinate from `package-golden-prefix-smoke`
    JSONL when that coordinate is available in `per_token_hot_hidden_trace`.
  - Emits JSON and Markdown chain views with layer input diff, delta diff,
    output diff, attention output, MLP output, coordinate availability, and
    output max location.
- Generated artifacts:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-coordinate-chain-layer10-actual-h3456-t0-p4p46-inproj.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-coordinate-chain-layer10-actual-h3456-t0-p4p46-inproj.md`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-coordinate-chain-layer6-layer10-actual-h3456-t0-p4p46-inproj.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-coordinate-chain-layer6-layer10-actual-h3456-t0-p4p46-inproj.md`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-coordinate-chain-layer6-layer10-actual-h3994-t11-p4p46-inproj.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-coordinate-chain-layer6-layer10-actual-h3994-t11-p4p46-inproj.md`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-coordinate-chain-p4p46-inproj-summary.md`
- Key result:
  - layer10-only metadata leaves token `0` hidden `3456` at layer11 output
    diff `-0.911422729`.
  - layer6+layer10 metadata removes hidden `3456` from the hot-coordinate
    trace, but token `11` hidden `3994` reaches layer11 output diff
    `-0.891334534`.
- Interpretation:
  - The layer6+layer10 metadata successfully addressed the earlier hidden
    `3456` chain.
  - The remaining hidden `3994` issue looks like input-distribution-sensitive
    propagation through layers `7`, `8`, `9`, and `11`, not a single final-row
    scale problem.

## Actual-Input Full-Reference Split

- Added an optional final `package-golden-prefix-smoke` argument:
  `[INPUT_DUMP_DIR]`.
  - Writes `layer-XXXX-input.f32` and `layer-XXXX-input.json` for the actual
    tensor sent into each layer.
  - Keeps the JSONL report compact and lets Python full-reference tools replay
    the exact same layer input.
- Extended `tools/export-qwen-layer-module-trace.py` with
  `--input-override-dir`.
  - When `layer-XXXX-input.f32` exists, it replaces the fixture `before`
    tensor for that layer.
- Fixed `tools/compare-qwen-module-trace.py` so self-attention
  `self_attention_o_proj` row-dot traces are used as the attention fallback.
- Generated actual-input artifacts:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix0-12-manifest-row-scale-layer6-layer10-p4p46-inproj-input-dump.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer-module-trace-actual-input-layers7-9-11-hidden3994-layer6-layer10-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer-module-trace-actual-input-layers7-9-11-hidden3994-layer6-layer10-p4p46-inproj.md`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-module-trace-comparison-actual-input-layers7-9-11-hidden3994-layer6-layer10-p4p46-inproj.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-module-trace-comparison-actual-input-layers7-9-11-hidden3994-layer6-layer10-p4p46-inproj.md`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-actual-input-layer7-9-11-hidden3994-layer6-layer10-p4p46-inproj-summary.md`
- Key split:
  - Layer `11`, token `6`, hidden `3994`:
    - package output diff: `-0.889577866`
    - package-vs-full-reference local delta error: `-0.014577866`
    - conclusion: layer `11` mostly inherits the error.
  - Layer `7`, token `8`, hidden `3994`:
    - package output diff: `-0.756856918`
    - package-vs-full-reference local delta error: `-0.631856918`
    - attention row-only error: `0.006025539`
    - attention activation-path error: `-1.085749765`
    - conclusion: layer `7` self-attention activation path is the strongest
      local target.
- Next target:
  - Added self-attention stage aliases to
    `tools/export-qwen-layer-module-trace.py` so package/full-reference
    comparisons include:
    - `attention_q_query`
    - `attention_q_gate`
    - `attention_k_projected`
    - `attention_v_projected`
    - `attention_q_normed`
    - `attention_k_normed`
  - Re-generated the actual-input trace and comparison artifacts.
  - Layer `7`, token `8`, feature `503`:
    - `attention_input_normed` diff: `0.000284731`
    - `attention_q_query` diff: `0.091293752`
    - `attention_q_gate` diff: `0.043898165`
    - `attention_q_normed` diff: `0.106798269`
    - `attention_projection_input` diff: `0.499136567`
  - Updated interpretation:
    - layer input after RMSNorm is effectively aligned.
    - q/gate differences exist, but the largest jump is visible at the gated
      `o_proj` input.
    - The next dump should expose the raw pre-gate causal attention vector for
      layer `7` token `8`, feature `503`.

## Layer 7 Feature 503 Self-Attention Replay

- Extended `tools/analyze-qwen-self-attention-propagation.py`.
  - Added `--input-override-dir`.
  - Added `--token-index` and `--feature-index` per-feature stage tracing.
  - The replay now can emit query, gate, key, value, q/k normed, q/k rope,
    raw attention, gate sigmoid, and gated `o_input`.
- Generated artifacts:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-self-attention-propagation-layer7-actual-input-token8-feature503-hidden3994-p4p46-inproj.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-self-attention-propagation-layer7-actual-input-token8-feature503-hidden3994-p4p46-inproj.md`
- Key feature trace for layer `7`, token `8`, feature `503`:
  - query projection diff: `0.088807434`
  - gate projection diff: `0.042518735`
  - raw attention diff after dequantized package q/k/v replay: `-0.012199044`
  - gate sigmoid diff: `0.009531260`
  - gated `o_input` diff: `0.000905693`
- Important contrast:
  - The actual package JSONL reported `attention_projection_input` feature
    `503` as `1.128042817`, while full-reference reported `0.628906250`.
  - The dequantized package q/k/v replay reports `0.629811943`, almost matching
    source.
- Updated interpretation:
  - q/k/v row dequantization by itself does not reproduce the actual runtime
    feature `503` jump.
  - The next useful diagnostic is Rust-side: compare actual layer `7` package
    runtime causal attention against `runtime_host_causal_attn_f32`, or dump the
    full prepared q/k/v and raw attention vectors from the package path.
    - MLP activation-path error: `-13.9850410`

## Updated Root-Cause Shape

- The AQ4 rows are not the primary cause at hidden `3994`.
- With full-reference activations, package AQ4 rows reproduce the hot coordinate closely enough:
  - layer 6 attention row-only error is only `+0.0258382`
  - layer 6 MLP row-only error is `-0.3727856`
- The real loss is in the vectors feeding those rows:
  - layer 6 attention activation-path error is `-10.9017638`
  - layer 6 MLP activation-path error is `-13.9850410`
- Next best test: compare package and full-reference internal activation vectors before `out_proj` and `down_proj`, starting at layer 6 token 0. The likely failing point is earlier than the final projection rows.

## Row-Dot Verification

- `python3 -m py_compile tools/export-qwen-layer-module-trace.py tools/compare-qwen-module-trace.py tools/analyze-golden-prefix-module-contribution.py tools/analyze-package-row-quant-error.py tools/analyze-golden-prefix-hidden-stats.py`: passed
- `git diff --check`: passed

## Hot Input Vector Trace

- Extended `package-golden-prefix-smoke` module contribution output with `hot_input_vectors`.
  - Records the input vector at the same token that owns the max output diff.
  - For linear-attention layers, also records intermediate vectors:
    - `attention_input_normed`
    - `attention_qkv_projection`
    - `attention_z_projection`
    - `attention_a_projection`
    - `attention_b_projection`
    - `attention_conv`
    - `attention_gate`
    - `attention_beta`
    - `attention_recurrent`
    - `attention_normed`
    - `attention_projection_input`
    - `mlp_activation`
- Extended `tools/export-qwen-layer-module-trace.py` to emit matching full-reference hot vector summaries.
- Extended `tools/compare-qwen-module-trace.py` to compare package/full-reference hot vector stats and top absolute features.

## Implementation Differences Found

- Found a Qwen3.5 RMSNorm convention mismatch.
  - Reference Qwen3.5 RMSNorm uses `output * (1.0 + weight)`.
  - uLLM was passing raw `input_layernorm`, `post_attention_layernorm`, `q_norm`, and `k_norm` weights directly.
  - Added `effective_rmsnorm_weight_values()` to convert additive RMSNorm weights while leaving direct weights such as `linear_attn.norm.weight` unchanged.
- Found a depthwise conv1d kernel orientation mismatch.
  - PyTorch `Conv1d(padding=kernel_size-1)` uses the last kernel element for the current token.
  - uLLM host/runtime helpers were treating the first kernel element as current-token weight.
  - Updated CPU host, HIP kernel source, Rust expected helper, and smoke-side expected helper to use the causal Conv1d weight order.

## Effective RMSNorm / Conv Results

- Re-ran `package-golden-prefix-smoke` in `golden_before_each_layer` mode for layers `4..6`.
  - Artifact: `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-hot-input-vectors-cpu-golden-before4-7-seq8.jsonl`
  - Result:
    - `max_mse=0.063285328750`
    - `max_mean_abs_diff=0.107925192`
    - `max_abs_diff=21.987182617`
    - `min_cosine_similarity=0.940971567`
    - `verified=true`
- Re-ran `package-golden-prefix-smoke` in `golden_before_each_layer` mode for layers `0..7`.
  - Artifact: `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-golden-before0-8-effective-rmsnorm.jsonl`
  - Result:
    - `max_mse=0.075359231910`
    - `max_mean_abs_diff=0.168298812`
    - `max_abs_diff=21.987182617`
    - `min_cosine_similarity=0.708968308`
    - `verified=true`
- The previous catastrophic collapse at layer 6 hidden `3994` was largely explained by raw RMSNorm weight handling.
- The remaining largest local mismatch shifted to:
  - layer 0 hidden `3994`: `max_abs_diff=14.15096664428711`, cosine `0.708968308`
  - layer 6 hidden `3456`: `max_abs_diff=21.987182617`, cosine `0.960706`

## Remaining Debug Target

- This is still worth debugging, but the target has changed.
- Already resolved:
  - Qwen3.5 additive RMSNorm handling
  - causal Conv1d weight order
  - final AQ4 row-only error as primary explanation for the hidden-channel collapse
- Still open:
  - small projection errors in linear attention appear to be amplified through `conv/gate/beta/recurrent/norm`.
  - The layer 6 hidden `3456` comparison shows close early projections but a much larger difference by `attention_projection_input`.
- Current best next target:
  - compare full-vector or per-head linear attention recurrent state for layer 6 token 0 hidden `3456`
  - separate quantization sensitivity from any remaining layout or recurrence formula mismatch

## Layer 0 / Gated RMSNorm Follow-up

- Added full-reference trace for layer `0`, hidden `3994`.
  - Artifact: `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer-module-trace-fullref-layer0-hidden3994.jsonl`
  - Comparison: `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-module-trace-comparison-hot-input-vectors-golden-before0-hidden3994.md`
- Extended comparison markdown with all common hot-vector stage errors.
- Confirmed Qwen3.5 reference semantics from local `transformers`:
  - `Qwen3_5GatedDeltaNet` reshapes recurrent output to `[-1, head_v_dim]`.
  - `Qwen3_5RMSNormGated.forward(hidden_states, gate)` normalizes per `head_v_dim`, then multiplies by `F.silu(gate)`.
  - The output of `linear_attn.norm` is already the gated `out_proj` input.
- Layer 0 token `3` hidden `3994`:
  - package output diff: `14.151`
  - attention row-only error: `0.0236746`
  - attention activation-path error: `2.45946`
  - MLP row-only error: `-0.0051128`
  - MLP activation-path error: `11.6909`
  - attention recurrent max-abs stage error: `-0.0991468`
  - attention projection input max-abs stage error: `4.25088`
  - MLP activation max-abs stage error: `28.3077`
- Layer 6 token `0` hidden `3456`:
  - attention recurrent max-abs stage error: `0.0652994`
  - attention projection input max-abs stage error: `57.3472`
  - MLP activation max-abs stage error: `9.023`
- Interpretation:
  - The remaining error is still a real debug target.
  - It is now localized to linear attention post-recurrent gated RMSNorm sensitivity and its downstream MLP amplification.
  - Final `out_proj` / `down_proj` row reconstruction remains a secondary contributor.

## Internal Projection Row-Dot Check

- Extended the full-reference exporter with selected projection row-dot diagnostics for hot-token top features.
  - `attention_qkv_projection`
  - `attention_z_projection`
  - `attention_a_projection`
  - `attention_b_projection`
- The selected rows use the full-reference `attention_input_normed` vector and compare source safetensors rows with package AQ4 rows.
- Worst selected package-vs-source row-dot errors:
  - layer 0 hidden `3994`:
    - `attention_qkv_projection`: `-0.398449211`
    - `attention_z_projection`: `+0.439037216`
    - `attention_a_projection`: `-0.326066728`
    - `attention_b_projection`: `-0.109638577`
  - layer 6 hidden `3456`:
    - `attention_qkv_projection`: `-0.310529786`
    - `attention_z_projection`: `+0.168087556`
    - `attention_a_projection`: `-0.209607230`
    - `attention_b_projection`: `-0.124259242`
- Interpretation:
  - Internal projection AQ4 row errors are visible but still small in direct row-dot terms.
  - The large final hidden error is more likely from recurrent/gated-normalization sensitivity to those small projection perturbations than from a single large projection-row reconstruction bug.

## In-Projection Policy Comparison

- Tried to run `p4p65-inproj` as a full golden-prefix smoke, but the available direct package lacks passthrough RMSNorm tensors and cannot run the full package smoke as-is.
- Used the full-reference exporter row-dot path instead, which only needs AQ4 projection rows.
- Compared layer 6 hidden `3456` selected projection row-dot errors across:
  - current `p4p6`
  - `p4p65-inproj`
  - `p4p46-inproj`
- Artifact:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-projection-rowdot-policy-comparison-layer6-hidden3456.md`
- Result:
  - `p4p65-inproj` improved the selected qkv worst row-dot error from about `0.31` to about `0.22`.
  - `p4p46-inproj` left qkv at about `0.31` and made selected z worst error about `0.20`.
  - a/b selected row-dot errors stayed around `0.13..0.21`.
- Interpretation:
  - In-projection policy changes matter, especially qkv group sizing.
  - They do not by themselves prove the full prefix drift will disappear, because the dominant failure still appears after recurrent/gated RMSNorm amplification.
  - A complete policy experiment needs a fullpkg package with passthrough tensors for the alternative in-projection policies.

## p4p65 Fullpkg Golden-Prefix Experiment

- Built a fullpkg p4p65-inproj package with passthrough tensors:
  - package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p65-inproj-reservoir65536-jobs4.ullm.d`
  - summary: `uLLM-project/benchmarks/results/2026-07-05/engine/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p65-inproj-reservoir65536-jobs4.json`
  - time: `8:03.07`
  - max RSS: `338564 KiB`
  - quantized tensors: `255`
  - passthrough tensors: `520`
  - codebooks: `12`
  - total file bytes: `9150134127`
  - convert failure count: `0`
- Separate verify completed with exit status `0`.
  - verify time file: `uLLM-project/benchmarks/results/2026-07-05/engine/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p65-inproj-reservoir65536-jobs4-verify.time`
  - verified passthrough tensor count: `520`
  - verified passthrough payload bytes: `5049777120`
- Ran golden-before prefix smoke `0..8`.
  - artifact: `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-golden-before0-8-p4p65-inproj-effective-rmsnorm.jsonl`
  - `max_mse=0.077991208741`
  - `max_mean_abs_diff=0.172711058`
  - `max_abs_diff=22.264991760`
  - `min_cosine_similarity=0.704322037`
  - `verified=true`
- Comparison with p4p6:
  - p4p6: `max_mse=0.075359231910`, `max_abs_diff=21.987182617`, `min_cosine_similarity=0.708968308`
  - p4p65-inproj is slightly worse on this golden-prefix fixture.
  - Layer 1/4/5 improve slightly, but layer 0/2/6 worsen.
  - Layer 6 hidden `3456` gets worse from `21.9872` to `22.265`.
- Hot-vector comparison for layer 6:
  - p4p6 attention projection input max-abs stage error: `57.3472`
  - p4p65 attention projection input max-abs stage error: `57.8818`
  - p4p65 does not reduce the gated RMSNorm/post-recurrent amplification for this case.
- Interpretation:
  - The earlier selected qkv row-dot improvement does not translate to end-to-end layer improvement.
  - The issue is not solved by simply promoting the p4p65 in-projection set.
  - The next likely target is a more local sensitivity test around recurrent/gated RMSNorm, using controlled perturbations rather than whole-policy swaps.

## p4p46 Fullpkg Golden-Prefix Experiment

- Built a fullpkg p4p46-inproj package with passthrough tensors:
  - package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-reservoir65536-jobs4.ullm.d`
  - summary: `uLLM-project/benchmarks/results/2026-07-05/engine/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-reservoir65536-jobs4.json`
  - quantized tensors: `255`
  - passthrough tensors: `520`
  - codebooks: `12`
  - total file bytes: `9122609002`
  - convert failure count: `0`
- Separate verify completed with exit status `0`.
  - verify time file: `uLLM-project/benchmarks/results/2026-07-05/engine/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-reservoir65536-jobs4-verify.time`
  - verified passthrough tensor count: `520`
  - verified passthrough payload bytes: `5049777120`
- Ran golden-before prefix smoke `0..8`.
  - artifact: `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-golden-before0-8-p4p46-inproj-effective-rmsnorm.jsonl`
  - `max_mse=0.074200270813`
  - `max_mean_abs_diff=0.168852029`
  - `max_abs_diff=22.632156372`
  - `min_cosine_similarity=0.710565711`
  - `verified=true`
- Comparison:
  - p4p46 improves global MSE and minimum cosine over p4p6, but worsens the largest absolute outlier.
  - layer 0 improves from `14.151` to `13.9422`.
  - layer 4 improves from `1.52773` to `1.46616`.
  - layer 5 improves from `3.39466` to `3.35409`.
  - layer 6 worsens from `21.9872` to `22.6322`.
- Layer 6 hot-vector comparison:
  - qkv max-abs stage error improves from `-0.301666` to `-0.228619`.
  - z max-abs stage error improves from `-0.111973` to `-0.076355`.
  - recurrent max-abs stage error improves from `0.0652994` to `0.0618228`.
  - attention projection input max-abs stage error worsens from `57.3472` to `58.9123`.
- Interpretation:
  - p4p46 is better than p4p65 on broad layer metrics for this fixture, but still fails the layer 6 outlier.
  - Improving projection/recurrent scalar errors can still worsen the final gated vector because the normalization/gating is direction-sensitive.
  - The next debug task should not be another whole-policy swap; it should isolate the per-head gated RMSNorm denominator, gate value, and top feature direction around layer 6 token 0 feature `2656`.

## Sampled Gated RMSNorm / Head RMS Diagnostic

- Extended the package and full-reference hot-vector traces with `sampled_features`.
  - The top absolute `attention_projection_input` feature indices are now sampled across other hidden-dim stages.
  - This keeps the same feature coordinate visible even when that coordinate is not in another stage's own top-abs list.
- Added derived stage values:
  - package: `attention_gate_silu` and `attention_pre_gate_normed`
  - full-reference: `attention_gate_silu` and reconstructed `attention_pre_gate_normed`
- Added per-sampled-feature head stats for hidden-dim vectors.
  - For Qwen3.5 linear attention, the sampled feature's `group_index`, `group_offset`, group RMS, and group max_abs are now preserved in comparison JSON.
- Artifacts:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-golden-before6-7-sampled-gated-rmsnorm-p4p6.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-golden-before6-7-sampled-gated-rmsnorm-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-module-trace-comparison-sampled-gated-rmsnorm-layer6-hidden3456-p4p6.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-module-trace-comparison-sampled-gated-rmsnorm-layer6-hidden3456-p4p46-inproj.json`
- Layer 6 token `0`, sampled feature `2656` maps to head `20`, offset `96`.
- p4p6 values at feature `2656`:
  - `attention_recurrent`: package `-0.0044211084`, fullref `-0.0009040833`, diff `-0.0035170252`
  - recurrent head RMS: package `0.0007543053`, fullref `0.0001630136`, diff `0.0005912917`
  - `attention_pre_gate_normed`: package `-3.3520663`, fullref `-0.8474286`, diff `-2.5046377`
  - pre-gate head RMS: package `0.5549241`, fullref `0.1482038`, diff `0.4067203`
  - `attention_gate_silu`: package `22.888027`, fullref `23.0`, diff `-0.1119728`
  - `attention_projection_input`: package `-76.72218`, fullref `-19.375`, diff `-57.34718`
- p4p46 values at feature `2656`:
  - `attention_recurrent`: package `-0.0045273574`, fullref `-0.0009040833`, diff `-0.0036232742`
  - recurrent head RMS: package `0.0007649103`, fullref `0.0001630136`, diff `0.0006018968`
  - `attention_pre_gate_normed`: package `-3.4151342`, fullref `-0.8474286`, diff `-2.5677056`
  - pre-gate head RMS: package `0.5595467`, fullref `0.1482038`, diff `0.4113429`
  - `attention_gate_silu`: package `22.923645`, fullref `23.0`, diff `-0.0763550`
  - `attention_projection_input`: package `-78.28732`, fullref `-19.375`, diff `-58.91232`
- Interpretation:
  - The main amplification happens before the final `out_proj`, inside the head-wise gated RMSNorm boundary.
  - `silu(z)` is not the primary source for feature `2656`; its difference is only about `0.08..0.11`.
  - The recurrent head energy is already about `4.6x` too large for head `20`, and head-wise RMSNorm turns that small recurrent vector mismatch into a pre-gate difference of about `2.5`.
  - Multiplication by the large positive gate near `23` expands that pre-gate difference to the observed `57..59` post-gate mismatch.
  - p4p46 improves some projection-level scalar errors, but worsens this head-local recurrent energy and therefore worsens the final outlier.
- Next debug target:
  - inspect the value-head `20` recurrent inputs across q/k/v/gate/beta, especially value vector and recurrent state update terms, rather than trying another whole-policy in-projection swap first.

## Linear Attention Conv Activation Fix

- Re-checked the reference Qwen3.5 linear attention path after the gated RMSNorm diagnostic.
  - `linear_attn.conv1d` is followed by `silu`.
  - q/k/v split and recurrent input should consume the post-SiLU conv output.
  - uLLM `depthwise_conv1d_f32` remains a pure depthwise convolution kernel; the missing activation was in the package linear-attention smoke/workflow caller path.
- Updated package linear-attention paths so recurrent q/k/v split uses `runtime_host_silu_f32(&conv_output)`.
- Split trace stages into:
  - `attention_conv_pre_silu`
  - `attention_conv` (post-SiLU)
  - `attention_recurrent_q`
  - `attention_recurrent_k`
  - `attention_recurrent_v`
- This supersedes the previous interpretation that the layer 6 giant outlier was intrinsic gated RMSNorm sensitivity.
  - The gated RMSNorm boundary amplified the mismatch, but the upstream cause was using pre-SiLU conv output for recurrent q/k/v.

## Conv Activation Results

- Layer 6 only, `golden_before_each_layer`, p4p6:
  - artifact: `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-golden-before6-7-sampled-gated-rmsnorm-p4p6.jsonl`
  - `max_mse=0.000511560667`
  - `max_mean_abs_diff=0.016369533`
  - `max_abs_diff=0.645427704`
  - `min_cosine_similarity=0.998966216`
- Layer 6 only, `golden_before_each_layer`, p4p46-inproj:
  - artifact: `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-golden-before6-7-sampled-gated-rmsnorm-p4p46-inproj.jsonl`
  - `max_mse=0.000487887468`
  - `max_mean_abs_diff=0.016121546`
  - `max_abs_diff=0.486274719`
  - `min_cosine_similarity=0.999005169`
- Full `0..8`, `golden_before_each_layer`, after Conv+SiLU fix:
  - p4p6: `max_mse=0.012856537339`, `max_mean_abs_diff=0.065956056`, `max_abs_diff=6.994463444`, `min_cosine_similarity=0.903654392`
  - p4p46-inproj: `max_mse=0.012861041333`, `max_mean_abs_diff=0.065597419`, `max_abs_diff=6.988537312`, `min_cosine_similarity=0.903646913`
  - p4p65-inproj: `max_mse=0.012856537339`, `max_mean_abs_diff=0.065956056`, `max_abs_diff=6.994463444`, `min_cosine_similarity=0.903654392`
- Layer 6 internal check after the fix:
  - p4p6 feature `2656` projection input: package `-19.608695984`, fullref `-19.375`, diff `-0.233695984`
  - p4p46 feature `2656` projection input: package `-19.823699951`, fullref `-19.375`, diff `-0.448699951`
  - recurrent feature `2656` is now close to full-reference:
    - p4p6 diff `-0.000010137`
    - p4p46 diff `-0.000018725`
- Interpretation:
  - The layer 6 `21..22` max-abs outlier is fixed.
  - p4p46 is now slightly better than p4p6 on layer 6, but the global `0..8` limit is no longer layer 6.
  - The remaining dominant error is shared across policies at self-attention layer 3, hidden `3994`, token `3`:
    - p4p6 max abs `6.994463444`
    - p4p46 max abs `6.988537312`
    - p4p65 max abs `6.994463444`

## Updated Next Target

- Continue debugging, but switch focus from linear attention layer 6 to self-attention layer 3.
- The current dominant failure shape is:
  - `golden_before_each_layer` input to layer 3 is already close (`input_max_abs_diff` around `0.12..0.16`).
  - layer 3 self-attention/MLP module output creates a hidden `3994` max output diff around `6.99`.
  - the same outlier appears across p4p6, p4p46-inproj, and p4p65-inproj, so another in-projection policy swap is unlikely to be the next best step.
- Next best test:
  - add a self-attention module trace for layer 3 token `3` hidden `3994`;
  - compare q/k/v projection, q/k head RMSNorm, RoPE, causal attention, output gate, o-projection, post RMSNorm, and MLP activation against full-reference internals.

## Conv Activation Verification

- `cargo fmt --all --check`: passed
- `cargo check -p ullm-engine`: passed
- `cargo test -p ullm-runtime-sys depthwise_conv1d -- --test-threads=1`: passed
- `python3 -m py_compile tools/export-qwen-layer-module-trace.py tools/compare-qwen-module-trace.py`: passed
- `git diff --check`: passed

## Self-Attention Input RMSNorm Fix

- While preparing the layer 3 self-attention trace, found another direct implementation mismatch.
  - Reference Qwen3.5 decoder layer applies `input_layernorm` before self-attention q/k/v projection.
  - The package golden-prefix self-attention branch was passing raw layer input to `qwen3_self_attn_prepare_sequence_for_paged_decode_f32`.
  - Residual add still needs the original unnormalized layer input, so the fix keeps two sequences:
    - normalized `attention_input_normed` for q/k/v projection;
    - original `layer_input_for_delta` for decoder residual add.
- Added self-attention hot-vector diagnostics for:
  - `attention_input_normed`
  - `attention_q_query`
  - `attention_k_projected`
  - `attention_v_projected`
  - `attention_q_normed`
  - `attention_k_normed`
  - `attention_q_rope`
  - `attention_k_rope`
  - `attention_q_gate`
  - `attention_output`
- Layer 3 only, p4p6, `golden_before_each_layer`:
  - before fix: `max_abs_diff=6.994463444`, `min_cosine_similarity=0.903654392`
  - after fix: `max_abs_diff=0.302194595`, `min_cosine_similarity=0.999100887`

## Self-Attention Input RMSNorm Results

- Full `0..8`, `golden_before_each_layer`, after both Conv+SiLU and self-attention input RMSNorm fixes:
  - p4p6: `max_mse=0.000511560667`, `max_mean_abs_diff=0.016369533`, `max_abs_diff=0.645427704`, `min_cosine_similarity=0.998966216`
  - p4p46-inproj: `max_mse=0.000487887468`, `max_mean_abs_diff=0.016121546`, `max_abs_diff=0.486274719`, `min_cosine_similarity=0.999005169`
  - p4p65-inproj: `max_mse=0.000489076887`, `max_mean_abs_diff=0.015966775`, `max_abs_diff=0.612869263`, `min_cosine_similarity=0.999012379`
- Full `0..8`, `actual_prefix`, after both fixes:
  - p4p6: `max_mse=0.002535515858`, `max_mean_abs_diff=0.037895676`, `max_abs_diff=0.894840240`, `min_cosine_similarity=0.993748165`
  - p4p46-inproj: `max_mse=0.001895285663`, `max_mean_abs_diff=0.033220127`, `max_abs_diff=0.665708542`, `min_cosine_similarity=0.995342680`
  - p4p65-inproj: `max_mse=0.002149916914`, `max_mean_abs_diff=0.035086083`, `max_abs_diff=0.828275681`, `min_cosine_similarity=0.994710240`
- Updated interpretation:
  - The two dominant outliers were both implementation mismatches:
    - linear attention Conv1d post-SiLU activation before recurrent q/k/v split;
    - self-attention input RMSNorm before q/k/v projection.
  - p4p46-inproj is now the best tested full package policy on this prefix fixture.
  - The remaining `actual_prefix 0..8` drift is sub-1.0 max abs and appears consistent with accumulated quantization error around hot channels, not an obvious missing layer primitive.

## Self-Attention Fix Verification

- `cargo fmt --all --check`: passed
- `cargo check -p ullm-engine`: passed
- `package-golden-prefix-smoke` layer `3..4`, p4p6, `golden_before_each_layer`: passed
- `package-golden-prefix-smoke` layers `0..8`, p4p6/p4p46/p4p65, `golden_before_each_layer`: passed
- `package-golden-prefix-smoke` layers `0..8`, p4p6/p4p46/p4p65, `actual_prefix`: passed

## Self-Attention Model-Loop Impact Check

- Checked whether the self-attention input RMSNorm fix was only local to `package-golden-prefix-smoke`.
- Found that `Qwen3PackageDecoderLayerRuntime` did not keep `input_layernorm.weight`, so package self-attention model-loop preparation could still build q/k/v sequences from unnormalized residuals while remaining internally self-consistent.
- Updated `Qwen3PackageDecoderLayerRuntime` and `Qwen3PackageModelRuntime` to load and validate `input_norm`.
- Updated `qwen3_self_attn_prepare_model_loop_sequence_smoke`:
  - applies input RMSNorm before q/k/v projection preparation;
  - keeps the original residual sequence for decoder residual add;
  - exposes `input_norm_tensors` and `input_norm_dtypes` in model-loop smoke output.
- Verification:
  - `cargo fmt --all --check`: passed
  - `cargo check -p ullm-engine`: passed
  - `cargo test -p ullm-engine qwen3_loader -- --test-threads=1`: passed
  - `package-self-attn-mlp-block-model-loop-smoke` on CPU with p4p46 `[3,7]`, sequence length `3`: passed, runtime/cache diffs all `0`

## HIP Backend Verification

- Re-ran p4p46-inproj golden-prefix validation on R9700 device `2` and V620 device `1` after the Conv+SiLU and self-attention input RMSNorm fixes.
- Artifacts:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-r9700-golden-before0-8-self-attn-input-rmsnorm-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-r9700-actual-prefix0-8-self-attn-input-rmsnorm-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-v620-golden-before0-8-self-attn-input-rmsnorm-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-v620-actual-prefix0-8-self-attn-input-rmsnorm-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-hip-backend-self-attn-input-rmsnorm-p4p46-summary.md`
- R9700/V620 `golden_before_each_layer` matched CPU within tiny backend noise:
  - CPU: `max_mse=0.000487887468`, `max_mean_abs_diff=0.016121546`, `max_abs_diff=0.486274719`, `min_cosine_similarity=0.999005169`
  - R9700/V620: `max_mse=0.000487887774`, `max_mean_abs_diff=0.016121550`, `max_abs_diff=0.486297607`, `min_cosine_similarity=0.999005168`
- R9700/V620 `actual_prefix` also matched CPU within tiny backend noise:
  - CPU: `max_mse=0.001895285663`, `max_mean_abs_diff=0.033220127`, `max_abs_diff=0.665708542`, `min_cosine_similarity=0.995342680`
  - R9700/V620: `max_mse=0.001895293292`, `max_mean_abs_diff=0.033220179`, `max_abs_diff=0.665769577`, `min_cosine_similarity=0.995342660`
- Re-ran `package-self-attn-mlp-block-model-loop-smoke` on R9700 and V620 with p4p46 `[3,7]`, sequence length `3`.
  - Both HIP backends passed.
  - Runtime/cache diffs stayed `0`.
  - Prepared q/k/RoPE/causal attention diffs remained in the existing small HIP range.

## Current Answer To User Question

- Yes, this was and still is worth debugging.
- Reason:
  - The earlier large layer 6 drift was caused by missing post-Conv1d SiLU before linear-attention recurrent q/k/v split.
  - The later layer 3 drift was caused by missing self-attention input RMSNorm before q/k/v projection.
  - These are implementation mismatches against the Qwen3.5 reference path, not random quantization noise.
- The urgency has changed:
  - before the fixes, the package path was materially wrong in two layer primitives;
  - after the fixes, CPU/R9700/V620 agree and the remaining `0..8` p4p46 drift is sub-1.0 max abs in the tested prefix.
- Next technical focus:
  - stop treating GPU backend as the likely cause for this fixture;
  - validate longer prefix or logits/generation behavior with p4p46;
  - if a new outlier appears, debug it with module-level traces first rather than trying another broad quantization-policy swap.

## Seq16 Prefix Validation

- Generated a new fixture:
  - `uLLM-project/benchmarks/golden/2026-07-05/qwen35-9b-prefix0-8-seq16`
  - token ids: `1..16`
  - layer range: `0..8`
  - dtype: `torch.bfloat16`
  - export device: CPU, because current Python Torch reports CUDA disabled.
- Ran p4p46-inproj package golden-prefix validation on CPU `0`, R9700 `2`, and V620 `1`.
- Artifacts:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-golden-before0-8-seq16-self-attn-input-rmsnorm-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix0-8-seq16-self-attn-input-rmsnorm-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-r9700-golden-before0-8-seq16-self-attn-input-rmsnorm-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-r9700-actual-prefix0-8-seq16-self-attn-input-rmsnorm-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-v620-golden-before0-8-seq16-self-attn-input-rmsnorm-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-v620-actual-prefix0-8-seq16-self-attn-input-rmsnorm-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-seq16-p4p46-backend-summary.md`
- `golden_before_each_layer` seq16:
  - CPU: `max_mse=0.000361034838`, `max_mean_abs_diff=0.014008828`, `max_abs_diff=0.486213684`, `min_cosine_similarity=0.998973254`
  - R9700/V620: `max_mse=0.000361035509`, `max_mean_abs_diff=0.014008830`, `max_abs_diff=0.486255646`, `min_cosine_similarity=0.998973254`
  - worst layer/token/hidden: layer `6`, token `0`, hidden `3994`
- `actual_prefix` seq16:
  - CPU: `max_mse=0.001575089386`, `max_mean_abs_diff=0.030268902`, `max_abs_diff=0.665708542`, `min_cosine_similarity=0.995290748`
  - R9700/V620: `max_mse=0.001575093148`, `max_mean_abs_diff=0.030268933`, `max_abs_diff=0.665769577`, `min_cosine_similarity=0.995290736`
  - worst layer/token/hidden: layer `7`, token `0`, hidden `3994`
- Compared with seq8:
  - seq8 actual p4p46 CPU: `max_abs_diff=0.665708542`, `max_mse=0.001895285663`
  - seq16 actual p4p46 CPU: `max_abs_diff=0.665708542`, `max_mse=0.001575089386`
- Interpretation:
  - The longer `seq16 / 0..8` fixture does not reintroduce the earlier large implementation-mismatch outliers.
  - CPU/R9700/V620 remain backend-stable.
  - The persistent worst coordinate is still hidden `3994`, but the residual error now looks like quantization/prefix accumulation rather than another obvious missing primitive in the tested path.
  - Next best validation target is longer layer coverage or logits/generation, not more backend comparison on `0..8`.

## Seq16 0..12 Prefix Validation

- Generated another fixture:
  - `uLLM-project/benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16`
  - token ids: `1..16`
  - layer range: `0..12`
  - dtype: `torch.bfloat16`
  - export device: CPU
- Ran p4p46-inproj package golden-prefix validation on CPU `0`, R9700 `2`, and V620 `1`.
- Artifacts:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-golden-before0-12-seq16-self-attn-input-rmsnorm-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix0-12-seq16-self-attn-input-rmsnorm-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-r9700-golden-before0-12-seq16-self-attn-input-rmsnorm-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-r9700-actual-prefix0-12-seq16-self-attn-input-rmsnorm-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-v620-golden-before0-12-seq16-self-attn-input-rmsnorm-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-v620-actual-prefix0-12-seq16-self-attn-input-rmsnorm-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-seq16-0-12-p4p46-backend-summary.md`
- `golden_before_each_layer` seq16 `0..12`:
  - CPU: `max_mse=0.000361034838`, `max_mean_abs_diff=0.014148540`, `max_abs_diff=0.875896454`, `min_cosine_similarity=0.998973254`
  - R9700/V620: `max_mse=0.000361035509`, `max_mean_abs_diff=0.014148535`, `max_abs_diff=0.875885010`, `min_cosine_similarity=0.998973254`
  - worst layer/token/hidden: layer `10`, token `0`, hidden `3456`
- `actual_prefix` seq16 `0..12`:
  - CPU: `max_mse=0.003190722428`, `max_mean_abs_diff=0.043881594`, `max_abs_diff=1.744266510`, `min_cosine_similarity=0.994555928`
  - R9700/V620: `max_mse=0.003190725513`, `max_mean_abs_diff=0.043881624`, `max_abs_diff=1.744228363`, `min_cosine_similarity=0.994555922`
  - worst layer/token/hidden: layer `10`, token `0`, hidden `3456`
- Interpretation:
  - Extending from `0..8` to `0..12` exposes a new layer 10 outlier at token `0`, hidden `3456`.
  - The outlier is already visible in `golden_before_each_layer`, so it is not only prefix accumulation.
  - `actual_prefix` roughly doubles the max absolute difference at the same coordinate.
  - CPU/R9700/V620 remain backend-stable, so this is a model/package numeric issue rather than a backend issue.
  - Next narrow debug target is layer 10 token `0` hidden `3456` with module-level tracing.

## Layer 10 Hidden 3456 Diagnosis

- Exported full-reference module trace:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer-module-trace-fullref-layer10-hidden3456-seq16-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer-module-trace-fullref-layer10-hidden3456-seq16-p4p46-inproj.md`
- Compared against package `golden_before_each_layer` row:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-module-trace-comparison-golden-before-layer10-hidden3456-seq16-p4p46-inproj.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-module-trace-comparison-golden-before-layer10-hidden3456-seq16-p4p46-inproj.md`
- Checked row reconstruction:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/golden-prefix-row-error-layer10-hidden3456-p4p46-inproj.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/golden-prefix-row-error-layer10-hidden3456-p4p46-inproj.md`
- Added diagnosis summary:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-layer10-hidden3456-diagnosis.md`
- Golden-before split at layer `10`, token `0`, hidden `3456`:
  - package output diff: `-0.875896454`
  - expected delta: `22.75`
  - package delta: `21.874103546`
  - attention error: `-0.183621407`
  - attention row-only error: `-0.169056547`
  - attention activation-path error: `-0.021181354`
  - MLP error: `-0.754775047`
  - MLP row-only error: `-0.613532790`
  - MLP activation-path error: `-0.172058034`
- Row reconstruction:
  - `linear_attn.out_proj[3456]`: row RMS `0.001093848`, row max abs `0.017172441`, relative MSE `0.003870684`
  - `mlp.down_proj[3456]`: row RMS `0.000707810`, row max abs `0.018229157`, relative MSE `0.007366217`
- Actual-prefix accumulation at the same coordinate:
  - output diff: `-1.744266510`
  - input diff: `-0.769666672`
  - delta diff: `-0.974599838`
- Interpretation:
  - This differs from the earlier layer 6 issue. The layer 6 issue was dominated by a missing primitive before projection; layer 10 hidden `3456` is mostly final projection row dot-product sensitivity.
  - The row max error is not huge by itself, but it aligns with high-impact activation directions.
  - In actual-prefix mode, prefix input drift and layer-local row-dot error combine.
  - Next useful work is quantization-policy or row-compensation investigation for sensitive output rows such as layer 10 `mlp.down_proj[3456]`, not another backend comparison.

## Seq16 0..12 Policy Comparison

- Ran CPU `0` seq16 `0..12` validation for:
  - p4p6
  - p4p46-inproj
  - p4p65-inproj
- Added summary:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-seq16-0-12-policy-comparison.md`
- Artifacts added for p4p6:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-golden-before0-12-seq16-self-attn-input-rmsnorm-p4p6.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix0-12-seq16-self-attn-input-rmsnorm-p4p6.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-module-trace-comparison-golden-before-layer10-hidden3456-seq16-p4p6.json`
- Artifacts added for p4p65-inproj:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-golden-before0-12-seq16-self-attn-input-rmsnorm-p4p65-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix0-12-seq16-self-attn-input-rmsnorm-p4p65-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-module-trace-comparison-golden-before-layer10-hidden3456-seq16-p4p65-inproj.json`
- CPU `actual_prefix 0..12` comparison:
  - p4p6: `max_mse=0.003687207709`, `max_mean_abs_diff=0.046535894`, `max_abs_diff=2.235244751`, `min_cosine_similarity=0.993695660`
  - p4p46-inproj: `max_mse=0.003190722428`, `max_mean_abs_diff=0.043881594`, `max_abs_diff=1.744266510`, `min_cosine_similarity=0.994555928`
  - p4p65-inproj: `max_mse=0.003322313414`, `max_mean_abs_diff=0.044468240`, `max_abs_diff=1.787147522`, `min_cosine_similarity=0.994327194`
- Layer 10 hidden `3456` module split:
  - p4p6 output diff: `-1.040985107`
  - p4p46-inproj output diff: `-0.875896454`
  - p4p65-inproj output diff: `-0.886669159`
  - attention row-only error is identical across policies: `-0.169056547`
  - MLP row-only error is identical across policies: `-0.613532790`
- Interpretation:
  - p4p46-inproj remains the best tested policy on this fixture.
  - The layer 10 outlier is not fixed by in-projection policy changes because the dominant row-only contribution is in `linear_attn.out_proj` and especially `mlp.down_proj`.
  - Further improvement likely needs output-row/down-row treatment or compensation, not another in-projection-only policy swap.

## Row-Dot Sensitivity Tool

- Added `tools/analyze-qwen-row-dot-sensitivity.py`.
  - Reads `row_dot` blocks from `export-qwen-layer-module-trace.py` JSONL.
  - Summarizes per-projection row-dot RMSE, max error, worst token, and a simple optimal row-scale estimate over the traced token set.
  - Outputs JSON and Markdown summaries.
  - Added `--dedupe` to collapse repeated trace rows with identical row-dot metrics.
- Verified:
  - `python3 -m py_compile tools/analyze-qwen-row-dot-sensitivity.py`: passed
- Applied it to layer `10`, hidden `3456`, p4p46-inproj:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-row-dot-sensitivity-layer10-hidden3456-seq16-p4p46-inproj.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-row-dot-sensitivity-layer10-hidden3456-seq16-p4p46-inproj.md`
- Row scale estimate:
  - `mlp_down_proj`: original RMSE `0.153679879`, max abs `0.613532790`, optimal scale `1.04165701172`, scaled RMSE `0.006549434`, scaled max abs `0.014996152`, improvement ratio `0.957382620`
  - `attention_out_proj`: original RMSE `0.042451593`, max abs `0.169056547`, optimal scale `1.02307179310`, scaled RMSE `0.002901533`, scaled max abs `0.006276145`, improvement ratio `0.931650791`
- Interpretation:
  - The layer 10 row-dot error is not just random point noise; over the 16 traced tokens, a single row scale explains most of the observed package-vs-source row-dot error.
  - This does not prove row scaling is globally safe, but it justifies a targeted row-compensation experiment on sensitive `out_proj` / `down_proj` rows.
- Applied the same analyzer to all existing full-reference row-dot traces from 2026-07-05:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-row-dot-sensitivity-existing-traces-summary.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-row-dot-sensitivity-existing-traces-summary.md`
- Dedupe summary:
  - raw rows: `30`
  - deduped rows: `16`
- Top deduped candidates:
  - layer `10`, hidden `3456`, `mlp_down_proj`: original max abs `0.613532790`, scaled max abs `0.014996152`, improvement `0.957382620`
  - layer `6`, hidden `3456`, `mlp_down_proj`: original max abs `0.442788575`, scaled max abs `0.026397223`, improvement `0.927011471`
  - layer `6`, hidden `3994`, `mlp_down_proj`: original max abs `0.372785634`, scaled max abs `0.132066763`, improvement `0.558370511`
  - layer `6`, hidden `3456`, `attention_out_proj`: original max abs `0.228395332`, scaled max abs `0.009216717`, improvement `0.935385444`
  - layer `10`, hidden `3456`, `attention_out_proj`: original max abs `0.169056547`, scaled max abs `0.006276145`, improvement `0.931650791`
- Updated interpretation:
  - Row-scale-like sensitivity is not unique to layer 10; it also appears in earlier layer 6 traces.
  - The strongest candidates are concentrated in `mlp_down_proj` and `attention_out_proj`.
  - A future experiment should test a package/runtime-compatible compensation mechanism on a small allowlist of sensitive rows before considering broader policy changes.

## Row-Dot Compensation Validation Plan

- Added `uLLM-project/docs/plans/row-dot-compensation-validation-plan-v0.1.md`.
- Purpose:
  - Define a smoke-only row scale override experiment for the remaining layer 10 hidden `3456` outlier.
  - Keep the experiment local to `package-golden-prefix-smoke`, without changing the package file format, quantizer, or production runtime API.
- Initial validation scope:
  - package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-reservoir65536-jobs4.ullm.d`
  - fixture: `benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16`
  - backend: CPU first, R9700 only if CPU improves
  - mode: compare `golden_before_each_layer` and `actual_prefix` with/without overrides
- Proposed first overrides:
  - layer `10`, `linear_attn.out_proj.weight`, row `3456`, scale `1.02307179310`
  - layer `10`, `mlp.down_proj.weight`, row `3456`, scale `1.04165701172`
- Rationale:
  - The row-dot sensitivity analyzer estimates >93% row-dot error reduction on the traced layer 10 tokens for both projections.
  - This is still only a validation experiment; the scale values may be prompt-specific and must not be treated as a production solution yet.
- Next implementation step:
  - Add optional row scale override JSON support to `package-golden-prefix-smoke`.
  - Apply overrides only to selected materialized f32 runtime matrices in the linear-attention/MLP sequence path.

## Row-Scale Override Smoke Implementation

- Implemented smoke-only row scale override support in `uLLM-project/crates/ullm-engine/src/main.rs`.
  - Added optional `package-golden-prefix-smoke` positional argument: `[ROW_SCALE_OVERRIDES_JSON]`.
  - Added schema `package-row-scale-overrides-v0.1`.
  - Supported tensors are intentionally narrow:
    - `linear_attn.out_proj.weight`
    - `mlp.down_proj.weight`
  - Runtime path:
    - copy the selected materialized f32 matrix to host
    - scale the selected row in-place in the byte buffer
    - copy the matrix back to the runtime buffer
    - record applied overrides in the JSONL report
- Validation commands:
  - `cargo fmt --all --check`: passed
  - `cargo check -p ullm-engine`: passed
  - `cargo build -p ullm-engine`: passed
- Added override config:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-row-scale-overrides-layer10-hidden3456-p4p46-inproj.json`
- Added summary:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-row-scale-override-p4p46-inproj-summary.md`
- CPU current-binary baseline vs override:
  - `golden_before_each_layer`
    - no override: `max_abs_diff=0.875896454`, worst max-abs layer `10`
    - override: `max_abs_diff=0.486213684`, worst max-abs layer `6`
  - `actual_prefix`
    - no override: `max_abs_diff=1.744266510`, `max_mse=0.004029775765`
    - override: `max_abs_diff=0.967845917`, `max_mse=0.003994860341`
- Layer `10` direct effect:
  - `golden_before_each_layer`: layer `10` max abs fell from `0.875896454` to `0.304975510`.
  - `actual_prefix`: layer `10` max abs fell from `1.744266510` to `0.967845917`.
- R9700 override validation:
  - `golden_before_each_layer`: `max_abs_diff=0.486255646`
  - `actual_prefix`: `max_abs_diff=0.967796326`
  - CPU/R9700 results are backend-stable.
- Interpretation:
  - Continuing the debug was justified.
  - Row-scale compensation reduces the layer `10` hot-coordinate outlier as predicted by the row-dot sensitivity analysis.
  - Aggregate MSE and mean absolute diff are still dominated by layer `11`, so layer `11` needs a separate investigation.
  - The current implementation should remain a smoke-only validation path; production work should move the idea into quantizer-side row compensation if further validation holds.

## Self-Attention Row-Scale Probe

- Extended row scale override support to the self-attention package path.
  - `self_attn.o_proj.weight` is now accepted by the schema.
  - `mlp.down_proj.weight` overrides can now apply in self-attention layers as well as linear-attention layers.
- Motivation:
  - Layer `11` is a self-attention layer.
  - In `golden_before_each_layer`, layer `11` local max diff was hidden `3377`, token `13`, with `max_abs_diff=0.179061234`.
  - A simple least-squares fit from the package module-contribution trace suggested scaling layer `11` `mlp.down_proj[3377]` by `1.218300518695`.
- Added override config:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-row-scale-overrides-layer11-hidden3377-self-attn-mlp-p4p46-inproj.json`
- Added validation report:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-golden-before-layer11-row-scale-override-p4p46-inproj.jsonl`
- Result:
  - layer `11` `golden_before_each_layer` max abs improved only from `0.179061234` to `0.167163849`.
  - MSE and mean absolute diff barely changed:
    - MSE: `0.000715637264` to `0.000714828336`
    - mean abs: `0.020423743` to `0.020418806`
  - After scaling row `3377`, the max coordinate moved to hidden `3994`.
- Interpretation:
  - Layer `11` is not the same clean single-row case as layer `10`.
  - It likely needs broader multi-row analysis or a self-attention-specific row-dot trace, not another one-off row scale.
- Follow-up two-row test:
  - Added layer `11` `mlp.down_proj[3994]` with least-squares scale `1.020286172534` after applying row `3377`.
  - Added override config:
    - `uLLM-project/benchmarks/results/2026-07-05/engine/package-row-scale-overrides-layer11-hidden3377-3994-self-attn-mlp-p4p46-inproj.json`
  - Added report:
    - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-golden-before-layer11-two-row-scale-override-p4p46-inproj.jsonl`
  - Result:
    - MSE: `0.000714424519`
    - mean abs: `0.020416921`
    - max abs: `0.179649353`
  - Interpretation:
    - The two-row least-squares version improves MSE only marginally but worsens max abs versus the single-row probe.
    - A max-coordinate fit for row `3994` was also tested locally and was much worse (`max_abs_diff=2.040929794`), so that artifact was discarded.
    - Layer `11` should be treated as a separate multi-row or mixed-path drift problem.

## Module-Contribution Scale-Fit Analyzer

- Extended `uLLM-project/tools/analyze-golden-prefix-module-contribution.py`.
  - For each `module_contribution.per_token_hot_hidden_trace`, it now estimates:
    - optimal MLP component scale while holding attention output fixed
    - optimal attention component scale while holding MLP output fixed
    - original/scaled RMSE
    - original/scaled max absolute delta error
    - RMSE improvement ratio
- Regenerated:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-row-scale-module-contribution-summary.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-row-scale-module-contribution-summary.md`
- Verification:
  - `python3 -m py_compile tools/analyze-golden-prefix-module-contribution.py`: passed
- Useful comparisons from the regenerated summary:
  - Layer `10`, golden-before, hidden `3456`:
    - MLP scale fit improvement: `0.862742`
    - attention scale fit improvement: `0.919726`
  - Layer `11`, golden-before, hidden `3377`:
    - MLP scale fit improvement: `0.349441`
    - attention scale fit improvement: `0.382474`
- Interpretation:
  - The analyzer now makes the difference between layer `10` and layer `11` explicit.
  - Layer `10` is highly scale-like.
  - Layer `11` is only weakly scale-like and should not be debugged by blindly adding hand-picked row scales.

## Quantizer Row Compensation Plan

- Added `uLLM-project/docs/plans/quantizer-row-compensation-plan-v0.1.md`.
- Purpose:
  - Decide how to move the validated layer `10` smoke-only row scale override toward a package/quantizer-compatible prototype.
- Compared implementation options:
  - manifest row-scale metadata
  - pre-scaled quantization
  - hybrid row override tensor
- Current decision:
  - Start with manifest row-scale metadata.
  - Keep AQ payload unchanged.
  - Let loader/materialize/fused-dequant eventually apply explicit row multipliers.
- Reasoning:
  - This preserves a clear distinction between quantized weights and compensation metadata.
  - It is easiest to validate against the current smoke-only override.
  - It avoids hiding compensation inside quantizer metrics.
- Initial target:
  - layer `10`, `linear_attn.out_proj.weight`, row `3456`, scale `1.02307179310`
  - layer `10`, `mlp.down_proj.weight`, row `3456`, scale `1.04165701172`
- Explicit non-target:
  - layer `11`, because the row-scale behavior is weak and mixed-path.

## Manifest Row-Scale Compensation Prototype

- Implemented optional package manifest metadata:
  - `row_scale_overrides.schema_version = "row-scale-overrides-v0.1"`
  - entries use full `tensor_name`, `row_index`, `scale`, and optional `source`
- Engine changes:
  - `uLLM-project/crates/ullm-engine/src/package.rs`
    - Parses optional `row_scale_overrides`.
    - Attaches matching entries to `TensorPayloadBundle`.
    - Validates schema version, non-empty tensor name, positive finite scale, and duplicate `(tensor_name,row_index)`.
  - `uLLM-project/crates/ullm-engine/src/loader.rs`
    - Applies matching row scales immediately after AQ4 materialization in `materialize_selected_aq4_matrix`.
    - Existing packages without metadata remain a no-op.
    - Added a loader test proving `tensor.weight` row `1` changes from `[3,4]` to `[30,40]` after metadata scale `10.0`.
- Quantizer changes:
  - `uLLM-project/crates/ullm-quant/src/main.rs`
    - Added `--row-scale-overrides-json PATH`.
    - Adds optional metadata to direct package `manifest.json`.
    - Validates that target tensors are selected into the direct package, are 2D, and have in-range row indices.
    - Existing manifest JSON without the field still deserializes with `None`.
- Docs/artifacts:
  - Updated `uLLM-project/docs/plans/quantizer-row-compensation-plan-v0.1.md`.
  - Added result summary:
    - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-manifest-row-scale-p4p46-inproj-summary.md`
  - Added terms to `uLLM-project/docs/words.txt`:
    - `row-scale overrides`
    - `manifest row-scale compensation`
- Validation:
  - `cargo fmt --all --check`: passed
  - `cargo check -p ullm-engine`: passed
  - `cargo check -p ullm-quant`: passed
  - `cargo test -p ullm-engine package -- --test-threads=1`: passed
  - `cargo test -p ullm-engine loader -- --test-threads=1`: passed
  - `cargo test -p ullm-quant prototype_manifest_defaults_missing_passthrough_tensors -- --test-threads=1`: passed
  - `cargo test -p ullm-quant direct_package_writes_quantized_and_passthrough_payloads -- --test-threads=1`: passed
  - `cargo build -p ullm-engine`: passed
- Runtime validation package:
  - Created hardlink copy:
    - `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer10.ullm.d`
  - Only `manifest.json` differs from the p4p46-inproj package.
  - Metadata entries:
    - layer `10` `linear_attn.out_proj.weight[3456]`, scale `1.02307179310`
    - layer `10` `mlp.down_proj.weight[3456]`, scale `1.04165701172`
- CPU current-binary no-metadata vs manifest metadata:
  - `golden_before_each_layer`
    - max MSE: `0.000740506879` to `0.000740506879`
    - max abs: `0.875896454` to `0.508314133`
    - layer `10` max abs: `0.875896454` to `0.304975510`
  - `actual_prefix`
    - max MSE: `0.004141662294` to `0.004106469453`
    - max abs: `1.744266510` to `0.967845917`
    - layer `10` max abs: `1.744266510` to `0.967845917`
- R9700 manifest metadata:
  - `golden_before_each_layer`
    - max MSE `0.000740507114`
    - max mean abs diff `0.020715803`
    - max abs diff `0.508314133`
    - min cosine `0.998585695`
  - `actual_prefix`
    - max MSE `0.004106476000`
    - max mean abs diff `0.050080222`
    - max abs diff `0.967796326`
    - min cosine `0.992982658`
- Interpretation:
  - Manifest metadata reproduces the layer `10` max-abs improvement without passing the smoke-only row-scale JSON argument.
  - CPU and R9700 agree within expected tolerance.
  - Aggregate MSE remains controlled by later-layer drift, so layer `11` remains a separate debugging problem.

## Layer 11 Self-Attention Row-Dot Trace

- Extended `uLLM-project/tools/export-qwen-layer-module-trace.py` beyond `linear_attention`.
  - Added `full_attention` layer replay using the model rotary embedding.
  - Captures self-attention q/k/v projections, q/k norm outputs, q projection query/gate split, gated `o_proj` input, `o_proj` output, post RMSNorm, MLP activation, and MLP output.
  - Emits `row_dot.self_attention_o_proj`, `row_dot.mlp_down_proj`, and q/k/v `projection_row_dot` diagnostics.
- Generated layer `11` traces:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer-module-trace-layer11-hidden3377-full-attn-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer-module-trace-layer11-hidden3994-full-attn-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-row-dot-sensitivity-layer11-hidden3377-3994-full-attn-p4p46-inproj.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer11-self-attention-row-dot-p4p46-inproj-summary.md`
- Replay validation:
  - `fixture_match.max_abs_diff = 0.03125`
  - `fixture_match.mean_abs_diff = 0.000389998604`
  - `fixture_match.mse = 0.000000643706810`
- Final row-dot sensitivity:
  - hidden `3377`, `self_attention_o_proj`: original max abs `0.0113390533`, scaled max abs `0.00737102677`
  - hidden `3377`, `mlp_down_proj`: original max abs `0.0151258546`, scaled max abs `0.0151908693`
  - hidden `3994`, `self_attention_o_proj`: original max abs `0.106661194`, scaled max abs `0.107175835`
  - hidden `3994`, `mlp_down_proj`: original max abs `0.0381102959`, scaled max abs `0.0384955943`
- q/k/v projection hotspots:
  - hidden `3377`, q projection worst row-dot error `0.453516544`
  - hidden `3994`, k projection worst row-dot error `0.507676098`
  - hidden `3994`, v projection worst row-dot error `-0.987190539`
- Verification:
  - `python3 -m py_compile tools/export-qwen-layer-module-trace.py tools/analyze-qwen-row-dot-sensitivity.py`: passed
- Interpretation:
  - Layer `11` is still worth debugging, but the evidence now points away from final-row scale overrides.
  - The next useful target is q/k/v projection error propagation through q/k norm, RoPE, attention value mix, output gate, and `o_proj` input.

## Layer 11 Self-Attention Propagation Diagnostic

- Added `uLLM-project/tools/analyze-qwen-self-attention-propagation.py`.
  - Reconstructs package q/k/v projections from AQ4 payloads.
  - Replays the same SDPA causal attention path used by the Qwen3.5 layer.
  - Verifies the source replay against the actual layer `o_proj` pre-hook.
  - Measures how q/k/v projection error survives into `o_proj` input and selected hidden-row output contributions.
- Generated:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-self-attention-propagation-layer11-hidden3377-3994-3456-p4p46-inproj.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-self-attention-propagation-layer11-hidden3377-3994-3456-p4p46-inproj.md`
- Stage-level result:
  - source replay vs layer hook max abs: `0`
  - package q projection vs source max abs: `0.823976994`
  - package k projection vs source max abs: `0.479312897`
  - package v projection vs source max abs: `0.952980042`
  - package `o_proj` input vs source max abs: `0.187569141`
- Selected hidden-row propagation:
  - hidden `3377`: worst source-row input contribution `-0.0221332256`; worst total package-row contribution `0.0242107697`
  - hidden `3994`: worst source-row input contribution `-0.0975656509`; worst total package-row contribution `0.11529398`
  - hidden `3456`: worst source-row input contribution `0.00994926319`; worst total package-row contribution `-0.0106792711`
- Verification:
  - `python3 -m py_compile tools/analyze-qwen-self-attention-propagation.py`: passed
- Interpretation:
  - q/k/v projection quantization is real, but causal attention and output gating reduce the error before `o_proj`.
  - hidden `3994` still carries a measurable attention-input contribution, while hidden `3377` and `3456` are small in the layer-local golden-before replay.
  - Next debugging should compare this layer-local propagation against actual-prefix input drift, because the larger runtime layer `11` max error appears to depend on incoming residual error as well as local q/k/v quantization.

## Layer 11 Actual-Prefix Drift Check

- Compared existing actual-prefix artifacts for layer `11`:
  - no metadata CPU:
    - input max abs diff `1.744266510`, token `0`, hidden `3456`
    - output max abs diff `1.686901093`, token `0`, hidden `3456`
  - manifest row-scale CPU:
    - input max abs diff `0.967845917`, token `0`, hidden `3456`
    - output max abs diff `0.911422729`, token `0`, hidden `3456`
  - manifest row-scale R9700:
    - input max abs diff `0.967796326`, token `0`, hidden `3456`
    - output max abs diff `0.911373138`, token `0`, hidden `3456`
- Added this comparison to:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer11-self-attention-row-dot-p4p46-inproj-summary.md`
- Interpretation:
  - The biggest actual-prefix layer `11` error is already present at the layer input.
  - The layer-local propagation diagnostic says hidden `3456` contributes only about `0.011` through local self-attention/o-row paths.
  - Therefore the layer `11` max coordinate should be treated primarily as inherited residual drift from earlier layers, not as a new layer `11` row-scale target.
  - The remaining useful layer-local layer `11` target is hidden `3994` attention-input propagation, but it is smaller than the inherited hidden `3456` drift.

## Layer 6 + Layer 10 Manifest Row-Scale Probe

- Created hardlink package:
  - `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer10.ullm.d`
- Added four manifest row-scale entries:
  - layer `6` `linear_attn.out_proj.weight[3456]`, scale `1.032273364777375`
  - layer `6` `mlp.down_proj.weight[3456]`, scale `1.036585679248007`
  - layer `10` `linear_attn.out_proj.weight[3456]`, scale `1.0230717930961908`
  - layer `10` `mlp.down_proj.weight[3456]`, scale `1.0416570117172528`
- Added artifacts:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-row-scale-overrides-layer6-layer10-hidden3456-p4p46-inproj.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-manifest-row-scale-layer6-layer10-p4p46-inproj-summary.md`
  - CPU/R9700 actual-prefix and golden-before JSONL files for the layer6+10 package.
- CPU results:
  - `golden_before_each_layer`: max abs `0.508314133`, max MSE `0.000740506879`
  - `actual_prefix`: max abs `0.891334534`, max MSE `0.004097481631`
- R9700 results:
  - `golden_before_each_layer`: max abs `0.508314133`, max MSE `0.000740507114`
  - `actual_prefix`: max abs `0.891326904`, max MSE `0.004097484565`
- Comparison against layer `10`-only manifest metadata:
  - `actual_prefix` CPU max abs improved from `0.967845917` to `0.891334534`.
  - `actual_prefix` R9700 max abs improved from `0.967796326` to `0.891326904`.
  - `golden_before_each_layer` max abs stayed at `0.508314133`.
- Layer movement in CPU `actual_prefix`:
  - layer `6` max moved from token `0`, hidden `3456`, `0.588710785` to token `0`, hidden `3994`, `0.480636597`
  - layer `10` max moved from token `0`, hidden `3456`, `0.967845917` to token `7`, hidden `3994`, `0.461685181`
  - layer `11` max moved from token `0`, hidden `3456`, `0.911422729` to token `11`, hidden `3994`, `0.891334534`
- Interpretation:
  - Layer `6` metadata removes the hidden `3456` actual-prefix drift chain.
  - The remaining dominant coordinate is hidden `3994`, consistent with the layer `11` self-attention propagation diagnostic.
  - Row-scale metadata is helpful for scale-like rows, but not sufficient as a complete policy.

## Layers 7-9 Hidden 3994 Row-Dot Check

- Generated module trace and row-dot sensitivity for hidden `3994`:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer-module-trace-layers7-9-hidden3994-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-row-dot-sensitivity-layers7-9-hidden3994-p4p46-inproj.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer7-9-hidden3994-row-dot-p4p46-inproj-summary.md`
- Golden-before final-row sensitivity:
  - layer `7` `self_attention_o_proj`: original max abs `0.105299867`, scaled max abs `0.0496769869`
  - layer `9` `mlp_down_proj`: original max abs `0.0793600213`, scaled max abs `0.0682068663`
  - layer `9` `attention_out_proj`: original max abs `0.0781332336`, scaled max abs `0.0823951967`
- Token `11`, hidden `3994` row-dot errors:
  - layer `7` final rows: `-0.00885246908` and `0.0120474609`
  - layer `8` final rows: `-0.0195115685` and `-0.0220584367`
  - layer `9` final rows: `-0.0481112904` and `0.0216266042`
- Actual-prefix hidden `3994`, token `11`, after layer6+10 metadata:
  - layer `7`: input diff `-0.139934540`, output diff `-0.463647842`, delta diff `-0.323713303`
  - layer `8`: input diff `-0.463647842`, output diff `0.211425781`, delta diff `0.675073624`
  - layer `9`: input diff `0.211425781`, output diff `-0.300535202`, delta diff `-0.511960983`
  - layer `11`: input diff `-0.444337845`, output diff `-0.891334534`, delta diff `-0.446996689`
- Interpretation:
  - The hidden `3994` chain is not explained by one final projection row.
  - Layer `7` has a modest scale-like final-row signal, but token `11` itself is small.
  - Layer `8/9/11` behavior looks like input-distribution-sensitive nonlinear propagation rather than a simple row-scale target.

## Manifest Row-Scale Smoke Reporting

- Updated `package-golden-prefix-smoke` output in `uLLM-project/crates/ullm-engine/src/main.rs`.
  - Summary line now includes `manifest_row_scale_overrides=<count>`.
  - JSONL rows now include `manifest_row_scale_override_count`.
  - Existing `row_scale_overrides=<path|none>` remains the smoke-only CLI override source.
- Reason:
  - Manifest metadata packages previously printed `row_scale_overrides=none`, which was technically true for the CLI override but misleading for packages with manifest metadata.
- Validation:
  - `cargo fmt --all --check`: passed
  - `cargo check -p ullm-engine`: passed
  - `cargo build -p ullm-engine`: passed
  - `cargo test -p ullm-engine package -- --test-threads=1`: passed
  - Short CPU smoke on layer `6..7` with the layer6+10 metadata package printed `manifest_row_scale_overrides=4`.
  - The smoke JSONL row contained `manifest_row_scale_override_count = 4`.

## Layer 7 Feature 503 Self-Attention Replay

- Extended `uLLM-project/tools/analyze-qwen-self-attention-propagation.py`.
  - Added `--input-override-dir` so the Python replay can use actual-prefix layer input dumps.
  - Added per-token/per-feature stage traces.
  - Emitted source/package values for q projection, gate projection, q/k normed, RoPE, raw attention, sigmoid gate, and gated `o_proj` input.
- Generated:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-self-attention-propagation-layer7-actual-input-token8-feature503-hidden3994-p4p46-inproj.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-self-attention-propagation-layer7-actual-input-token8-feature503-hidden3994-p4p46-inproj.md`
- Layer `7`, token `8`, feature `503`:
  - source raw attention: `1.0234375`
  - package replay raw attention: `1.011238456`
  - source gated `o_proj` input: `0.628906250`
  - package replay gated `o_proj` input: `0.629811943`
  - actual Rust package JSONL gated `attention_projection_input`: `1.128042817`
- Interpretation:
  - PyTorch replay with dequantized package q/k/v does not reproduce the Rust package runtime value.
  - The mismatch is not explained by simple q/k/v row reconstruction on the sampled feature alone.

## Layer 7 Rust Causal Attention Diagnostic

- Extended `package-golden-prefix-smoke` self-attention JSONL detail with `causal_attention_runtime_diagnostic`.
  - Compares prepared causal attention, layer paged attention, and pure host causal attention on the same Rust prepared q/k/v.
  - Replays q gate application and records token `8`, feature `503`.
- Generated:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix-layer0-8-causal-attn-diag-layer7-p4p46-inproj.jsonl`
- Re-running layers `0..8` reproduced the prior layer `7` max abs diff: `0.756856918`.
- Layer `7`, token `8`, feature `503`:
  - prepared attention output: `1.810265899`
  - layer attention output: `1.810265899`
  - pure host attention output: `1.810265899`
  - q gate: `0.502882540`
  - sigmoid(q gate): `0.623136520`
  - prepared/layer/host projection input: `1.128042817`
- Full-vector max diffs:
  - prepared attention vs host causal: `0`
  - layer attention vs host causal: `0`
  - layer attention vs prepared attention: `0`
  - layer projection input vs host projection input: `0`
  - layer projection input vs prepared projection input: `0`
  - layer projection input vs replayed gate application: `0`
- Interpretation:
  - Rust causal attention, paged attention, and q gate application are internally consistent.
  - The next likely fault line is the Python replay tensor interpretation versus Rust prepared tensors, especially q/k/v layout, RoPE placement, or head/feature mapping around the head containing feature `503`.

## Rotary Dimension Correction

- Checked Qwen3.5-9B config:
  - `head_dim = 256`
  - `partial_rotary_factor = 0.25`
  - effective RoPE rotated width = `64`
  - `mrope_section = [11, 11, 10]`, summed `32`, then duplicated by HF rotary embedding to `64`
- Earlier smoke runs explicitly passed `rotary_dim = 32`.
  - This rotated only half of the expected Qwen3.5 text RoPE width.
  - The Rust package runtime and Python replay mismatch at layer `7`, token `8`, feature `503` was therefore a smoke configuration problem.
- Re-ran Rust and Python with `rotary_dim = 64` and fresh actual-prefix input dumps:
  - Rust package runtime:
    - raw attention `0.901784599`
    - sigmoid gate `0.624433100`
    - gated input `0.563104153`
  - Python package replay:
    - raw attention `0.901458323`
    - sigmoid gate `0.624092400`
    - gated input `0.562593281`
  - Source-token score/weight/value breakdowns aligned closely.
- Full CPU `0..12` actual-prefix results:
  - `rotary_dim=32`, layer6+10 row-scale: max abs `0.889577866`, layer `11`, token `6`, hidden `3994`
  - `rotary_dim=64`, no row-scale: max abs `1.744266510`, layer `10`, token `0`, hidden `3456`
  - `rotary_dim=64`, layer6+10 row-scale: max abs `0.645338058`, layer `11`, token `7`, hidden `3994`
- Additional rotary64 validation:
  - R9700 actual-prefix, layer6+10 row-scale: max abs `0.645345688`, layer `11`, token `7`, hidden `3994`
  - CPU golden-before, no row-scale: max abs `0.875896454`, layer `10`, token `0`, hidden `3456`
  - CPU golden-before, layer6+10 row-scale: max abs `0.472949982`, layer `6`, token `0`, hidden `3994`
- Interpretation:
  - The hidden `3456` row-scale chain is still real under the correct RoPE width.
  - The hidden `3994` self-attention discrepancy was amplified by the old `rotary_dim=32` smoke setting.
  - CPU and R9700 remain aligned under the corrected RoPE width.
  - Remaining useful work should localize layer `11`, token `7`, hidden `3994` under `rotary_dim=64` and avoid hardcoded `rotary_dim=32` in future Qwen3.5 text smoke commands.

## Rotary64 Layer 11 Hidden 3994 Local Comparison

- Re-ran actual-input full-reference module trace for layer `11`, hidden `3994`, using rotary64 input dumps.
- Generated:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer-module-trace-actual-input-rotary64-layer11-hidden3994-layer6-layer10-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer-module-trace-actual-input-rotary64-layer11-hidden3994-layer6-layer10-p4p46-inproj.md`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-module-trace-comparison-actual-input-rotary64-layer11-hidden3994-layer6-layer10-p4p46-inproj.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-module-trace-comparison-actual-input-rotary64-layer11-hidden3994-layer6-layer10-p4p46-inproj.md`
- Layer `11`, token `7`, hidden `3994`:
  - package output diff: `-0.645338058`
  - package input diff: `-0.376991272`
  - fixture expected delta: `2.25`
  - full-reference delta on actual input: `2.001991272`
  - package delta on actual input: `1.981653214`
  - package local delta error: `-0.020338058`
  - attention row-only error: `0.068367780`
  - attention activation-path error: `-0.052062498`
  - MLP row-only error: `-0.028672408`
  - MLP activation-path error: `-0.010083559`
- Interpretation:
  - The remaining layer `11` max is mostly inherited/input-distribution drift.
  - Layer `11` local quantization adds only about `-0.0203` beyond full-reference behavior on the same actual input.
  - Next useful target is where hidden `3994` input drift is introduced before layer `11`, especially layers `6..8`.

## Rotary64 Token 7 Hidden 3994 Chain

- Generated fixed-coordinate chain:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-coordinate-chain-rotary64-layer6-layer10-actual-h3994-t7-p4p46-inproj.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-coordinate-chain-rotary64-layer6-layer10-actual-h3994-t7-p4p46-inproj.md`
- Generated layers `7..9` actual-input traces and comparisons:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer-module-trace-actual-input-rotary64-layers7-9-hidden3994-layer6-layer10-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-module-trace-comparison-actual-input-rotary64-layers7-9-token7-hidden3994-layer6-layer10-p4p46-inproj.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-module-trace-comparison-actual-input-rotary64-layers7-9-token7-hidden3994-layer6-layer10-p4p46-inproj.md`
- Fixed coordinate chain for token `7`, hidden `3994`:
  - layer `7`: input diff `0.286402`, delta diff `-0.442692`, output diff `-0.156290`
  - layer `8`: input diff `-0.156290`, delta diff `0.452469`, output diff `0.296179`
  - layer `9`: input diff `0.296179`, delta diff `-0.576244`, output diff `-0.280066`
  - layer `10`: input diff `-0.280066`, delta diff `-0.096926`, output diff `-0.376991`
  - layer `11`: input diff `-0.376991`, delta diff `-0.268347`, output diff `-0.645338`
- Full-reference/package local delta on same actual inputs:
  - layer `7`: full delta `1.713598`, package delta `1.682308`, local error `-0.031290`
  - layer `8`: full delta `0.906290`, package delta `1.077469`, local error `0.171179`
  - layer `9`: full delta `1.828821`, package delta `1.798756`, local error `-0.030066`
  - layer `11`: full delta `2.001991`, package delta `1.981653`, local error `-0.020338`
- Interpretation:
  - For the layer `11` max coordinate, layer `8` is the largest local package-error candidate under `rotary_dim=64`.
  - Hot input vector detail for arbitrary token `7` is currently limited because `package-golden-prefix-smoke` records detailed hot vectors for the layer max token, not an arbitrary requested token.

## Sampled Token 7 Hot Vectors

- Extended `package-golden-prefix-smoke` with optional `SAMPLED_TOKEN_INDICES`.
  - Passing `7` emits `module_contribution.sampled_hot_input_vectors` for token `7`.
  - `INPUT_DUMP_DIR=none` is now treated as no input dump so the sampled token argument can be supplied safely.
- Re-ran package prefix `0..10` with `rotary_dim=64`, layer6+10 row-scale, sampled token `7`:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix0-10-rotary64-manifest-row-scale-layer6-layer10-sample-t7-p4p46-inproj.jsonl`
- Rebuilt fixed-token comparison:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-module-trace-comparison-actual-input-rotary64-layers7-9-token7-hidden3994-sampled-layer6-layer10-p4p46-inproj.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-module-trace-comparison-actual-input-rotary64-layers7-9-token7-hidden3994-sampled-layer6-layer10-p4p46-inproj.md`
- Layer `8`, token `7`, hidden `3994`:
  - local delta error: `0.171179`
  - attention row-only error: `-0.053269`
  - attention activation-path error: `0.125137`
  - MLP row-only error: `-0.017845`
  - MLP activation-path error: `0.070061`
  - top attention projection input value diff: feature `845`, `0.432841`
  - top MLP activation value diff: feature `6340`, `0.269271`
- Interpretation:
  - Layer `8` is not a simple final-row scaling issue for token `7`; row-only errors are negative and smaller.
  - The remaining local error is mainly activation-path drift in linear attention and MLP.

## Layer 8 MLP Gate/Up Diagnostic

- Extended linear-attention hot-vector diagnostics with:
  - `mlp_gate_projection`
  - `mlp_gate_silu`
  - `mlp_up_projection`
- Updated full-reference traces to emit the same MLP stages for both linear-attention and self-attention layers.
- Adjusted sampled feature selection so `mlp_*` stages use the top features from `mlp_activation`, not the attention projection input.
- Generated artifacts:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix0-10-rotary64-manifest-row-scale-layer6-layer10-sample-t7-mlp-gate-silu-up-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer-module-trace-actual-input-rotary64-layers7-9-hidden3994-mlp-gate-silu-up-layer6-layer10-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-module-trace-comparison-actual-input-rotary64-layers7-9-token7-hidden3994-mlp-gate-silu-up-layer6-layer10-p4p46-inproj.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer8-token7-hidden3994-mlp-gate-silu-up-summary.md`
- Layer `8`, token `7`, hidden `3994` remains the largest local package-error point:
  - local delta error: `0.171178818`
  - attention activation-path error: `0.125136728`
  - MLP activation-path error: `0.070061073`
- Attention feature `845`:
  - input RMSNorm diff: `-0.000215501`
  - recurrent diff: `0.000538458`
  - pre-gate normed diff: `0.144075036`
  - gate SiLU diff: `0.067126036`
  - projection input diff: `0.432840586`
- MLP feature `6340`:
  - gate projection diff: `-0.050819874`
  - gate SiLU diff: `-0.053045750`
  - up projection diff: `0.057474554`
  - activation diff: `0.269271016`
  - product split:
    - gate-SiLU term: `0.025486825`
    - up term: `0.244581638`
    - interaction: `-0.003048781`
- Interpretation:
  - The layer `8` attention-side error is head-wise RMSNorm/gate amplification of a small recurrent difference.
  - The MLP-side feature `6340` error is mainly `up_proj` error multiplied by a large positive gate-SiLU value.
  - This is still worth debugging, but the target is now internal projection-row sensitivity under high-gain activation paths, not backend behavior or final-row scale only.

## Layer 8 MLP Projection Row-Dot Diagnostic

- Extended `export-qwen-layer-module-trace.py` to schema `qwen-layer-module-trace-v0.6`.
  - Added `--token-index` so full-reference row-dot traces can target fixed token `7` instead of the per-layer max-delta token.
  - Added `projection_row_dot.mlp_gate_projection` and `projection_row_dot.mlp_up_projection` for both linear-attention and self-attention layers.
  - MLP projection row-dot feature selection now follows top `mlp_activation` features.
- Generated token `7` fixed artifacts:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer-module-trace-actual-input-rotary64-layers7-9-token7-hidden3994-mlp-proj-rowdot-layer6-layer10-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-module-trace-comparison-actual-input-rotary64-layers7-9-token7-hidden3994-mlp-proj-rowdot-layer6-layer10-p4p46-inproj.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer8-token7-hidden3994-mlp-proj-rowdot-summary.md`
- Layer `8`, token `7`, hidden `3994`:
  - local delta error vs fullref actual-input trace: `0.171178818`
  - attention row-only / activation-path: `-0.053268506` / `0.125136728`
  - MLP row-only / activation-path: `-0.017845260` / `0.070061073`
- MLP feature `6340` runtime:
  - gate projection diff: `-0.050819874`
  - gate SiLU diff: `-0.053045750`
  - up projection diff: `0.057474554`
  - activation diff: `0.269271016`
- Projection row-dot on full-reference `post_normed`:
  - `mlp.up_proj[6340]`: source row dot `-0.479665107`, package row dot `-0.442952289`, package-source error `0.036712819`
  - `mlp.gate_proj[6340]`: source row dot `4.318901433`, package row dot `4.246871380`, package-source error `-0.072030053`
- Interpretation:
  - `up_proj[6340]` package row-dot error explains roughly 65% of the runtime `up_proj[6340]` difference (`0.0375 / 0.0575`); the rest comes from `post_normed` input-path drift.
  - The MLP feature `6340` activation error is therefore not only inherited input drift. It includes a concrete internal projection-row quantization component.
  - Next focused experiment should target MLP internal projection row sensitivity, especially `mlp.up_proj[6340]`, rather than adding more final-output row-scale overrides.

## Layer 8 MLP Projection Row-Scale Experiment

- Extended `export-qwen-layer-module-trace.py` to schema `qwen-layer-module-trace-v0.7`.
  - `projection_row_dot` now emits `per_token_by_feature` for selected projection rows.
  - Each selected projection row includes least-squares `scale_fit` across all `16` tokens.
- For layer `8`, token `7`, hidden `3994`, MLP feature `6340`:
  - single-token row-dot scales:
    - `mlp.up_proj[6340]`: `1.082882107`
    - `mlp.gate_proj[6340]`: `1.016960733`
  - all-token least-squares scales:
    - `mlp.up_proj[6340]`: `1.035102073`
    - `mlp.gate_proj[6340]`: `1.011957038`
- All-token fit quality:
  - `mlp.up_proj[6340]`: RMSE `0.032796258 -> 0.025626328`, improvement `0.218620371`
  - `mlp.gate_proj[6340]`: RMSE `0.045137044 -> 0.012981628`, improvement `0.712395264`
- Built four hardlink package variants from the layer6/layer10 baseline package:
  - layer8 `up` single-token scale
  - layer8 `gate+up` single-token scale
  - layer8 `upfit` all-token scale
  - layer8 `gateupfit` all-token scale
- Full actual-prefix `0..12`, `rotary_dim=64`, sampled token `7`:
  - baseline layer6/layer10: max abs `0.645338058`, layer `11`, token `7`, hidden `3994`
  - layer8 `up`: max abs `0.653738022`
  - layer8 `gate+up`: max abs `0.655612946`
  - layer8 `upfit`: max abs `0.648880005`
  - layer8 `gateupfit`: max abs `0.650119781`
- Target coordinate comparison, layer `8`, token `7`, hidden `3994`:
  - baseline local delta error: `0.171178818`; MLP feature `6340` activation diff: `0.269271016`
  - `up`: local delta error `0.154756546`; feature diff `0.121943831`
  - `gate+up`: local delta error `0.150905609`; feature diff `0.087397814`
  - `upfit`: local delta error `0.164222717`; feature diff `0.206876397`
  - `gateupfit`: local delta error `0.161626816`; feature diff `0.183586121`
- Interpretation:
  - Internal MLP projection row-scale fixes part of the targeted token `7` feature drift, but overfits and worsens layer `8` token `3`, then layer `11` max.
  - All-token fit reduces overfit but still does not beat the layer6/layer10 baseline.
  - Simple scalar row-scale should not be promoted for `mlp.gate_proj[6340]` / `mlp.up_proj[6340]`; next direction is row reconstruction or quantizer-side calibration against multi-token downstream error.

## Layer 8 MLP Projection Row Reconstruction

- Checked whether broader package policies already exercise the suspicious layer `8` MLP projection rows.
  - For layer `8`, both `mlp.gate_proj.weight` and `mlp.up_proj.weight` use `aq4_e4m3_g16_ts_flloyd16` in the checked p4p46, p4p65, and p4p6 package variants.
  - Therefore the existing p4p65-inproj and p4p6 variants do not answer the current MLP-row question; they leave `mlp.gate_proj[6340]` and `mlp.up_proj[6340]` reconstructed the same way.
- Generated row reconstruction / dot-term artifacts:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer-module-trace-actual-input-rotary64-layer8-token7-hidden3994-mlp-proj-rowdot-tokenfit-dotterms-layer6-layer10-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer-module-trace-actual-input-rotary64-layer8-token7-hidden3994-mlp-proj-rowdot-tokenfit-p4p65-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer-module-trace-actual-input-rotary64-layer8-token7-hidden3994-mlp-proj-rowdot-tokenfit-p4p6.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-row-quant-error-layer8-mlp-gate-up-row6340-p4p46-inproj.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer8-token7-hidden3994-mlp-proj-row-reconstruction-summary.md`
- Row-level quantization error for layer `8`, row `6340` is modest:
  - `mlp.gate_proj.weight`: row RMS `0.000785230`, relative MSE `0.004946714`, max abs `0.003092449`
  - `mlp.up_proj.weight`: row RMS `0.000716250`, relative MSE `0.005367165`, max abs `0.003012434`
- The downstream row-dot error is dominated by activation-weighted terms rather than a uniformly bad row.
  - `mlp.up_proj[6340]`, token `7`: row-dot error `0.036712819`; column `3994` alone contributes `0.025054066` because input is `23.625000` and weight error is `0.001060490`.
  - `mlp.gate_proj[6340]`, token `7`: row-dot error `-0.072030053`; column `3994` contributes `-0.024613345`.
  - For `mlp.up_proj[6340]`, column `3994` is also the top dot-error term at token `3` and token `10`, contributing `0.021740036` and `0.025451750`.
  - For `mlp.gate_proj[6340]`, token `3` has total row-dot error `0.006678195` while column `3994` contributes `-0.021357612`, so other columns cancel it.
- Interpretation:
  - The current failure mode is not an obviously bad whole row; it is a high-activation column/group sensitivity problem.
  - This explains why scalar row-scale improves the target token but worsens the full-prefix max.
  - The next useful experiment should be sparse column/group compensation or activation-weighted quantization for sensitive MLP projection rows, evaluated over multiple tokens and downstream hidden error.

## Layer 8 MLP Sparse Cell-Delta Experiment

- Added a smoke-only `CELL_DELTA_OVERRIDES_JSON` argument to `package-golden-prefix-smoke`.
  - Schema version: `package-cell-delta-overrides-v0.1`
  - Each entry targets `(layer_index, tensor_suffix, row_index, col_index, delta)`.
  - The implementation applies `delta` to the materialized F32 matrix after AQ4 reconstruction and before runtime execution.
  - This is intentionally a validation hook, not package metadata promotion.
- Built two sparse correction files for layer `8`, row `6340`, column `3994`:
  - `mlp.up_proj.weight`: package `0.006980899721`, source `0.005920410156`, delta `-0.001060489565`
  - `mlp.gate_proj.weight`: package `-0.003788416740`, source `-0.002746582031`, delta `0.001041834708`
- Generated artifacts:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-cell-delta-overrides-layer8-up6340-col3994-p4p46-inproj.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-cell-delta-overrides-layer8-gateup6340-col3994-p4p46-inproj.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix0-12-rotary64-manifest-row-scale-layer6-layer10-cell-delta-layer8up6340col3994-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix0-12-rotary64-manifest-row-scale-layer6-layer10-cell-delta-layer8gateup6340col3994-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer8-token7-hidden3994-mlp-cell-delta-experiment-summary.md`
- Full actual-prefix `0..12`, `rotary_dim=64`, CPU:
  - baseline layer6/layer10 row-scale: layer `11` max abs `0.645338058`
  - up cell correction: layer `11` max abs `0.654584885`
  - gate+up cell correction: layer `11` max abs `0.654893875`
- Layer `8`, token `7`, hidden `3994` improves locally:
  - baseline output diff `0.296178818`
  - up cell correction `0.284414291`
  - gate+up cell correction `0.283128738`
- But layer `8`, token `3`, hidden `3994` worsens:
  - baseline output diff `-0.578010559`
  - up cell correction `-0.580806732`
  - gate+up cell correction `-0.583854675`
- Interpretation:
  - The sparse correction hook is valid and the intended materialized matrix cell changes are confirmed.
  - Returning one high-leverage cell to source weight is still not a globally useful correction.
  - The next target should be multi-token least-squares cell/group compensation or quantizer-side activation-weighted row/group calibration, not single-cell source restoration.

## Layer 8 MLP Sparse Cell-Delta LS Fit

- Fitted column `3994` cell deltas against all `16` tokens for row `6340`, minimizing package-vs-source row-dot error:
  - `mlp.up_proj[6340,3994]`: LS delta `-0.001297428526`; row-dot RMSE `0.032796258 -> 0.015284518`
  - `mlp.gate_proj[6340,3994]`: LS delta `0.001825227037`; row-dot RMSE `0.045137044 -> 0.019261286`
- Generated additional artifacts:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-cell-delta-overrides-layer8-up6340-col3994-lsfit-p4p46-inproj.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-cell-delta-overrides-layer8-gateup6340-col3994-lsfit-p4p46-inproj.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix0-12-rotary64-manifest-row-scale-layer6-layer10-cell-delta-layer8up6340col3994-lsfit-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix0-12-rotary64-manifest-row-scale-layer6-layer10-cell-delta-layer8gateup6340col3994-lsfit-p4p46-inproj.jsonl`
- Full actual-prefix `0..12`, `rotary_dim=64`, CPU:
  - up LS fit: layer `11` max abs `0.656669617`
  - gate+up LS fit: layer `11` max abs `0.657253265`
- Layer `8`, token `7`, hidden `3994` improves further:
  - up LS fit output diff `0.281785965`
  - gate+up LS fit output diff `0.279504776`
- But layer `8`, token `3`, hidden `3994` worsens further:
  - up LS fit output diff `-0.581432343`
  - gate+up LS fit output diff `-0.586801529`
- Interpretation:
  - Minimizing the selected row-dot error is still the wrong objective for full-prefix hidden max.
  - The next compensation experiment needs to optimize downstream hidden error directly, at least across token `3` and token `7`, rather than row-dot reconstruction alone.

## Layer 8 MLP Tracked Column Diagnostic

- Extended `export-qwen-layer-module-trace.py` to schema `qwen-layer-module-trace-v0.9`.
  - Added repeatable `--tracked-column COLUMN`.
  - `projection_row_dot` entries now keep `top_dot_error_terms` and also emit `tracked_dot_error_terms` for requested columns on every token.
- Generated tracked-column artifacts:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer-module-trace-actual-input-rotary64-layer8-token7-hidden3994-mlp-proj-rowdot-trackedcols-layer6-layer10-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer-module-trace-actual-input-rotary64-layer8-token7-hidden3994-mlp-proj-rowdot-trackedcols-layer6-layer10-p4p46-inproj.md`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer8-token7-hidden3994-mlp-tracked-column-summary.md`
- Tracked column set:
  - `22, 220, 310, 577, 933, 1304, 1571, 1679, 1726, 1778, 2086, 2560, 2805, 3098, 3115, 3384, 3461, 3608, 3842, 3908, 3994`
- Main finding:
  - `up_proj[6340,3994]` has high leverage but token `3` and token `7` inputs have the same sign, so it tends to move both in the same direction.
  - Some columns have opposite token signs and are better candidates for downstream fitting:
    - column `933`: token `3` input `-1.851562`, token `7` input `0.886719`
    - column `3461`: token `3` input `0.165039`, token `7` input `-1.992188`
    - column `3608`: token `3` input `0.398438`, token `7` input `-0.902344`
- Interpretation:
  - The next useful compensation search should fit a small set such as `up_proj[6340,{3994,933,3461,3608}]` against downstream hidden error.
  - It should allow deltas away from source weights, because the best downstream direction may differ from source restoration.

## Layer 8 Attention QKV V845 Cell-Delta Experiment

- Investigated the attention activation-path side of layer `8`, token `7`, hidden `3994`.
  - `attention_recurrent_v[845]` already differs by `0.009822965`.
  - This maps to `linear_attn.in_proj_qkv.weight` row `4096 + 845 = 4941`.
- From the tracked row-dot trace for `linear_attn.in_proj_qkv.weight[4941]`:
  - token `7` package-vs-source row-dot error: `-0.065318765`
  - top term: column `3994`, input `41.0`, weight error `-0.002373229`, dot term `-0.097302380`
  - source weight `0.010986328125`, package weight `0.008613099344`
- Extended smoke-only cell delta validation to allow `linear_attn.in_proj_qkv.weight`.
  - Added qkv source-restore override:
    - layer `8`
    - tensor suffix `linear_attn.in_proj_qkv.weight`
    - row `4941`
    - column `3994`
    - delta `0.002373228781`
- Generated artifacts:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-cell-delta-overrides-layer8-qkv-v845-col3994-p4p46-inproj.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix0-12-rotary64-manifest-row-scale-layer6-layer10-cell-delta-layer8qkv-v845col3994-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer8-attention-qkv-v845-cell-delta-summary.md`
- Full actual-prefix `0..12`, `rotary_dim=64`, CPU:
  - baseline layer6/layer10 row-scale overall max abs: `0.645338058`, layer `11`, token `7`, hidden `3994`
  - qkv V845 cell source-restore overall max abs: `0.627647400`, layer `7`, token `0`, hidden `3994`
  - layer `11` max improves from `0.645338058` to `0.619235992`
  - layer `8` local max worsens from `0.578010559` to `0.588329315`
- Interpretation:
  - This is the first sparse cell-delta experiment in this batch that improves full-prefix max.
  - The useful target is attention-side `linear_attn.in_proj_qkv` V row `4941`, not the MLP row `6340` alone.
  - The next promotion-oriented experiment should be row/group quantizer policy or compensation for the qkv V row around column/group `3994`, not a single smoke-only cell override.

## Layer 8 Attention QKV V845 Cell-Delta LS Fit

- Fitted a single-cell LS delta for `linear_attn.in_proj_qkv.weight[4941,3994]` over all `16` tokens:
  - source-restore delta: `0.002373228781`
  - LS delta: `0.001503075294`
  - row-dot RMSE: `0.082291864 -> 0.057757336`
- Generated artifacts:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-cell-delta-overrides-layer8-qkv-v845-col3994-lsfit-p4p46-inproj.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix0-12-rotary64-manifest-row-scale-layer6-layer10-cell-delta-layer8qkv-v845col3994-lsfit-p4p46-inproj.jsonl`
- Full actual-prefix `0..12`, `rotary_dim=64`, CPU:
  - qkv V845 source-restore: overall max abs `0.627647400`; layer `11` max abs `0.619235992`
  - qkv V845 LS fit: overall max abs `0.628797531`; layer `11` max abs `0.628797531`
  - layer `8` max is less worsened by LS fit (`0.584562302`) than source-restore (`0.588329315`), but layer `11` improves less.
- Interpretation:
  - For the current full-prefix max objective, source-restoring the qkv V845 column `3994` cell is a better single-cell probe than row-dot LS.
  - The result suggests the downstream objective weighs token `7` / layer `11` more strongly than all-token row-dot RMSE.

## Layer 8 Attention QKV V845 Group249 Source-Restore

- Generated group249 tracked-column trace for qkv V row `4941`, columns `3984..3999`.
  - Artifact: `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer-module-trace-actual-input-rotary64-layer8-token7-hidden3994-qkv-v845-group249-trackedcols-layer6-layer10-p4p46-inproj.jsonl`
- Built source-restore overrides for all `16` cells in qkv row `4941`, group `249`.
  - Artifact: `uLLM-project/benchmarks/results/2026-07-05/engine/package-cell-delta-overrides-layer8-qkv-v845-group249-p4p46-inproj.json`
  - Smoke: `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix0-12-rotary64-manifest-row-scale-layer6-layer10-cell-delta-layer8qkv-v845group249-p4p46-inproj.jsonl`
- Full actual-prefix `0..12`, `rotary_dim=64`, CPU:
  - qkv V845 single-cell source-restore:
    - overall max abs `0.627647400`
    - layer `8` max abs `0.588329315`
    - layer `11` max abs `0.619235992`
  - qkv V845 group249 source-restore:
    - overall max abs `0.627647400`
    - layer `8` max abs `0.585012436`
    - layer `11` max abs `0.621194839`
- Interpretation:
  - Restoring the full group reduces the layer `8` local worsening versus single-cell source-restore, but layer `11` remains slightly better with the single-cell correction.
  - Both qkv source-restore variants push the overall max back to the existing layer `7` floor.
  - The next major target is now layer `7`, token `0`, hidden `3994`, because it becomes the limiting max once the layer `11` chain is reduced.

## Layer 6 Hidden3994 MLP Down Row-Scale

- Localized the remaining layer `7`, token `0`, hidden `3994` floor.
  - layer `7` input diff: `-0.480636597`
  - layer `7` local delta diff: `-0.147010803`
  - layer `7` output diff: `-0.627647400`
  - Therefore most of the layer `7` max is inherited from layer `6`.
- Generated layer `6`, token `0`, hidden `3994` fullref/package comparison:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer-module-trace-actual-input-rotary64-layer6-token0-hidden3994-trace-layer6-layer10-p4p46-inproj.jsonl`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-module-trace-comparison-actual-input-rotary64-layer6-token0-hidden3994-layer6-layer10-p4p46-inproj.json`
- Layer `6`, token `0`, hidden `3994` local decomposition:
  - actual delta error `-0.730636597`
  - attention row-only / activation-path: `0.021879351` / `-0.096949037`
  - MLP row-only / activation-path: `-0.376916865` / `-0.269932446`
- `mlp.down_proj.weight[3994]` row-dot:
  - fullref module output `14.625000000`
  - source row dot `14.607685722`
  - package row dot `14.230768857`
  - package-source row-dot error `-0.376916865`
  - all-token LS row scale `1.02647171355`
- Generated row-scale artifact:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-row-scale-overrides-layer6-hidden3994-mlp-down-p4p46-inproj.json`
- Full actual-prefix `0..12`, `rotary_dim=64`, CPU:
  - baseline layer6/layer10 row-scale: overall max `0.645338058`
  - layer6 hidden3994 MLP down row-scale: overall max `0.637172699`
  - layer8 qkv V845 cell source-restore: overall max `0.627647400`
  - combined layer6 row-scale + layer8 qkv V845 cell: overall max `0.610977173`
- Layer detail, max abs:
  - baseline: layer6 `0.480636597`, layer7 `0.627647400`, layer8 `0.578010559`, layer11 `0.645338058`
  - layer6 row-scale: layer6 `0.465695381`, layer7 `0.428003311`, layer8 `0.565040588`, layer11 `0.637172699`
  - qkv V845 cell: layer6 `0.480636597`, layer7 `0.627647400`, layer8 `0.588329315`, layer11 `0.619235992`
  - combined: layer6 `0.465695381`, layer7 `0.428003311`, layer8 `0.575433731`, layer11 `0.610977173`
- Partial recheck on existing `prefix0-8-seq16` fixture:
  - fixture: `uLLM-project/benchmarks/golden/2026-07-05/qwen35-9b-prefix0-8-seq16`
  - baseline report: `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix0-8-seq16-rotary64-manifest-row-scale-layer6-layer10-p4p46-inproj.jsonl`
  - layer6 row-scale report: `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix0-8-seq16-rotary64-layer6h3994-row-scale-p4p46-inproj.jsonl`
  - baseline overall max `0.627647400` at layer `7`, token `0`, hidden `3994`
  - layer6 row-scale overall max `0.542758942` at layer `4`, token `14`, hidden `3994`
  - layer `7` max improves from `0.627647400` to `0.428003311`
- Interpretation:
  - layer6 row-scale and layer8 qkv V845 cell correction are complementary.
  - The combined smoke-only intervention is the current best result in this batch.
  - The existing shorter fixture partially reproduces the layer6 row-scale benefit.
  - A durable fix should now test whether these corrections generalize to a genuinely different prompt and whether they can be represented as quantizer/package metadata rather than smoke-only overrides.

## Combined Residual Debug: Layer11 Propagation and Layer10 MLP Probe

- Generated input-dump smoke for the current best combined condition:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix0-12-rotary64-combined-layer6h3994-layer8qkv-input-dump-sample-t7-p4p46-inproj.jsonl`
  - overall max abs remains `0.610977173`
  - max location remains layer `11`, token `7`, hidden `3994`
- Layer `11`, token `7`, hidden `3994` fullref/package comparison:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-module-trace-comparison-actual-input-rotary64-layer11-token7-hidden3994-combined-layer6h3994-layer8qkv-p4p46-inproj.json`
  - package output diff vs fixture: `-0.610977173`
  - package delta: `2.001220703`
  - fullref delta on package input: `1.987197876`
  - local delta error: `0.014022827`
  - Interpretation: layer `11` is mostly propagating inherited state drift, not creating the current max locally.
- Layer `10`, token `7`, hidden `3994` fullref/package comparison:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-module-trace-comparison-actual-input-rotary64-layer10-token7-hidden3994-combined-layer6h3994-layer8qkv-p4p46-inproj.json`
  - package output diff vs fixture: `-0.362197876`
  - package delta: `-0.577299118`
  - fullref delta on package input: `-0.715101242`
  - local delta error: `0.137802124`
  - attention row-only / activation-path: `0.002499638` / `-0.011271075`
  - MLP row-only / activation-path: `0.051982190` / `0.105653703`
  - Interpretation: layer `10` has a real local MLP error worth debugging further.
- Tested a layer `10` MLP gate single-cell source-restore probe:
  - target: `mlp.gate_proj.weight[9256,3994]`
  - source-restore delta: `-0.003059203`
  - override: `uLLM-project/benchmarks/results/2026-07-05/engine/package-cell-delta-overrides-layer8qkv-v845-layer10gate9256-col3994-p4p46-inproj.json`
  - smoke report: `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix0-12-rotary64-combined-layer6h3994-layer8qkv-layer10gate9256-cell-p4p46-inproj.jsonl`
  - result worsened overall max abs from `0.610977173` to `0.625913620`
  - layer `10`, token `7`, hidden `3994` worsened from `0.362197876` to `0.393999100`
  - layer `11`, token `7`, hidden `3994` worsened from `0.610977173` to `0.625913620`
- Tested a layer `10` MLP down row-scale probe:
  - target: `mlp.down_proj.weight[3994]`
  - all-token LS scale: `0.9639810684228307`
  - row-dot RMSE improved from `0.073699112` to `0.043323898`
  - override: `uLLM-project/benchmarks/results/2026-07-05/engine/package-row-scale-overrides-layer6h3994-layer10h3994-mlp-down-p4p46-inproj.json`
  - smoke report: `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix0-12-rotary64-combined-layer6h3994-layer10h3994-row-scale-layer8qkv-p4p46-inproj.jsonl`
  - result worsened overall max abs from `0.610977173` to `0.616283417`
  - layer `10` overall max improved from `0.463138580` to `0.437673450`, but token `7`, hidden `3994` worsened from `0.362197876` to `0.368671417`
  - layer `11`, token `7`, hidden `3994` worsened from `0.610977173` to `0.616283417`
- Summary artifact:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-combined-layer6h3994-layer8qkv-residual-debug-summary.md`
- Judgment:
  - This is still a problem worth continuing to debug.
  - The residual is traceable and not backend noise.
  - The next useful step should be broader layer `10` MLP group-level analysis or cross-fixture validation, not more isolated single-cell source-restore or one-row scaling by default.

## Cross-Fixture Recheck: Tokens101-116

- Exported a new golden fixture with different token ids:
  - fixture: `uLLM-project/benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16-tokens101-116`
  - token ids: `101..116`
  - layers: `0..12`
  - dtype: BF16 export to F32 golden tensors
- Compared four variants on the new fixture:
  - baseline report: `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix0-12-seq16-tokens101-116-rotary64-manifest-row-scale-layer6-layer10-p4p46-inproj.jsonl`
  - layer6 row-scale report: `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix0-12-seq16-tokens101-116-rotary64-layer6h3994-row-scale-p4p46-inproj.jsonl`
  - layer8 QKV cell report: `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix0-12-seq16-tokens101-116-rotary64-layer8qkv-v845-cell-p4p46-inproj.jsonl`
  - combined report: `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix0-12-seq16-tokens101-116-rotary64-combined-layer6h3994-layer8qkv-p4p46-inproj.jsonl`
- Overall max abs:
  - baseline: `1.080525398` at layer `7`, token `12`, hidden `3994`
  - layer6 row-scale: `1.043153763` at layer `7`, token `12`, hidden `3994`
  - layer8 QKV V845 cell: `1.080525398` at layer `7`, token `12`, hidden `3994`
  - combined: `1.043153763` at layer `7`, token `12`, hidden `3994`
- Layer detail:
  - baseline: layer6 `0.714679718`, layer7 `1.080525398`, layer11 `0.946708679`
  - layer6 row-scale: layer6 `0.652941704`, layer7 `1.043153763`, layer11 `0.916080475`
  - QKV cell: layer6 `0.714679718`, layer7 `1.080525398`, layer11 `0.969970703`
  - combined: layer6 `0.652941704`, layer7 `1.043153763`, layer11 `0.939382553`
- Summary artifact:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-cross-fixture-tokens101-116-summary.md`
- Interpretation:
  - layer6 row-scale partially generalizes to a genuinely different token fixture.
  - layer8 QKV V845 cell does not improve the new fixture's overall max and worsens layer `11` versus layer6 row-scale alone.
  - Durable-fix priority should move toward a quantizer/package policy for the layer6 hidden3994 MLP down row or a systematic row-bias mechanism.
- Additional layer6 local trace on `tokens101-116`:
  - input-dump smoke: `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix0-12-seq16-tokens101-116-rotary64-baseline-input-dump-sample-t12-p4p46-inproj.jsonl`
  - fullref trace: `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer-module-trace-actual-input-rotary64-layer6-token12-hidden3994-tokens101-116-baseline-p4p46-inproj.jsonl`
  - comparison: `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-module-trace-comparison-actual-input-rotary64-layer6-token12-hidden3994-tokens101-116-baseline-p4p46-inproj.json`
  - layer6 token12 hidden3994 local delta error: `0.0908833`
  - attention row-only / activation-path: `0.000485314` / `0.0120055`
  - MLP row-only / activation-path: `-0.0119364` / `0.106516`
  - original token ids `1..16` layer6 `mlp.down_proj[3994]` all-token scale: `1.026471714`, row-dot RMSE `0.117735388 -> 0.063680278`
  - tokens101-116 layer6 `mlp.down_proj[3994]` all-token scale: `1.023383096`, row-dot RMSE `0.131756300 -> 0.061972585`
  - Interpretation: the row-scale direction is stable across the two token fixtures.

## Cross-Fixture Recheck: Tokens201-216

- Exported another golden fixture with different token ids:
  - fixture: `uLLM-project/benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16-tokens201-216`
  - token ids: `201..216`
  - layers: `0..12`
  - dtype: BF16 export to F32 golden tensors
- Compared baseline package against the manifest-patched layer6 hidden3994 row-scale package:
  - baseline report: `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix0-12-seq16-tokens201-216-rotary64-manifest-row-scale-layer6-layer10-p4p46-inproj.jsonl`
  - layer6 row-scale report: `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix0-12-seq16-tokens201-216-rotary64-manifest-row-scale-layer6h3994-layer6-layer10-p4p46-inproj.jsonl`
- Overall max abs:
  - baseline: `1.140727997` at layer `11`, token `13`, hidden `3994`
  - layer6 hidden3994 row-scale: `1.145284653` at layer `11`, token `13`, hidden `3994`
- Layer detail:
  - baseline: layer6 `0.537414551`, layer7 `0.966460228`, layer11 `1.140727997`
  - layer6 row-scale: layer6 `0.476898193`, layer7 `0.497438431`, layer11 `1.145284653`
- Summary artifact:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-cross-fixture-tokens201-216-summary.md`
- Interpretation:
  - layer6 row-scale is a real local compensation candidate because it strongly reduces layer6/layer7 hidden3994 drift.
  - It is not safe to promote unconditionally based only on the current fixture set because the final layer11 max worsens slightly on tokens201-216.
  - The durable policy should optimize a multi-fixture objective or use a gated/targeted row-bias policy.
- Layer11 trace under the tokens201 row-scale condition:
  - input-dump smoke: `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix0-12-seq16-tokens201-216-rotary64-manifest-row-scale-layer6h3994-layer6-layer10-input-dump-sample-t13-p4p46-inproj.jsonl`
  - fullref trace: `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer-module-trace-actual-input-rotary64-layer11-token13-hidden3994-tokens201-216-layer6h3994-p4p46-inproj.jsonl`
  - comparison: `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-module-trace-comparison-actual-input-rotary64-layer11-token13-hidden3994-tokens201-216-layer6h3994-p4p46-inproj.json`
  - package output diff vs fixture: `-1.145284653`
  - local delta error: `0.104715`
  - attention row-only / activation-path: `0.073156` / `0.092791`
  - MLP row-only / activation-path: `0.052599` / `-0.039741`
  - `self_attention_o_proj[3994]` row-dot scale fit: `0.984954853`, RMSE `0.059855519 -> 0.024166208`
  - Interpretation: tokens201 has a remaining layer11 local component; this should be treated as a fixture-specific next candidate until cross-fixture validation exists.
- Tested that fixture-specific layer11 candidate:
  - override: `uLLM-project/benchmarks/results/2026-07-05/engine/package-row-scale-overrides-layer11-self-attn-o3994-tokens201-p4p46-inproj.json`
  - report: `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix0-12-seq16-tokens201-216-rotary64-manifest-row-scale-layer6h3994-layer6-layer10-cli-row-scale-layer11-o3994-p4p46-inproj.jsonl`
  - result worsened from `1.145284653` to `1.206287384`
  - Interpretation: row-dot RMSE fit alone is not enough; end-to-end smoke remains required before promoting row-scale candidates.

## Manifest Metadata Prototype: Layer6 Hidden3994 Row-Scale

- Created a hardlink package copy outside the repository:
  - source: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer10.ullm.d`
  - patched: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6h3994-layer6-layer10.ullm.d`
- Added one manifest row-scale entry:
  - tensor: `model.language_model.layers.6.mlp.down_proj.weight`
  - row: `3994`
  - scale: `1.02647171355`
  - source: `golden-prefix-row-dot-sensitivity-layer6-hidden3994`
- Added manifest row-scale JSON artifact:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/package-row-scale-overrides-layer6h3994-layer6-layer10-p4p46-inproj.json`
  - schema: `row-scale-overrides-v0.1`
  - entries: `5`
- Manifest-only smoke, no CLI row-scale:
  - report: `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix0-12-rotary64-manifest-row-scale-layer6h3994-layer6-layer10-p4p46-inproj.jsonl`
  - `manifest_row_scale_overrides=5`
  - max abs `0.637172699`
  - layers `6`, `7`, `8`, and `11` match the CLI layer6 row-scale report exactly.
- Manifest layer6 row-scale + QKV cell smoke:
  - report: `uLLM-project/benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-actual-prefix0-12-rotary64-manifest-row-scale-layer6h3994-layer6-layer10-cell-delta-layer8qkv-v845col3994-p4p46-inproj.jsonl`
  - `manifest_row_scale_overrides=5`
  - max abs `0.610977173`
  - matches the current best CLI row-scale + QKV cell result.
- Interpretation:
  - layer6 hidden3994 row-scale can be represented as package manifest metadata.
  - Promotion should still be gated by multi-fixture end-to-end results.
  - layer8 QKV V845 remains smoke-only and should not be promoted without broader fixture support.
- Validation:
  - JSON duplicate/positive-scale check passed.
  - `cargo test -p ullm-engine row_scale -- --nocapture` passed.
  - `cargo test -p ullm-quant row_scale -- --nocapture` exited successfully with `0` matching tests.

## Trace Tooling: Row-Dot Scale Fit

- Updated `tools/export-qwen-layer-module-trace.py`:
  - schema version: `qwen-layer-module-trace-v0.9 -> v0.10`
  - added `row_dot.<projection>.scale_fit`
  - fields include `optimal_scale`, `original_rmse`, `scaled_rmse`, `rmse_improvement_ratio`, max/mean errors, and worst token indices.
- Re-generated the tokens101 layer6 trace:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer-module-trace-actual-input-rotary64-layer6-token12-hidden3994-tokens101-116-baseline-p4p46-inproj.jsonl`
  - `mlp_down_proj.scale_fit.optimal_scale`: `1.0233830958654118`
  - `original_rmse -> scaled_rmse`: `0.13175630023164644 -> 0.061972584645991806`
- Validation:
  - `python3 -m py_compile tools/export-qwen-layer-module-trace.py` passed.
  - `tools/compare-qwen-module-trace.py` successfully consumed the `v0.10` trace.

## Multi-Fixture Row-Scale Summary

- Added consolidated summary:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-row-scale-multi-fixture-summary.md`
- Key table:
  - token ids `1..16`: baseline `0.645338058`, layer6 row-scale `0.637172699`, qkv cell `0.627647400`, combined `0.610977173`
  - token ids `101..116`: baseline `1.080525398`, layer6 row-scale `1.043153763`, qkv cell `1.080525398`, combined `1.043153763`
  - token ids `201..216`: baseline `1.140727997`, layer6 row-scale `1.145284653`, layer6+layer11 o row-scale `1.206287384`
- Final interpretation for now:
  - Continue debugging.
  - Use multi-fixture end-to-end acceptance gates.
  - Do not promote row-scale candidates solely from local row-dot RMSE improvement.

## Next Plan

- Added plan:
  - `uLLM-project/docs/plans/multi-fixture-row-scale-validation-plan-v0.1.md`
- Proposed next implementation:
  - report aggregator for prefix smoke JSONL files
  - row-scale candidate extractor from `row_dot.<projection>.scale_fit`
  - sequential multi-fixture smoke runner
  - acceptance summary with hard reject thresholds
- Main policy:
  - layer6 hidden3994 row-scale is a local compensation candidate, not an unconditional promotion yet.
  - layer8 QKV V845 and layer11 tokens201 row candidates remain smoke-only/rejected until broader support exists.
- Implemented first deliverable from the plan:
  - `uLLM-project/tools/summarize-qwen-prefix-smokes.py`
  - generated `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-prefix-smoke-multi-fixture-summary.json`
  - generated `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-prefix-smoke-multi-fixture-summary.md`
  - validation: `python3 -m py_compile tools/summarize-qwen-prefix-smokes.py` passed.
- Implemented row-scale candidate extraction:
  - `uLLM-project/tools/extract-qwen-row-scale-candidates.py`
  - generated `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-row-scale-candidates-from-traces.json`
  - generated `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-row-scale-candidates-from-traces-manifest.json`
  - generated `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-row-scale-candidates-from-traces-smoke.json`
  - generated `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-row-scale-candidates-from-traces.md`
  - extracted three available v0.10 trace candidates with `min_rmse_improvement=0.05`:
    - layer6 `linear_attn.out_proj.weight[3994]`, scale `0.992316548`
    - layer6 `mlp.down_proj.weight[3994]`, scale `1.023383096`
    - layer11 `self_attn.o_proj.weight[3994]`, scale `0.984954853`
  - validation: `python3 -m py_compile tools/extract-qwen-row-scale-candidates.py` and JSON parsing passed.
- Refreshed tokens1 layer6 hidden3994 trace to v0.10:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-layer-module-trace-actual-input-rotary64-layer6-token0-hidden3994-trace-layer6-layer10-p4p46-inproj-v0.10.jsonl`
  - `mlp_down_proj.scale_fit.optimal_scale`: `1.0264717135497252`
  - `original_rmse -> scaled_rmse`: `0.11773538751126016 -> 0.06368027788589073`
  - candidate extraction now records two observations for layer6 `mlp.down_proj.weight[3994]`.
- Implemented prefix candidate gate evaluation:
  - `uLLM-project/tools/evaluate-qwen-prefix-candidate-gates.py`
  - generated `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-prefix-candidate-gates.json`
  - generated `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-prefix-candidate-gates.md`
  - gate settings: max fixture worsen `0.001`, min median improvement `0.005`, min fixture count `3`
  - `layer6` decision: `reject`, because tokens201 worsens by `0.00455665588`
  - `combined` decision: `needs_more_fixtures`, because only tokens1 has a paired baseline/candidate result in the current summary.
  - validation: `python3 -m py_compile tools/evaluate-qwen-prefix-candidate-gates.py` and JSON parsing passed.
- Implemented sequential smoke matrix runner:
  - `uLLM-project/tools/run-qwen-prefix-smoke-matrix.py`
  - runs fixture/condition matrices one smoke at a time to keep memory bounded.
  - supports condition-specific package, row-scale JSON, and cell-delta JSON.
  - dry-run verified with tokens1/tokens101 fixtures and baseline/layer6 conditions.
  - validation: `python3 -m py_compile tools/run-qwen-prefix-smoke-matrix.py` passed.
- Ran the extracted three-row candidate set across three fixtures:
  - runner summary: `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-prefix-smoke-matrix-extracted-candidates/summary.md`
  - smoke summary: `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-prefix-smoke-extracted-candidates-summary.md`
  - gate result: `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-prefix-extracted-candidate-gates.md`
  - tokens1: `0.645338058 -> 0.676628113`
  - tokens101: `1.080525398 -> 1.039905548`
  - tokens201: `1.140727997 -> 1.213508606`
  - decision: `reject`, because max regression is `0.0727806091`.
  - interpretation: candidate extraction is useful for proposing rows, but extracted rows must not be applied as a bundle without gate search.
- Ran layer6 attention+MLP candidate set across three fixtures:
  - override: `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-row-scale-candidates-layer6-attn-mlp-smoke.json`
  - runner summary: `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-prefix-smoke-matrix-layer6-attn-mlp/summary.md`
  - smoke summary: `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-prefix-smoke-layer6-attn-mlp-summary.md`
  - gate result: `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-prefix-layer6-attn-mlp-gates.md`
  - tokens1: `0.645338058 -> 0.639551163`
  - tokens101: `1.080525398 -> 1.039905548`
  - tokens201: `1.140727997 -> 1.152515411`
  - decision: `reject`, because max regression is `0.0117874146`.
  - interpretation: layer6 attention row-scale improves two fixtures but worsens tokens201 more than MLP-only; do not promote it.
- Ran selected-scale layer6 MLP-only candidate across three fixtures:
  - override: `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-row-scale-candidates-layer6-mlp-selected-smoke.json`
  - runner summary: `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-prefix-smoke-matrix-layer6-mlp-selected/summary.md`
  - smoke summary: `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-prefix-smoke-layer6-mlp-selected-summary.md`
  - gate result: `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-prefix-layer6-mlp-selected-gates.md`
  - tokens1: `0.645338058 -> 0.638132095`
  - tokens101: `1.080525398 -> 1.047520638`
  - tokens201: `1.140727997 -> 1.144765854`
  - decision: `reject`, because max regression is `0.00403785706`.
  - interpretation: selected lower scale reduces tokens201 regression slightly versus `1.026471714`, but not enough to pass the hard gate.
- Updated plan progress:
  - `uLLM-project/docs/plans/multi-fixture-row-scale-validation-plan-v0.1.md`
  - recorded implemented tools and current gate outcomes.
- Added formal report:
  - `uLLM-project/docs/research/qwen-prefix-row-scale-debug-report-2026-07-05.md`
  - includes previous context, current changes, gate result, key evidence, verification, and next action.
- Final verification bundle passed:
  - `python3 -m py_compile` for `summarize-qwen-prefix-smokes.py`, `extract-qwen-row-scale-candidates.py`, `evaluate-qwen-prefix-candidate-gates.py`, `run-qwen-prefix-smoke-matrix.py`, and `export-qwen-layer-module-trace.py`
  - `python3 -m json.tool` for the main generated summary, candidate, gate, matrix, layer6 MLP selected-scale, layer6 attention+MLP, extracted-candidate, and v0.10 trace JSON artifacts.
- Added a resolution-focused plan:
  - `uLLM-project/docs/plans/qwen-prefix-hidden3994-resolution-plan-v0.1.md`
  - goal: solve the hidden3994 drift, not merely evaluate candidates.
  - includes success criteria, hypotheses, fixture expansion, v0.10 trace completion, tokens201 layer11 regression localization, gated candidate search, manifest-vs-quantizer decision path, backend verification, and next actions.

## 12:25 JST Continuation

- Continued the hidden3994 resolution work under the 18:00 JST stop/report goal.
- Added two additional Qwen3.5-9B golden fixtures:
  - `uLLM-project/benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16-tokens301-316`
  - `uLLM-project/benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16-tokens401-416`
- Ran baseline CPU prefix smoke for the two new fixtures:
  - `tokens301`: overall max `1.37130928`, layer `10`, token `12`, hidden `3994`
  - `tokens401`: overall max `0.959306717`, layer `8`, token `9`, hidden `3994`
- Produced the five-fixture baseline summary:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-prefix-baseline-five-fixture-summary.json`
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-prefix-baseline-five-fixture-summary.md`
- Committed the extra fixture/baseline artifacts:
  - `439d7fb Add extra Qwen prefix baseline fixtures`
- Implemented row-scale grid generation:
  - `uLLM-project/tools/generate-qwen-row-scale-grid.py`
  - output schema: `qwen-row-scale-grid-v0.1`
  - smoke override schema: `package-row-scale-overrides-v0.1`
- Extended `tools/summarize-qwen-prefix-smokes.py` so it can read matrix runner summaries through `--matrix-summary-json`.
- Generated a layer6 MLP hidden3994 coarse grid:
  - `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-row-scale-grid-layer6-mlp-h3994/summary.md`
  - scales: `1.000`, `1.004`, `1.008`, `1.012`, `1.016`, `1.020`, `1.023383096`, `1.026471714`
- Started the five-fixture CPU smoke matrix:
  - output: `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-prefix-smoke-matrix-layer6-mlp-grid5/`
  - fixtures: `tokens1`, `tokens101`, `tokens201`, `tokens301`, `tokens401`
  - conditions: `baseline` plus the eight grid scales.
- Validation so far:
  - `python3 -m py_compile tools/generate-qwen-row-scale-grid.py tools/summarize-qwen-prefix-smokes.py` passed.

## 14:30 JST Five-Fixture Gate Update

- Completed the five-fixture layer6 MLP hidden3994 row-scale grid:
  - matrix: `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-prefix-smoke-matrix-layer6-mlp-grid5/summary.md`
  - smoke summary: `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-prefix-layer6-mlp-grid5-summary.md`
  - gate summary: `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-prefix-layer6-mlp-grid5-gates.md`
- Result:
  - all positive layer6 MLP hidden3994 scales are rejected under the five-fixture gate.
  - tokens401 is the strongest counterexample.
  - scale `1.004` would stay within the tokens201 hard gate, but worsens tokens401 by `+0.00535869598`.
  - scale `1.026471714` improves tokens1, tokens101, and tokens301, but worsens tokens201 and tokens401.
- Implemented/updated tooling:
  - added `uLLM-project/tools/generate-qwen-row-scale-grid.py`
  - extended `uLLM-project/tools/summarize-qwen-prefix-smokes.py` with `--matrix-summary-json`.
- Localized the tokens401 counterexample:
  - baseline worst coordinate: layer8 token9 hidden3994, diff `-0.959306717`
  - layer8 input at token9 hidden3994 is already low by `-0.464719772`
  - full-reference layer8 replay on the package actual input outputs `-1.0` vs the golden fixture output
  - package-vs-full-reference actual-input delta error at layer8 token9 hidden3994 is only `+0.0406933`
  - layer7 token10 hidden3994 shows the same pattern: full-reference actual-input replay is already `-0.875`, package-vs-full-reference delta error is only `-0.0277519`
  - interpretation: tokens401 is primarily input-drift amplification, not a single layer8 row-quantization-only miss.
- Checked whether the existing manifest row-scale should be removed:
  - no-row-scale candidate is rejected.
  - it worsens tokens1, tokens101, tokens301, and tokens401 heavily.
  - conclusion: existing row3456 manifest compensation remains necessary.
- Checked package-level p4p65 candidates:
  - `p4p65-inproj` without row3456 compensation is rejected.
  - `p4p65+row3456` is also rejected; it helps row3456 but worsens hidden3994, especially tokens401.
  - conclusion: simple p4p65 replacement does not solve the hidden3994 drift.
- Updated documentation:
  - `uLLM-project/docs/research/qwen-prefix-row-scale-debug-report-2026-07-05.md`
  - `uLLM-project/docs/plans/multi-fixture-row-scale-validation-plan-v0.1.md`

## 15:25 JST Layer8 Manifest Package Probe

- Evaluated existing layer8 gate/up manifest packages across five fixtures:
  - matrix: `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-prefix-smoke-matrix-layer8-manifest-packages-five-fixture/summary.md`
  - gate: `uLLM-project/benchmarks/results/2026-07-05/engine/qwen-prefix-manifest-vs-layer8-manifest-packages-five-fixture-gates.md`
- Result:
  - all four layer8 manifest packages are rejected under the five-fixture hard gate.
  - `layer8-upfit` is the closest candidate:
    - tokens201 improves `1.140727997 -> 1.13804817`
    - tokens301 improves slightly `1.371309280 -> 1.37123108`
    - tokens401 stays inside hard gate `0.959306717 -> 0.959452629`
    - tokens1 fails with regression `0.645338058 -> 0.648880005`
- Tried to create a weaker layer8 `mlp.up_proj.weight[6340]` smoke grid, but `package-golden-prefix-smoke` supports row-scale overrides only for:
  - `linear_attn.out_proj.weight`
  - `self_attn.o_proj.weight`
  - `mlp.down_proj.weight`
- Removed the failed smoke output and updated `tools/generate-qwen-row-scale-grid.py` to reject unsupported tensor suffixes before running engine smokes.

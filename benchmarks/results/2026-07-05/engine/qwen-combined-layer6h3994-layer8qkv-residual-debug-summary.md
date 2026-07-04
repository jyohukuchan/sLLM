# Combined Layer6 Hidden3994 + Layer8 QKV Residual Debug Summary

## Context

- Package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer10.ullm.d`
- Fixture: `benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16`
- Run mode: `actual_prefix`
- Rotary: `rotary_dim=64`, `rope_base=10000000`, `position_offset=0`
- Baseline for this summary:
  - manifest layer6/layer10 row-scale overrides
  - CLI layer6 `mlp.down_proj.weight[3994]` row-scale
  - layer8 `linear_attn.in_proj_qkv.weight[4941,3994]` source-restore cell delta

## Current Best Result

- Report: `package-golden-prefix-cpu-actual-prefix0-12-rotary64-combined-layer6h3994-layer8qkv-input-dump-sample-t7-p4p46-inproj.jsonl`
- Overall max abs: `0.610977173`
- Max location: layer `11`, token `7`, hidden `3994`
- Layer detail:
  - layer6 max abs: `0.465695381`
  - layer7 max abs: `0.428003311`
  - layer8 max abs: `0.575433731`
  - layer10 max abs: `0.463138580`
  - layer11 max abs: `0.610977173`

## Layer11 Residual Localization

- Fullref trace:
  - `qwen-layer-module-trace-actual-input-rotary64-layer11-token7-hidden3994-combined-layer6h3994-layer8qkv-p4p46-inproj.jsonl`
- Comparison:
  - `qwen-module-trace-comparison-actual-input-rotary64-layer11-token7-hidden3994-combined-layer6h3994-layer8qkv-p4p46-inproj.json`
- Decomposition:
  - package output diff vs fixture: `-0.610977173`
  - package delta: `2.001220703`
  - fullref delta on package input: `1.987197876`
  - local delta error: `0.014022827`
  - attention row-only / activation-path: `0.068709255` / `-0.059793750`
  - MLP row-only / activation-path: `-0.028733817` / `-0.012195678`
- Interpretation:
  - Layer11 is not the main local source of the current max.
  - The `0.610977173` output diff is mostly inherited state drift from earlier layers and then propagated by layer11.

## Layer10 Residual Localization

- Fullref trace:
  - `qwen-layer-module-trace-actual-input-rotary64-layer10-token7-hidden3994-combined-layer6h3994-layer8qkv-p4p46-inproj.jsonl`
- Comparison:
  - `qwen-module-trace-comparison-actual-input-rotary64-layer10-token7-hidden3994-combined-layer6h3994-layer8qkv-p4p46-inproj.json`
- Decomposition:
  - package output diff vs fixture: `-0.362197876`
  - package delta: `-0.577299118`
  - fullref delta on package input: `-0.715101242`
  - local delta error: `0.137802124`
  - attention row-only / activation-path: `0.002499638` / `-0.011271075`
  - MLP row-only / activation-path: `0.051982190` / `0.105653703`
- MLP down row-dot for token7 hidden3994:
  - fullref module output: `0.022094727`
  - source row dot: `0.022073944`
  - package row dot: `0.074056134`
  - package-source row-dot error: `0.051982190`
- Top MLP activation candidate:
  - `mlp_activation` top1: feature `9256`, diff `-0.187328`
  - `mlp_gate_projection[9256,3994]` tracked dot-error term: `+0.081068874`
  - package weight: `-0.026970094`
  - source weight: `-0.030029297`
  - source-restore delta: `-0.003059203`

## Rejected Layer10 Gate Single-Cell Probe

- Override:
  - `package-cell-delta-overrides-layer8qkv-v845-layer10gate9256-col3994-p4p46-inproj.json`
- Report:
  - `package-golden-prefix-cpu-actual-prefix0-12-rotary64-combined-layer6h3994-layer8qkv-layer10gate9256-cell-p4p46-inproj.jsonl`
- Result:
  - current best without layer10 gate cell: `0.610977173`
  - with layer10 gate cell source-restore: `0.625913620`
  - layer10 token7 hidden3994 worsened: `0.362197876 -> 0.393999100`
  - layer11 token7 hidden3994 worsened: `0.610977173 -> 0.625913620`
- Interpretation:
  - The layer10 MLP local error is real, but this single gate cell is not a good end-to-end correction.
  - It likely removes one local projection error while disturbing compensating effects through the nonlinear MLP path.

## Rejected Layer10 MLP Down Row-Scale Probe

- All-token LS scale for `mlp.down_proj.weight[3994]` on the combined actual-input trace:
  - scale: `0.9639810684228307`
  - row-dot RMSE: `0.073699112 -> 0.043323898`
- Override:
  - `package-row-scale-overrides-layer6h3994-layer10h3994-mlp-down-p4p46-inproj.json`
- Report:
  - `package-golden-prefix-cpu-actual-prefix0-12-rotary64-combined-layer6h3994-layer10h3994-row-scale-layer8qkv-p4p46-inproj.jsonl`
- Result:
  - current best without layer10 row-scale: `0.610977173`
  - with layer10 row-scale: `0.616283417`
  - layer10 overall max moved from `0.463138580` at token `1`, hidden `3994` to `0.437673450` at token `13`, hidden `2479`
  - layer10 token `7`, hidden `3994` worsened from `0.362197876` to `0.368671417`
  - layer11 token `7`, hidden `3994` worsened from `0.610977173` to `0.616283417`
- Interpretation:
  - The row-scale improves the layer10 down-row dot fit, but it does not improve the end-to-end residual path.
  - The current fixture is sensitive to compensation between inherited state drift and layer10/layer11 transformations.

## Debugging Judgment

This remains worth debugging. The residual is not backend noise, and the current max has a traceable path:

1. Layer6 row-scale and layer8 QKV cell probes are complementary and improved the full prefix max from `0.645338058` to `0.610977173`.
2. Layer11 is mostly propagation, not a local implementation bug at the current input state.
3. Layer10 has a meaningful local MLP error (`0.137802124`), but both the first single-cell gate correction and the all-token down-row scale worsened the full-prefix objective.

The next useful step is not another isolated single-cell restore by default. Better candidates are:

- Trace layer10 MLP over several tokens/features and test a broader group-level policy instead of one gate cell or one down row.
- Re-run the current best across additional fixtures to separate general quantization bias from prompt-specific compensation.
- Promote only corrections that improve both local fullref comparison and full-prefix objective across more than one fixture.

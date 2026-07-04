# Multi-Fixture Row-Scale Validation Plan v0.1

## Current Finding

Layer6 `mlp.down_proj.weight[3994]` row-scale is a real local compensation candidate, but it is not yet safe as an unconditional package policy.

Evidence:

- Token ids `1..16`:
  - baseline `0.645338058`
  - layer6 hidden3994 row-scale `0.637172699`
  - layer6 row-scale + layer8 QKV cell `0.610977173`
- Token ids `101..116`:
  - baseline `1.080525398`
  - layer6 hidden3994 row-scale `1.043153763`
- Token ids `201..216`:
  - baseline `1.140727997`
  - layer6 hidden3994 row-scale `1.145284653`
  - layer6/layer7 improve, but final layer11 max worsens slightly.

The key rule from this batch is: row-dot RMSE improvement is useful for proposing candidates, but full-prefix multi-fixture smoke must decide whether a candidate is accepted.

## Goal

Build a repeatable validation loop that turns row-dot traces into row-scale candidates, evaluates them across multiple fixtures, and accepts only candidates that improve the chosen multi-fixture objective without unacceptable regressions.

## Proposed Acceptance Gates

For a row-scale candidate to be promoted to manifest metadata:

1. It must improve or preserve the aggregate objective across all fixtures.
2. It must not worsen any fixture's final max abs by more than a configured tolerance.
3. It must improve at least one targeted local row-dot metric.
4. It must pass full-prefix smoke on CPU.
5. GPU/backend checks should be repeated only after CPU multi-fixture acceptance.

Initial tolerances:

- Hard reject if any fixture worsens final max abs by more than `0.001`.
- Prefer candidates that improve median final max abs by at least `0.005`.
- Track per-layer max abs and mean abs because some candidates improve early layers while worsening layer11.

## Implementation Steps

1. Add a report aggregator for `package-golden-prefix-smoke` JSONL files.
   - Input: named report paths.
   - Output: JSON + markdown table with overall max, per-layer max, location, mean abs, cosine.
2. Add a row-scale candidate extractor from `qwen-layer-module-trace` JSONL.
   - Use `row_dot.<projection>.scale_fit`.
   - Emit both manifest-schema and smoke-only CLI-schema row-scale JSON.
3. Add a multi-fixture smoke runner.
   - Inputs: package path, fixture list, row-scale candidate files.
   - Runs baseline and candidate conditions sequentially.
   - Keeps memory bounded by one smoke at a time.
4. Add an acceptance summary.
   - Compare baseline/candidate across fixtures.
   - Flag improvements, regressions, and hard rejects.
5. Re-test layer6 hidden3994 with this loop.
   - Expected result: accepted as local candidate, rejected or held as unconditional promotion because tokens201 worsens final max.
6. Search for paired candidates that reduce the tokens201 layer11 regression.
   - Start from layer11 token13 hidden3994 traces.
   - Do not promote single-fixture layer11 candidates without cross-fixture validation.

## Progress 2026-07-05

Implemented:

- `tools/summarize-qwen-prefix-smokes.py`
  - Generated `benchmarks/results/2026-07-05/engine/qwen-prefix-smoke-multi-fixture-summary.json`
  - Generated `benchmarks/results/2026-07-05/engine/qwen-prefix-smoke-multi-fixture-summary.md`
- `tools/extract-qwen-row-scale-candidates.py`
  - Generated candidate, manifest-schema, smoke-schema, and markdown outputs from available v0.10 traces.
- `tools/run-qwen-prefix-smoke-matrix.py`
  - Runs fixture/condition matrices sequentially to keep memory bounded.
  - Dry-run verified with two fixtures and baseline/layer6 conditions.
- `tools/evaluate-qwen-prefix-candidate-gates.py`
  - Generated `benchmarks/results/2026-07-05/engine/qwen-prefix-candidate-gates.json`
  - Generated `benchmarks/results/2026-07-05/engine/qwen-prefix-candidate-gates.md`

Current gate result:

| condition | decision | fixtures | median improvement | max regression | reason |
| --- | --- | ---: | ---: | ---: | --- |
| `layer6` | reject | 3 | 0.0081653595 | 0.00455665588 | tokens201 regression exceeds `0.001` |
| `combined` | needs_more_fixtures | 1 | 0.0343608856 | 0 | only one paired fixture |

Interpretation: layer6 hidden3994 remains a real local compensation candidate, but it should not be promoted unconditionally under the initial multi-fixture gate.

## Deliverables

- `tools/summarize-qwen-prefix-smokes.py`
- `tools/extract-qwen-row-scale-candidates.py`
- `tools/run-qwen-prefix-smoke-matrix.py`
- `tools/evaluate-qwen-prefix-candidate-gates.py`
- `benchmarks/results/2026-07-05/engine/qwen-row-scale-multi-fixture-summary.md`
- Updated manifest row-scale candidate JSON files.
- A final decision table marking each candidate as accepted, rejected, or needs-more-fixtures.

## Non-Goals

- Do not hard-code layer6 hidden3994 into runtime logic.
- Do not promote layer8 QKV V845 cell until it improves more than one fixture.
- Do not use row-dot RMSE alone as an acceptance metric.

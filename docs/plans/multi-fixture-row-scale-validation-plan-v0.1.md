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
| `layer6-mlp-selected` | reject | 3 | 0.00720596313 | 0.00403785706 | tokens201 regression exceeds `0.001` |
| `layer6-attn-mlp` | reject | 3 | 0.00578689575 | 0.0117874146 | tokens201 regression exceeds `0.001` |
| `combined` | needs_more_fixtures | 1 | 0.0343608856 | 0 | only one paired fixture |
| `extracted` | reject | 3 | -0.0312900543 | 0.0727806091 | tokens1 and tokens201 regress |

Interpretation: layer6 hidden3994 remains a real local compensation candidate, but it should not be promoted unconditionally under the initial multi-fixture gate.
The lower tokens101-selected MLP scale reduces the tokens201 regression slightly, but still fails the hard gate.
Adding the layer6 attention row-scale improves tokens1/tokens101 but increases the tokens201 regression relative to MLP-only.
The three-row extracted candidate bundle confirms the same rule: row-dot candidates should feed a gated search loop, not direct bundle promotion.

Additional five-fixture result:

- Added token ids `301..316` and `401..416` as extra golden fixtures.
- Baseline worst coordinate remains hidden `3994` in all five fixtures:
  - `tokens1`: `0.645338058`, layer11 token7 hidden3994
  - `tokens101`: `1.080525398`, layer7 token12 hidden3994
  - `tokens201`: `1.140727997`, layer11 token13 hidden3994
  - `tokens301`: `1.371309280`, layer10 token12 hidden3994
  - `tokens401`: `0.959306717`, layer8 token9 hidden3994
- Implemented `tools/generate-qwen-row-scale-grid.py` and generated a layer6 MLP hidden3994 grid for scales `1.000`, `1.004`, `1.008`, `1.012`, `1.016`, `1.020`, `1.023383096`, and `1.026471714`.
- Five-fixture grid result:

| condition | decision | fixtures | median improvement | max regression | reason |
| --- | --- | ---: | ---: | ---: | --- |
| `layer6-mlp-h3994-s1p004` | reject | 5 | 0.000791549683 | 0.00535869598 | tokens401 regression exceeds `0.001` |
| `layer6-mlp-h3994-s1p008` | reject | 5 | 0.00157546997 | 0.0107059479 | tokens201 and tokens401 regress |
| `layer6-mlp-h3994-s1p026471714` | reject | 5 | 0.00524330139 | 0.0354146957 | tokens401 regression dominates |

The important update is that tokens401 is now the strongest counterexample for layer6 MLP hidden3994 scaling.
Scale `1.004` would have stayed within the tokens201 hard gate, but it still worsens tokens401 by `0.00535869598`.
Therefore, layer6 MLP hidden3994 row-scale is rejected as an unconditional package candidate under the five-fixture gate.

No-row-scale comparison:

- The existing manifest row-scale entries target row3456, not hidden3994.
- Removing manifest row-scale worsens four of five fixtures:
  - tokens1: `0.645338058 -> 1.74426651`
  - tokens101: `1.080525398 -> 1.50819016`
  - tokens301: `1.371309280 -> 2.50828171`
  - tokens401: `0.959306717 -> 1.45381165`
- Keep the existing row3456 manifest compensation while investigating hidden3994.

Tokens401 localization:

- Layer8 token9 hidden3994 is the baseline max coordinate for tokens401.
- Actual-prefix layer8 input at token9 hidden3994 is already low by `-0.464719772`.
- Full-reference layer8 replay on that actual input outputs `-1.0` vs the golden fixture output.
- Package-vs-full-reference actual-input delta error at layer8 token9 hidden3994 is only `+0.0406933`, so the major issue is input-drift amplification rather than a layer8 row-quantization-only miss.
- Layer7 token10 hidden3994 shows the same pattern: full-reference actual-input replay is already `-0.875`, while package-vs-full-reference delta error is `-0.0277519`.

Next branch:

- Evaluate package-level quantization policy candidates, starting with the existing `p4p65-inproj` package.
- Only return to paired row-scale candidates if a candidate addresses tokens401 without worsening tokens1/tokens101/tokens201/tokens301.

Quantization-policy probe result:

- `p4p65-inproj` without row3456 compensation fails the five-fixture gate:
  - median improvement: `-0.797094345`
  - max regression: `1.39915657`
  - main failure: row3456 regressions on tokens1/tokens101/tokens301/tokens401.
- `p4p65` with the existing row3456 smoke overrides also fails:
  - median improvement: `-0.0370130539`
  - max regression: `0.26203537`
  - main failure: hidden3994 regression on tokens401, `0.959306717 -> 1.22134209`.
- Therefore, the next package-level candidate must preserve row3456 compensation and address hidden3994 input-drift amplification more directly than a simple p4p65 swap.

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

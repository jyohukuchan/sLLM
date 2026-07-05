# Qwen Prefix Row-Scale Debug Report 2026-07-05

## 前回の要点

- Qwen3.5-9B p4p46 in-projection packageのprefix smokeでは、hidden `3994` の差分が複数fixtureで支配的になっていた。
- Layer6 `mlp.down_proj.weight[3994]` row-scaleは、layer6/layer7の局所差分を強く下げる有望候補だった。
- ただし、単一fixtureの局所row-dot RMSE改善だけでは、後段layerの最終max absが悪化する例が出ていた。

## 今回の変更点

- 複数fixtureのprefix smoke結果を集約する `tools/summarize-qwen-prefix-smokes.py` を追加した。
- `qwen-layer-module-trace` の `row_dot.<projection>.scale_fit` からrow-scale候補を抽出する `tools/extract-qwen-row-scale-candidates.py` を追加した。
- 候補ごとの受入ゲートを機械判定する `tools/evaluate-qwen-prefix-candidate-gates.py` を追加した。
- OOMを避けるため、fixture/condition matrixを1 smokeずつ逐次実行する `tools/run-qwen-prefix-smoke-matrix.py` を追加した。
- `docs/plans/multi-fixture-row-scale-validation-plan-v0.1.md` に進捗と判定結果を追記した。

## 現在の判断

この問題は引き続きデバッグする価値がある。理由は、残差がbackendノイズではなく、特定のlayer/token/hiddenとrow-scale候補に局所化できているため。

ただし、今の扱いは「row-scale候補をすぐ昇格する」ではない。複数fixtureのend-to-end smokeで悪化しない候補だけを受け入れる。

## Gate Result

Input summary:

- `benchmarks/results/2026-07-05/engine/qwen-prefix-smoke-multi-fixture-summary.json`
- `benchmarks/results/2026-07-05/engine/qwen-prefix-candidate-gates.md`
- `benchmarks/results/2026-07-05/engine/qwen-prefix-extracted-candidate-gates.md`
- `benchmarks/results/2026-07-05/engine/qwen-prefix-layer6-attn-mlp-gates.md`
- `benchmarks/results/2026-07-05/engine/qwen-prefix-layer6-mlp-selected-gates.md`
- `benchmarks/results/2026-07-05/engine/qwen-prefix-layer6-mlp-grid5-gates.md`
- `benchmarks/results/2026-07-05/engine/qwen-prefix-manifest-vs-no-row-scale-five-fixture-gates.md`
- `benchmarks/results/2026-07-05/engine/qwen-prefix-manifest-vs-p4p65-inproj-five-fixture-gates.md`
- `benchmarks/results/2026-07-05/engine/qwen-prefix-manifest-vs-p4p65-row3456-five-fixture-gates.md`
- `benchmarks/results/2026-07-05/engine/qwen-prefix-manifest-vs-layer8-manifest-packages-five-fixture-gates.md`

Gate settings:

- fixtureごとの最終max abs悪化が `0.001` を超えたらreject
- median improvementは `0.005` 以上を優先
- paired fixtureは最低 `3`

| condition | decision | fixtures | median improvement | max regression | reason |
| --- | --- | ---: | ---: | ---: | --- |
| `layer6` | reject | 3 | 0.0081653595 | 0.00455665588 | tokens201 regression exceeds `0.001` |
| `layer6-mlp-selected` | reject | 3 | 0.00720596313 | 0.00403785706 | tokens201 regression exceeds `0.001` |
| `layer6-attn-mlp` | reject | 3 | 0.00578689575 | 0.0117874146 | tokens201 regression exceeds `0.001` |
| `combined` | needs_more_fixtures | 1 | 0.0343608856 | 0 | only one paired fixture |
| `extracted` | reject | 3 | -0.0312900543 | 0.0727806091 | tokens1 and tokens201 regress |

Five-fixture layer6 MLP grid:

| condition | decision | fixtures | median improvement | max regression | reason |
| --- | --- | ---: | ---: | ---: | --- |
| `layer6-mlp-h3994-s1p004` | reject | 5 | 0.000791549683 | 0.00535869598 | tokens401 regression exceeds `0.001` |
| `layer6-mlp-h3994-s1p008` | reject | 5 | 0.00157546997 | 0.0107059479 | tokens201 and tokens401 regress |
| `layer6-mlp-h3994-s1p026471714` | reject | 5 | 0.00524330139 | 0.0354146957 | tokens401 regression dominates |

Manifest baseline vs no-row-scale:

| condition | decision | fixtures | median improvement | max regression | reason |
| --- | --- | ---: | ---: | ---: | --- |
| `no-row-scale` | reject | 5 | -0.494504929 | 1.13697243 | row3456 manifest compensation is still needed |

Quantization-policy branch:

| condition | decision | fixtures | median improvement | max regression | reason |
| --- | --- | ---: | ---: | ---: | --- |
| `p4p65-inproj` | reject | 5 | -0.797094345 | 1.39915657 | row3456 manifest compensation is missing |
| `p4p65-row3456` | reject | 5 | -0.0370130539 | 0.26203537 | row3456 improves, but hidden3994 regresses |

Layer8 manifest package branch:

| condition | decision | fixtures | median improvement | max regression | reason |
| --- | --- | ---: | ---: | ---: | --- |
| `layer8-gateup` | reject | 5 | 0 | 0.0102748871 | tokens1 regression exceeds `0.001` |
| `layer8-gateupfit` | reject | 5 | 0 | 0.00478172302 | tokens1 regression exceeds `0.001` |
| `layer8-up` | reject | 5 | 0 | 0.00839996338 | tokens1 regression exceeds `0.001` |
| `layer8-upfit` | reject | 5 | 0 | 0.00354194641 | tokens1 regression exceeds `0.001` |

Interpretation:

- Layer6 hidden3994 row-scale is a real local compensation candidate.
- It should not be promoted unconditionally under the initial multi-fixture gate.
- Selecting the lower tokens101-fitted MLP scale reduces the tokens201 regression slightly, but not enough to pass the `0.001` gate.
- Adding layer6 attention row-scale to layer6 MLP row-scale improves tokens1/tokens101 but worsens tokens201 more than MLP-only.
- Layer8 QKV V845 cell remains smoke-only because current paired fixture coverage is insufficient.
- The automatically extracted candidate set is worse than layer6-only, so candidate extraction must feed a gated search loop rather than direct promotion.
- Adding two new fixtures changes the rejection reason: tokens401 is a stronger counterexample than tokens201 for layer6 MLP hidden3994 scaling.
- The current manifest row-scale entries for row3456 are still useful. Removing them worsens four of five fixtures, so the next path is not to remove manifest compensation wholesale.
- `p4p65-inproj` alone is not a fair replacement because it lacks row3456 compensation and regresses row3456-heavy fixtures.
- `p4p65` plus the same row3456 smoke overrides still fails, mainly because hidden3994 worsens on tokens401.
- Existing layer8 gate/up manifest packages are close but still fail: they improve tokens201 slightly and keep tokens401 inside the hard gate, but regress tokens1.

## Key Evidence

| fixture | baseline | layer6 h3994 row-scale | result |
| --- | ---: | ---: | --- |
| token ids `1..16` | 0.645338058 | 0.637172699 | improves |
| token ids `101..116` | 1.080525398 | 1.043153763 | improves |
| token ids `201..216` | 1.140727997 | 1.145284653 | worsens final max |

Layer6 row-scale still improves early layers on tokens201:

- layer6: `0.537414551 -> 0.476898193`
- layer7: `0.966460228 -> 0.497438431`
- layer11 final max: `1.140727997 -> 1.145284653`

Layer6 `mlp.down_proj.weight[3994]` row-dot scale is stable across the two available v0.10 traces:

- token ids `1..16`: scale `1.026471714`, RMSE `0.117735388 -> 0.063680278`
- token ids `101..116`: scale `1.023383096`, RMSE `0.131756300 -> 0.061972585`

The lower tokens101-selected MLP-only scale was tested:

- token ids `1..16`: `0.645338058 -> 0.638132095`
- token ids `101..116`: `1.080525398 -> 1.047520638`
- token ids `201..216`: `1.140727997 -> 1.144765854`

The extracted three-row candidate set was tested across all three fixtures:

- token ids `1..16`: `0.645338058 -> 0.676628113`
- token ids `101..116`: `1.080525398 -> 1.039905548`
- token ids `201..216`: `1.140727997 -> 1.213508606`

The layer6 attention+MLP candidate set was also tested:

- token ids `1..16`: `0.645338058 -> 0.639551163`
- token ids `101..116`: `1.080525398 -> 1.039905548`
- token ids `201..216`: `1.140727997 -> 1.152515411`

This means the local correction is real, but it changes later propagation in a way that the current policy cannot accept.

Five-fixture layer6 MLP grid:

- token ids `1..16`: scale `1.026471714` improves `0.645338058 -> 0.637172699`
- token ids `101..116`: scale `1.026471714` improves `1.080525398 -> 1.043153763`
- token ids `201..216`: scale `1.004` stays within the old hard gate, but larger scales regress; scale `1.026471714` worsens `1.140727997 -> 1.145284653`
- token ids `301..316`: scale `1.026471714` improves `1.371309280 -> 1.366065979`
- token ids `401..416`: every positive layer6 MLP scale worsens; scale `1.004` worsens `0.959306717 -> 0.964665413`, scale `1.026471714` worsens `0.959306717 -> 0.994721413`

Tokens401 localization:

- Baseline worst coordinate is layer8 token9 hidden `3994`: `actual=15.915693283`, `expected=16.875`, diff `-0.959306717`.
- Layer8 input for the same token/hidden is already low: `15.160280228` vs `15.625`, diff `-0.464719772`.
- Full-reference layer8 with the package actual input outputs `15.875`, exactly `-1.0` from the golden fixture output.
- Package layer8 with the same actual input is slightly less bad than full-reference actual-input replay: package delta error vs full-reference actual-input is `+0.0406933`.
- Layer7 token10 hidden `3994` shows the same pattern: full-reference actual-input replay is already `-0.875`, while package-vs-fullref delta error is only `-0.0277519`.
- Interpretation: tokens401 is mainly an input-drift amplification case, not a layer8 row-quantization-only case.

Quantization-policy probe:

- `p4p65-inproj` without row3456 compensation regresses:
  - tokens1: `0.645338058 -> 1.78714752`
  - tokens301: `1.371309280 -> 2.77046585`
  - tokens401: `0.959306717 -> 1.75640106`
- Adding the existing row3456 smoke overrides to p4p65 reduces the row3456 failures but does not pass:
  - tokens1: `0.645338058 -> 0.790296555`
  - tokens101: `1.080525398 -> 1.11753845`
  - tokens201: `1.140727997 -> 1.13527489`
  - tokens301: `1.371309280 -> 1.37655067`
  - tokens401: `0.959306717 -> 1.22134209`
- Interpretation: the next package-level candidate needs to preserve row3456 compensation while targeting hidden3994 input-drift amplification more directly than `p4p65`.

Layer8 manifest package probe:

- The best of the tested layer8 manifest packages is `layer8-upfit`.
- It improves tokens201 `1.140727997 -> 1.13804817` and tokens301 `1.371309280 -> 1.37123108`.
- It keeps tokens401 inside the hard gate: `0.959306717 -> 0.959452629`, delta `+0.00014591217`.
- It fails because tokens1 regresses `0.645338058 -> 0.648880005`, delta `+0.00354194641`.
- Attempting to tune layer8 `mlp.up_proj.weight[6340]` through smoke-only row-scale is not supported by `package-golden-prefix-smoke`; that path requires manifest package generation or engine support for gate/up smoke overrides.

Weak layer8-upfit manifest grid:

- Added `tools/build-qwen-row-scale-manifest-package.py` and `tools/generate-qwen-manifest-row-scale-grid.py`.
- Important package-builder fix: hardlink package copies must unlink destination `manifest.json` before writing, otherwise the source package manifest is mutated through the hardlink.
- Generated weak layer8 `mlp.up_proj.weight[6340]` manifest packages on top of the existing row3456 compensation.
- tokens1 prefilter:
  - scale `1.004`: `0.645338058 -> 0.645746231`, hard gate内
  - scale `1.008`: `0.645338058 -> 0.646137238`, hard gate内
  - scale `1.012`: `0.645338058 -> 0.646537781`, hard gate超え
- Full five-fixture gate:

| condition | decision | fixtures | median improvement | max regression | reason |
| --- | --- | ---: | ---: | ---: | --- |
| `layer8-up6340-s1p004` | hold | 5 | `0` | `0.000408172607` | aggregate effect too small |
| `layer8-up6340-s1p008` | hold | 5 | `0` | `0.000799179077` | aggregate effect too small |

Key interpretation:

- The weaker scale avoids the large tokens1 regression from full `layer8-upfit`.
- However, it only improves tokens201 slightly:
  - `1.004`: `1.140727997 -> 1.140417099`
  - `1.008`: `1.140727997 -> 1.140106201`
- It does not fix tokens401:
  - `1.004`: `0.959306717 -> 0.959323883`
  - `1.008`: `0.959306717 -> 0.959340096`
- The chain comparison confirms this: `layer8-up6340-s1p008` changes tokens401 layer8 hidden3994 by only `+0.000033378601`, in the wrong direction.

Tokens401-derived layer8 local candidate:

- Existing tokens401 layer8 row-scale candidates were tested as a prefilter:
  - layer8 `linear_attn.out_proj.weight[3994]` scale `0.9727276122005596`
  - layer8 `mlp.down_proj.weight[3994]` scale `1.012138640209781`
- This candidate was rejected before full five-fixture completion because tokens1 regressed:
  - baseline: `0.645338058`
  - candidate: `0.662992477`
  - delta: `+0.017654419`
- This reinforces that layer8 local row-scale can overfit tokens401 and conflict with tokens1.

Quantizer policy branch dry-run:

- Baseline `p4p46_inproj`:
  - high tensors: `114`
  - low tensors: `141`
  - estimated output bytes: `9121922016`
- Custom `p4p46 + mlp_up`:
  - high tensors: `147`
  - low tensors: `108`
  - estimated output bytes: `9225731040`
  - estimated output increase: `103809024` bytes
- Interpretation:
  - family-wide `mlp_up` high assignment is cheap enough to build if needed.
  - It is still broad and not activation-aware, so it may repeat the p4p65 pattern of cross-fixture regressions.
  - The evidence now favors a narrower tensor override policy or activation-aware row policy over another broad family-wide package.

Per-tensor high-format override:

- Added repeatable `--aq-high-tensor <TENSOR_NAME>` to `ullm-quant`.
- Dry-run `p4p46_inproj + model.language_model.layers.8.mlp.up_proj.weight`:
  - high tensors: `115`
  - low tensors: `140`
  - estimated output bytes: `9125067744`
  - estimated output increase over baseline: `3145728` bytes
- This confirms the next experiment can promote the target layer8 tensor without raising every `mlp_up` tensor.

## Verification

- `python3 -m py_compile tools/summarize-qwen-prefix-smokes.py`
- `python3 -m py_compile tools/extract-qwen-row-scale-candidates.py`
- `python3 -m py_compile tools/evaluate-qwen-prefix-candidate-gates.py`
- `python3 -m py_compile tools/run-qwen-prefix-smoke-matrix.py`
- JSON parse checks for generated summary/candidate/gate artifacts.
- JSON parse checks for five-fixture grid, no-row-scale comparison, coordinate-chain, and module-trace comparison artifacts.
- JSON parse checks for p4p65 and p4p65+row3456 five-fixture summary/gate artifacts.
- JSON parse checks for layer8 manifest package five-fixture summary/gate artifacts.
- JSON parse checks for weak layer8-up6340 manifest grid, selected five-fixture summary/gate artifacts, chain comparison, and tokens401 layer8 local prefilter artifacts.
- JSON parse checks for quantizer policy dry-run plan artifacts.
- `cargo fmt --all --check`, `cargo check -p ullm-quant`, and `cargo test -p ullm-quant -- --test-threads=1` for the per-tensor high-format override.
- `tools/run-qwen-prefix-smoke-matrix.py` dry-run with tokens1/tokens101 and baseline/layer6 conditions.
- `tools/run-qwen-prefix-smoke-matrix.py` real run for the extracted three-row candidate set across tokens1/tokens101/tokens201.
- `tools/run-qwen-prefix-smoke-matrix.py` real run for layer6 attention+MLP across tokens1/tokens101/tokens201.
- `tools/run-qwen-prefix-smoke-matrix.py` real run for selected layer6 MLP-only scale across tokens1/tokens101/tokens201.
- Regenerated tokens1 layer6 hidden3994 trace with `qwen-layer-module-trace-v0.10`.

## Next Action

1. Treat layer6 hidden3994 MLP row-scale as rejected for unconditional promotion under the five-fixture gate.
2. Do not promote `p4p65-inproj` or `p4p65+row3456`; both fail five-fixture gates.
3. Do not promote weak `layer8-up6340`; it is hard-gate safe at low scale but aggregate effect is too small and tokens401 does not improve.
4. Do not continue direct tokens401 layer8 local row-scale as a general fix; it strongly regresses tokens1.
5. Prefer a narrower tensor override policy or activation-aware / row-aware quantizer policy over broad family-wide row-scale work.
6. Keep row-dot extraction as a proposal mechanism only; full-prefix multi-fixture gate remains authoritative.

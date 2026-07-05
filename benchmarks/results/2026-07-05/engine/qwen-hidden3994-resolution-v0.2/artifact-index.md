# Qwen Hidden3994 Resolution v0.2 Artifact Index

## Package Builder

- `tools/build-qwen-row-scale-manifest-package.py`
- `tools/generate-qwen-manifest-row-scale-grid.py`
- `qwen-hidden3994-resolution-v0.2/rebuild-layer8upfit.json`
- `qwen-hidden3994-resolution-v0.2/rebuild-layer8upfit.md`

Validation:

- The rebuilt layer8-upfit package row-scale manifest matches the existing layer8-upfit package.
- The source baseline package manifest was restored and verified with `4` row-scale entries.
- The builder now unlinks destination `manifest.json` before writing, so hardlink package copies do not mutate the source package manifest.

## Weak Layer8 Up6340 Grid

- `qwen-manifest-row-scale-grid-layer8-up6340-weak/summary.json`
- `qwen-manifest-row-scale-grid-layer8-up6340-weak/summary.md`
- `qwen-manifest-row-scale-grid-layer8-up6340-weak/conditions.txt`
- `qwen-manifest-row-scale-grid-layer8-up6340-weak/conditions-selected.txt`

Selected full five-fixture matrix:

- `qwen-prefix-smoke-matrix-layer8-up6340-weak-selected-five-fixture/summary.json`
- `qwen-prefix-smoke-matrix-layer8-up6340-weak-selected-five-fixture/summary.md`
- `qwen-prefix-layer8-up6340-weak-selected-summary.json`
- `qwen-prefix-layer8-up6340-weak-selected-summary.md`
- `qwen-prefix-layer8-up6340-weak-selected-gates.json`
- `qwen-prefix-layer8-up6340-weak-selected-gates.md`

Gate result:

| condition | decision | median improvement | max regression | note |
| --- | --- | ---: | ---: | --- |
| `layer8-up6340-s1p004` | hold | `0` | `0.000408172607` | hard gate safe, aggregate effect too small |
| `layer8-up6340-s1p008` | hold | `0` | `0.000799179077` | hard gate safe, aggregate effect too small |

## Chain Comparison

- `package-golden-prefix-coordinate-chain-tokens1-baseline-token7-hidden3994-v0.2.{json,md}`
- `package-golden-prefix-coordinate-chain-tokens1-layer8-up6340-s1p008-token7-hidden3994-v0.2.{json,md}`
- `package-golden-prefix-coordinate-chain-tokens401-baseline-token9-hidden3994-v0.2.{json,md}`
- `package-golden-prefix-coordinate-chain-tokens401-layer8-up6340-s1p008-token9-hidden3994-v0.2.{json,md}`
- `qwen-hidden3994-upstream-drift-comparison-v0.2.{json,md}`

Finding:

- `layer8-up6340-s1p008` only changes tokens401 layer8 hidden3994 by `+0.000033378601` in the wrong direction.
- The candidate slightly improves tokens201 but does not address tokens401 input-drift amplification.

## Tokens401 Layer8 Local Prefilter

- `qwen-row-scale-candidates-tokens401-layer8-hidden3994-smoke.json`
- `qwen-prefix-smoke-matrix-tokens401-layer8-local-five-fixture/`
- `qwen-prefix-tokens401-layer8-local-prefilter.{json,md}`

Decision:

- Reject before full five-fixture completion.
- tokens1 worsened from `0.645338058` to `0.662992477`, delta `+0.017654419`.

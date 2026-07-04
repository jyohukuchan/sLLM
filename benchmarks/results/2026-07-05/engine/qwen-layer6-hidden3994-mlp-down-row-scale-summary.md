# Qwen layer 6 hidden3994 MLP down row-scale experiment

Artifacts:

- `package-row-scale-overrides-layer6-hidden3994-mlp-down-p4p46-inproj.json`
- `package-golden-prefix-cpu-actual-prefix0-12-rotary64-manifest-row-scale-layer6-layer10-cli-row-scale-layer6-hidden3994-mlp-down-p4p46-inproj.jsonl`
- `package-golden-prefix-cpu-actual-prefix0-12-rotary64-manifest-row-scale-layer6-layer10-cli-row-scale-layer6-hidden3994-mlp-down-cell-delta-layer8qkv-v845col3994-p4p46-inproj.jsonl`
- `qwen-layer-module-trace-actual-input-rotary64-layer6-token0-hidden3994-trace-layer6-layer10-p4p46-inproj.jsonl`
- `qwen-module-trace-comparison-actual-input-rotary64-layer6-token0-hidden3994-layer6-layer10-p4p46-inproj.json`

## Layer 6 Localization

Layer `7`, token `0`, hidden `3994` is mostly inherited from layer `6`:

- layer `7` input diff: `-0.480636597`
- layer `7` local delta diff: `-0.147010803`
- layer `7` output diff: `-0.627647400`

Layer `6`, token `0`, hidden `3994` comparison:

| component | value |
| --- | ---: |
| local actual delta error | -0.730636597 |
| attention row-only error | 0.021879351 |
| attention activation-path error | -0.096949037 |
| MLP row-only error | -0.376916865 |
| MLP activation-path error | -0.269932446 |

The strongest direct row target is `mlp.down_proj.weight[3994]`:

| item | token 0 value |
| --- | ---: |
| fullref module output | 14.625000000 |
| source row dot | 14.607685722 |
| package row dot | 14.230768857 |
| package-source row-dot error | -0.376916865 |

All-token least-squares scale for the package down row:

| tensor | row | scale |
| --- | ---: | ---: |
| `mlp.down_proj.weight` | 3994 | 1.02647171355 |

## Full Prefix Results

All runs use:

- package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer10.ullm.d`
- fixture: `benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16`
- run mode: `actual_prefix`
- layers: `0..12`
- `rotary_dim=64`
- CPU backend

| variant | overall max_abs | max layer | max token/hidden |
| --- | ---: | ---: | --- |
| baseline layer6/layer10 row-scale | 0.645338058 | 11 | token 7 / hidden 3994 |
| layer6 hidden3994 MLP down row-scale | 0.637172699 | 11 | token 7 / hidden 3994 |
| layer8 qkv V845 cell source-restore | 0.627647400 | 7 | token 0 / hidden 3994 |
| combined layer6 row-scale + layer8 qkv cell | 0.610977173 | 11 | token 7 / hidden 3994 |

Layer detail:

| variant | layer 6 | layer 7 | layer 8 | layer 11 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.480636597 | 0.627647400 | 0.578010559 | 0.645338058 |
| layer6 row-scale | 0.465695381 | 0.428003311 | 0.565040588 | 0.637172699 |
| layer8 qkv cell | 0.480636597 | 0.627647400 | 0.588329315 | 0.619235992 |
| combined | 0.465695381 | 0.428003311 | 0.575433731 | 0.610977173 |

## Prefix0-8 Seq16 Partial Recheck

This recheck uses the shorter existing fixture:

- fixture: `benchmarks/golden/2026-07-05/qwen35-9b-prefix0-8-seq16`
- token ids: `1..16`
- layers: `0..8`
- run mode: `actual_prefix`
- `rotary_dim=64`
- CPU backend

Artifacts:

- baseline report: `package-golden-prefix-cpu-actual-prefix0-8-seq16-rotary64-manifest-row-scale-layer6-layer10-p4p46-inproj.jsonl`
- layer6 row-scale report: `package-golden-prefix-cpu-actual-prefix0-8-seq16-rotary64-layer6h3994-row-scale-p4p46-inproj.jsonl`

| variant | overall max_abs | max layer | max token/hidden |
| --- | ---: | ---: | --- |
| baseline layer6/layer10 manifest row-scale | 0.627647400 | 7 | token 0 / hidden 3994 |
| layer6 hidden3994 MLP down row-scale | 0.542758942 | 4 | token 14 / hidden 3994 |

Layer detail:

| variant | layer 6 | layer 7 |
| --- | ---: | ---: |
| baseline | 0.480636597 | 0.627647400 |
| layer6 row-scale | 0.465695381 | 0.428003311 |

## Tokens101-116 Cross-Fixture Recheck

This recheck uses a newly exported fixture with different token ids:

- fixture: `benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16-tokens101-116`
- token ids: `101..116`
- layers: `0..12`
- run mode: `actual_prefix`
- `rotary_dim=64`
- CPU backend
- summary: `qwen-cross-fixture-tokens101-116-summary.md`

| variant | overall max_abs | max layer | max token/hidden | layer11 max_abs |
| --- | ---: | ---: | --- | ---: |
| baseline | 1.080525398 | 7 | token 12 / hidden 3994 | 0.946708679 |
| layer6 row-scale | 1.043153763 | 7 | token 12 / hidden 3994 | 0.916080475 |
| layer8 QKV V845 cell | 1.080525398 | 7 | token 12 / hidden 3994 | 0.969970703 |
| combined | 1.043153763 | 7 | token 12 / hidden 3994 | 0.939382553 |

## Manifest Metadata Prototype

A hardlink package copy was created outside the repository:

- package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6h3994-layer6-layer10.ullm.d`
- source package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer10.ullm.d`
- added manifest row-scale entry:
  - tensor: `model.language_model.layers.6.mlp.down_proj.weight`
  - row: `3994`
  - scale: `1.02647171355`
  - source: `golden-prefix-row-dot-sensitivity-layer6-hidden3994`

Manifest-only reports:

- manifest row-scale JSON:
  - `package-row-scale-overrides-layer6h3994-layer6-layer10-p4p46-inproj.json`
  - schema: `row-scale-overrides-v0.1`
  - entries: existing layer6/10 hidden3456 entries plus layer6 `mlp.down_proj.weight[3994]`
- layer6 row-scale only:
  - `package-golden-prefix-cpu-actual-prefix0-12-rotary64-manifest-row-scale-layer6h3994-layer6-layer10-p4p46-inproj.jsonl`
  - max abs: `0.637172699`
  - matches the CLI row-scale report exactly for layers `6`, `7`, `8`, and `11`
- manifest layer6 row-scale + layer8 QKV V845 cell:
  - `package-golden-prefix-cpu-actual-prefix0-12-rotary64-manifest-row-scale-layer6h3994-layer6-layer10-cell-delta-layer8qkv-v845col3994-p4p46-inproj.jsonl`
  - max abs: `0.610977173`
  - matches the current best CLI row-scale + QKV cell result

Validation:

- JSON duplicate/positive-scale check: passed, `5` entries.
- `cargo test -p ullm-engine row_scale -- --nocapture`: passed.
- `cargo test -p ullm-quant row_scale -- --nocapture`: passed with `0` matching tests.

## Interpretation

- The layer `6` row-scale directly reduces the inherited layer `7` floor from
  `0.627647400` to `0.428003311`.
- The `prefix0-8-seq16` partial recheck repeats the same layer `7` reduction,
  moving the fixture max from `0.627647400` to `0.542758942`.
- The `tokens101-116` cross-fixture recheck also improves with layer6 row-scale,
  moving the fixture max from `1.080525398` to `1.043153763`.
- The layer `8` qkv V845 cell correction mainly reduces the later layer `11`
  hidden `3994` chain.
- On the `tokens101-116` fixture, the layer `8` qkv V845 cell does not improve
  the overall max and worsens layer `11` versus layer6 row-scale alone.
- The layer6 hidden3994 row-scale can be represented as package manifest
  metadata, not only as a smoke-only CLI override.
- The two interventions are complementary: combined max improves to
  `0.610977173`, the best result in this batch.
- This is still smoke-only. A durable fix should be expressed as quantizer or
  package metadata policy after checking whether the layer `6` row-scale and
  layer `8` qkv V-row correction generalize beyond this fixture.

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

## Interpretation

- The layer `6` row-scale directly reduces the inherited layer `7` floor from
  `0.627647400` to `0.428003311`.
- The layer `8` qkv V845 cell correction mainly reduces the later layer `11`
  hidden `3994` chain.
- The two interventions are complementary: combined max improves to
  `0.610977173`, the best result in this batch.
- This is still smoke-only. A durable fix should be expressed as quantizer or
  package metadata policy after checking whether the layer `6` row-scale and
  layer `8` qkv V-row correction generalize beyond this fixture.

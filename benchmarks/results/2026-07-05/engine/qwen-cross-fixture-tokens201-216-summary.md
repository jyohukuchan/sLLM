# Qwen Cross-Fixture Tokens201-216 Summary

## Fixture

- Fixture: `benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16-tokens201-216`
- Token ids: `201..216`
- Package baseline: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer10.ullm.d`
- Package with layer6 hidden3994 manifest row-scale: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6h3994-layer6-layer10.ullm.d`
- Run mode: `actual_prefix`
- Layers: `0..12`
- Rotary: `rotary_dim=64`, `rope_base=10000000`, `position_offset=0`
- Backend: CPU

## Reports

- Baseline:
  - `package-golden-prefix-cpu-actual-prefix0-12-seq16-tokens201-216-rotary64-manifest-row-scale-layer6-layer10-p4p46-inproj.jsonl`
- Layer6 hidden3994 manifest row-scale:
  - `package-golden-prefix-cpu-actual-prefix0-12-seq16-tokens201-216-rotary64-manifest-row-scale-layer6h3994-layer6-layer10-p4p46-inproj.jsonl`

## Overall Results

| variant | overall max_abs | max layer | max token/hidden | layer6 | layer7 | layer11 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| baseline | 1.140727997 | 11 | token 13 / hidden 3994 | 0.537414551 | 0.966460228 | 1.140727997 |
| layer6 hidden3994 row-scale | 1.145284653 | 11 | token 13 / hidden 3994 | 0.476898193 | 0.497438431 | 1.145284653 |

## Interpretation

- The layer6 hidden3994 row-scale strongly reduces the early inherited hidden3994 floor:
  - layer6 max improves from `0.537414551` to `0.476898193`
  - layer7 max improves from `0.966460228` to `0.497438431`
- The final objective does not improve on this fixture:
  - overall max worsens slightly from `1.140727997` to `1.145284653`
  - layer11 remains the overall max and shifts only slightly worse.
- This weakens a simple unconditional promotion decision.
- The better interpretation is:
  - the layer6 row-scale is a real local compensation candidate
  - but a durable package policy should be validated on a multi-fixture objective, not promoted solely from one or two improving fixtures.

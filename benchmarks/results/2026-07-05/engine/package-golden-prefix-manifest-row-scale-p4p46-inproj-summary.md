# Package Golden Prefix Manifest Row-Scale Summary

Date: 2026-07-05

Package:

- Source: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-reservoir65536-jobs4.ullm.d`
- Manifest metadata copy: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer10.ullm.d`

Metadata entries:

- `model.language_model.layers.10.linear_attn.out_proj.weight`, row `3456`, scale `1.02307179310`
- `model.language_model.layers.10.mlp.down_proj.weight`, row `3456`, scale `1.04165701172`

Reports:

- CPU no metadata, `golden_before_each_layer`: `package-golden-prefix-cpu-golden-before0-12-current-manifest-compare-no-metadata-p4p46-inproj.jsonl`
- CPU no metadata, `actual_prefix`: `package-golden-prefix-cpu-actual-prefix0-12-current-manifest-compare-no-metadata-p4p46-inproj.jsonl`
- CPU manifest metadata, `golden_before_each_layer`: `package-golden-prefix-cpu-golden-before0-12-manifest-row-scale-p4p46-inproj.jsonl`
- CPU manifest metadata, `actual_prefix`: `package-golden-prefix-cpu-actual-prefix0-12-manifest-row-scale-p4p46-inproj.jsonl`
- R9700 manifest metadata, `golden_before_each_layer`: `package-golden-prefix-r9700-golden-before0-12-manifest-row-scale-p4p46-inproj.jsonl`
- R9700 manifest metadata, `actual_prefix`: `package-golden-prefix-r9700-actual-prefix0-12-manifest-row-scale-p4p46-inproj.jsonl`

CPU current-binary comparison:

| mode | max MSE before | max MSE after | max abs before | max abs after |
| --- | ---: | ---: | ---: | ---: |
| `golden_before_each_layer` | `0.000740506879` | `0.000740506879` | `0.875896454` | `0.508314133` |
| `actual_prefix` | `0.004141662294` | `0.004106469453` | `1.744266510` | `0.967845917` |

Layer `10` direct max-abs effect:

| mode | before | after |
| --- | ---: | ---: |
| `golden_before_each_layer` | `0.875896454` | `0.304975510` |
| `actual_prefix` | `1.744266510` | `0.967845917` |

R9700 manifest metadata:

| mode | max MSE | max mean abs diff | max abs diff | min cosine similarity |
| --- | ---: | ---: | ---: | ---: |
| `golden_before_each_layer` | `0.000740507114` | `0.020715803` | `0.508314133` | `0.998585695` |
| `actual_prefix` | `0.004106476000` | `0.050080222` | `0.967796326` | `0.992982658` |

Conclusion:

Manifest row-scale metadata preserves the layer `10` max-abs improvement without passing the smoke-only row-scale JSON argument. The effect is stable across CPU and R9700. Later-layer aggregate MSE remains a separate issue.

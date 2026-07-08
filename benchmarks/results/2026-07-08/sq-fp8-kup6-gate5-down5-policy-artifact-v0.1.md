# SQ FP8 `kup6_gate5_down5` Policy Artifact v0.1

Date: 2026-07-08

## Summary

The `kup6_gate5_down5` policy was used to generate a real SQ FP8 artifact with payload files.

This verifies the path:

```text
sq-fp8-policy-v0.1 -> build-sq-fp8-w8a16-artifact.py --policy-json -> sq_manifest.json -> runtime materialize smoke
```

The FP8 payload artifact lives under `/tmp` and is not committed to git.

## Artifact

| field | value |
| --- | --- |
| artifact dir | `/tmp/ullm-sq-fp8-kup6-gate5-down5-policy-v0.1-artifact` |
| manifest | `/tmp/ullm-sq-fp8-kup6-gate5-down5-policy-v0.1-artifact/sq_manifest.json` |
| candidate | `sq-fp8-w8a16-r9700-v0` |
| policy | `kup6_gate5_down5` |
| scale | `row_block32`, `f32` |
| FP8 tensors | `22` |
| passthrough tensors | `753` |
| artifact disk usage | `892 MiB` |
| artifact file count | `45` |

Storage fields from manifest:

| field | bytes |
| --- | ---: |
| FP8 payload | `830472192` |
| FP8 scales | `103809024` |
| passthrough source estimate | `17645272032` |
| compact resident estimate | `18579553248` |
| materialized working-set estimate | `201326592` |

The compact resident estimate is conservative because it includes passthrough source bytes.

## Runtime Materialize Smoke

Command:

```text
target/debug/ullm-engine sq-fp8-materialize-smoke \
  /tmp/ullm-sq-fp8-kup6-gate5-down5-policy-v0.1-artifact \
  2 \
  model.language_model.layers.3.self_attn.k_proj.weight \
  2 \
  0
```

Result:

| field | value |
| --- | ---: |
| device index | `2` |
| tensor index | `17` |
| tensor | `model.language_model.layers.3.self_attn.k_proj.weight` |
| shape | `[1024,4096]` |
| rows materialized | `2` |
| materialized elements | `8192` |
| output bytes | `32768` |
| roundtrip max abs diff | `0.000000000` |
| verified | `true` |

## Interpretation

This closes the T2 gap between the saved quality policy and an actual runtime-consumable artifact.
It is still not a throughput result and does not promote `kup6_gate5_down5` to the final SQ policy.

## Next Action

1. Use this artifact path for further runtime boundary checks when needed.
2. Do not use host-side materialize/load timing as SQ throughput.
3. Continue T1 full package real-batch work before AQ4/FP8 throughput comparison.

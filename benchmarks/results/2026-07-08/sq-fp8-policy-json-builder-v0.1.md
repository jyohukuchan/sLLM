# SQ FP8 Policy JSON Builder v0.1

Date: 2026-07-08

## Summary

`tools/build-sq-fp8-w8a16-artifact.py` can now consume `sq-fp8-policy-v0.1` directly.

This removes the need to manually copy the `kup6_gate5_down5` include regex when generating the
current six-layer regression subset artifact.

## Policy

- Policy JSON: `benchmarks/results/2026-07-08/sq-fp8-kup6-gate5-down5-policy-v0.1.json`
- Policy ID: `kup6_gate5_down5`
- Candidate: `sq-fp8-w8a16-r9700-v0`
- Scale: row-block32, F32 scale

## Dry-Run Verification

Command:

```text
python3 tools/build-sq-fp8-w8a16-artifact.py \
  --source-model-dir /home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B \
  --output-artifact /tmp/ullm-sq-fp8-policy-json-dry-run \
  --base-package /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d \
  --policy-json benchmarks/results/2026-07-08/sq-fp8-kup6-gate5-down5-policy-v0.1.json \
  --metadata-only \
  --dry-run
```

Observed dry-run fields:

| field | value |
| --- | ---: |
| policy ID | `kup6_gate5_down5` |
| FP8 tensor count | `22` |
| passthrough tensor count | `753` |
| scale granularity | `row_block` |
| scale block cols | `32` |

## Tests

```text
python3 -m unittest \
  tests.test_build_sq_fp8_artifact_policy \
  tests.test_sq_fp8_overlay_acceptance \
  tests.test_external_benchmark_batch_parser \
  tests.test_compare_package_guards \
  tests.test_sq_candidate_runtime_row
```

Result: `12` tests passed.

## Next Action

1. Use `--policy-json` for the next SQ FP8 artifact generation based on `kup6_gate5_down5`.
2. Keep the generated manifest `policy` block as the bridge from the quality boundary to the runtime artifact.
3. Continue T1 real batch runner work before using throughput rows for SQ performance decisions.

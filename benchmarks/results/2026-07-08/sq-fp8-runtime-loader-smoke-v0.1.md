# SQ FP8 Runtime Loader Smoke v0.1

## Summary

This records the first runtime-side SQ FP8 artifact load smoke.

- Candidate: `sq-fp8-w8a16-r9700-v0`
- Artifact fixture: `/tmp/ullm-sq-fp8-runtime-fixture-out`
- Tensor: `model.language_model.layers.0.mlp.gate_proj.weight`
- Shape: `4 x 8`
- Payload: FP8 E4M3, `32` bytes
- Scale: F32 row scale, `16` bytes

The smoke validates:

1. `sq_manifest.json` can be read and validated by `ullm-engine`.
2. A tensor can be selected by unique substring.
3. Selected rows can be materialized from FP8 E4M3 + F32 row scale to F32.
4. The materialized F32 rows can be copied to a runtime buffer and read back exactly.

## Commands

```bash
python3 tools/build-sq-fp8-w8a16-artifact.py \
  --source-model-dir /tmp/ullm-sq-fp8-runtime-fixture \
  --output-artifact /tmp/ullm-sq-fp8-runtime-fixture-out \
  --overwrite

./target/debug/ullm-engine sq-fp8-materialize-smoke \
  /tmp/ullm-sq-fp8-runtime-fixture-out 0 gate_proj 2 0

./target/debug/ullm-engine sq-fp8-materialize-smoke \
  /tmp/ullm-sq-fp8-runtime-fixture-out 2 gate_proj 2 1
```

## Results

| device | backend | rows | preview | roundtrip max abs diff | verified |
| --- | --- | ---: | --- | ---: | --- |
| `0` | CPU | `0..2` | `[0,1,2,3,4,5,6,7]` | `0.0` | true |
| `2` | R9700/RDNA4 | `1..3` | `[8.0357141,8.5714283,9.6428566,10.7142849,11.7857141,12.8571424,13.9285707,15.0]` | `0.0` | true |

The R9700 row values are not exact integers because the source values are encoded to FP8 E4M3 and
scaled back with row scales. The runtime roundtrip check compares the materialized F32 values before
and after runtime-buffer transfer, so `0.0` means the runtime transfer path is intact.

## Status

This completes the first runtime artifact-boundary smoke. It does not yet complete the T2 short
prompt guard, because the SQ FP8 materializer is not wired into the full package model load path.

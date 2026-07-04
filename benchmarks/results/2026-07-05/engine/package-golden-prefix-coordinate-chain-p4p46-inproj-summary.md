# Package golden-prefix coordinate chain check

Fixture:

- `benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16`

Packages:

- layer10 row-scale metadata:
  `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer10.ullm.d`
- layer6+layer10 row-scale metadata:
  `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer10.ullm.d`

Generated artifacts:

- `package-golden-prefix-coordinate-chain-layer10-actual-h3456-t0-p4p46-inproj.json`
- `package-golden-prefix-coordinate-chain-layer10-actual-h3456-t0-p4p46-inproj.md`
- `package-golden-prefix-coordinate-chain-layer6-layer10-actual-h3456-t0-p4p46-inproj.json`
- `package-golden-prefix-coordinate-chain-layer6-layer10-actual-h3456-t0-p4p46-inproj.md`
- `package-golden-prefix-coordinate-chain-layer6-layer10-actual-h3994-t11-p4p46-inproj.json`
- `package-golden-prefix-coordinate-chain-layer6-layer10-actual-h3994-t11-p4p46-inproj.md`

## Key comparison

Layer10-only metadata leaves hidden `3456` as the dominant actual-prefix coordinate:

| coordinate | package metadata | layer 10 input_diff | layer 10 delta_diff | layer 10 output_diff | layer 11 output_diff |
| --- | --- | ---: | ---: | ---: | ---: |
| token 0 hidden 3456 | layer10 | -0.769666672 | -0.198179245 | -0.967845917 | -0.911422729 |

Layer6+layer10 metadata removes hidden `3456` from the per-token hot-coordinate trace,
but hidden `3994` remains active across the whole prefix:

| coordinate | package metadata | layer 7 output_diff | layer 8 output_diff | layer 9 output_diff | layer 11 output_diff |
| --- | --- | ---: | ---: | ---: | ---: |
| token 11 hidden 3994 | layer6+layer10 | -0.463647842 | 0.211425781 | -0.300535202 | -0.891334534 |

## Interpretation

- The layer6+layer10 row-scale metadata addressed the previous hidden `3456`
  chain: that coordinate is no longer captured in the hot-coordinate trace.
- The remaining max error moved to hidden `3994`, with layer11 token `11`
  reaching `0.891334534`.
- Hidden `3994` is not shaped like a single final-row scale issue:
  layers `7`, `8`, `9`, `10`, and `11` alternately amplify and reverse the
  coordinate delta under actual-prefix inputs.
- The next debug step should focus on input-distribution-sensitive propagation
  through layers `7`, `8`, `9`, and `11`, not on blindly adding another row
  scale override.

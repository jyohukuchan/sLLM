# Qwen actual-input trace for hidden 3994

Fixture:

- `benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16`

Package:

- `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer10.ullm.d`

Input dump:

- `/tmp/qwen35-9b-prefix0-12-layer6-layer10-actual-inputs-p4p46-inproj`

Generated artifacts:

- `package-golden-prefix-cpu-actual-prefix0-12-manifest-row-scale-layer6-layer10-p4p46-inproj-input-dump.jsonl`
- `qwen-layer-module-trace-actual-input-layers7-9-11-hidden3994-layer6-layer10-p4p46-inproj.jsonl`
- `qwen-layer-module-trace-actual-input-layers7-9-11-hidden3994-layer6-layer10-p4p46-inproj.md`
- `qwen-module-trace-comparison-actual-input-layers7-9-11-hidden3994-layer6-layer10-p4p46-inproj.json`
- `qwen-module-trace-comparison-actual-input-layers7-9-11-hidden3994-layer6-layer10-p4p46-inproj.md`

## Local package error with actual-prefix inputs

This comparison feeds the same actual-prefix layer input into the package path
and the full-reference Qwen layer. The `delta_error` column is therefore local
package error for that layer/input, not inherited upstream drift.

| layer | token | package output diff | fullref delta | package delta | local delta error | attention row-only | attention activation-path | MLP row-only | MLP activation-path |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 8 | -0.756856918 | 1.719599724 | 1.087742805 | -0.631856918 | 0.006025539 | -1.085749765 | -0.035078970 | 0.414899407 |
| 8 | 12 | 0.675428391 | 2.329922676 | 2.505351067 | 0.175428391 | -0.024038778 | 0.129673032 | -0.024429226 | 0.052237593 |
| 9 | 3 | -0.417654037 | 1.733904037 | 1.816250000 | 0.082345963 | 0.022902974 | -0.145378160 | 0.080553811 | 0.028742901 |
| 11 | 6 | -0.889577866 | 1.375562668 | 1.360984802 | -0.014577866 | 0.065380225 | -0.119163659 | -0.042665088 | 0.066413937 |

## Interpretation

- Layer `11` is not the local source of the remaining hidden `3994` max error.
  Its package-vs-full-reference local delta error is only `-0.014577866`, while
  the package output diff is `-0.889577866`.
- Layer `7` is the strongest local package-error target under actual-prefix
  input. Its local delta error is `-0.631856918`.
- Layer `7` row-only error is tiny (`0.006025539`) but the attention
  activation-path error is large (`-1.085749765`). This points inside the
  self-attention activation path before `o_proj`, not at the final `o_proj`
  row itself.
- Layer `8` and layer `9` have smaller local errors. They still contribute to
  propagation, but they do not explain the remaining max as strongly as layer
  `7`.

Next useful target:

- Trace layer `7` self-attention internals for token `8`, hidden `3994`,
  especially the `o_proj` input feature `503`, where package/full-reference
  attention projection input differs by about `0.499137`.

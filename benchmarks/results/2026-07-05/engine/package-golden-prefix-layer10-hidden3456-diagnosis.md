# Package Golden Prefix Layer 10 Hidden 3456 Diagnosis

## Scope

Package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-reservoir65536-jobs4.ullm.d`

Fixture: `benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16`

Layer/token/hidden: layer `10`, token `0`, hidden `3456`

Run mode: `golden_before_each_layer`, with `actual_prefix` checked for accumulation.

## Golden-Before Module Split

| metric | value |
| --- | ---: |
| package output diff | -0.875896454 |
| expected delta | 22.75 |
| package delta | 21.874103546 |
| actual input diff | 0 |
| attention error | -0.183621407 |
| attention row-only error | -0.169056547 |
| attention activation-path error | -0.021181354 |
| MLP error | -0.754775047 |
| MLP row-only error | -0.613532790 |
| MLP activation-path error | -0.172058034 |

## Row Reconstruction Check

| tensor | row | row_rms | row_rel_mse | row_max_abs | max_col | manifest_max_abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `linear_attn.out_proj.weight` | 3456 | 0.001093848 | 0.003870684 | 0.017172441 | 1676 | 0.031907110 |
| `mlp.down_proj.weight` | 3456 | 0.000707810 | 0.007366217 | 0.018229157 | 5301 | 0.026692718 |

## Actual-Prefix Accumulation

| metric | value |
| --- | ---: |
| output diff | -1.744266510 |
| input diff | -0.769666672 |
| delta diff | -0.974599838 |
| expected delta | 22.75 |
| actual delta | 21.775400162 |
| attention output | 7.260499954 |
| MLP output | 14.514899254 |

## Interpretation

The layer 10 outlier differs from the earlier layer 6 issue. The layer 6 issue was dominated by a missing implementation primitive before the final projection. Here, the hot input vectors are close to the full reference, and the dominant golden-before error is the final projection row dot product:

- attention row-only error is `-0.169056547`, while attention activation-path error is only `-0.021181354`
- MLP row-only error is `-0.613532790`, while MLP activation-path error is `-0.172058034`

The row reconstruction max absolute error is not large by itself, but the error aligns with a high-impact activation direction. In `actual_prefix`, the same coordinate also receives an input diff of `-0.769666672`, so prefix accumulation and layer-local projection row error combine into the `-1.744266510` output diff.

Next useful work is quantization-policy or row-compensation investigation for sensitive output rows such as layer 10 `mlp.down_proj[3456]`, not another backend comparison.

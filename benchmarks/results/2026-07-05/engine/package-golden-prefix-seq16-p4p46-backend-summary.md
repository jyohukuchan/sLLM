# Package Golden Prefix Seq16 Backend Summary

## Scope

Package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-reservoir65536-jobs4.ullm.d`

Fixture: `benchmarks/golden/2026-07-05/qwen35-9b-prefix0-8-seq16`

Layers: `0..8`

Sequence length: `16`

## Results

| backend | device | run_mode | max_mse | max_mean_abs_diff | max_abs_diff | min_cosine_similarity | worst_layer | worst_token | worst_hidden |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | 0 | golden_before_each_layer | 0.000361034838 | 0.014008828 | 0.486213684 | 0.998973254 | 6 | 0 | 3994 |
| R9700 | 2 | golden_before_each_layer | 0.000361035509 | 0.014008830 | 0.486255646 | 0.998973254 | 6 | 0 | 3994 |
| V620 | 1 | golden_before_each_layer | 0.000361035509 | 0.014008830 | 0.486255646 | 0.998973254 | 6 | 0 | 3994 |
| CPU | 0 | actual_prefix | 0.001575089386 | 0.030268902 | 0.665708542 | 0.995290748 | 7 | 0 | 3994 |
| R9700 | 2 | actual_prefix | 0.001575093148 | 0.030268933 | 0.665769577 | 0.995290736 | 7 | 0 | 3994 |
| V620 | 1 | actual_prefix | 0.001575093148 | 0.030268933 | 0.665769577 | 0.995290736 | 7 | 0 | 3994 |

## Seq8 Comparison

The seq16 fixture did not introduce a larger outlier than the previous seq8 fixture after the Conv+SiLU and self-attention input RMSNorm fixes.

| sequence_len | run_mode | CPU max_mse | CPU max_mean_abs_diff | CPU max_abs_diff | CPU min_cosine_similarity |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8 | golden_before_each_layer | 0.000487887468 | 0.016121546 | 0.486274719 | 0.999005169 |
| 16 | golden_before_each_layer | 0.000361034838 | 0.014008828 | 0.486213684 | 0.998973254 |
| 8 | actual_prefix | 0.001895285663 | 0.033220127 | 0.665708542 | 0.995342680 |
| 16 | actual_prefix | 0.001575089386 | 0.030268902 | 0.665708542 | 0.995290748 |

## Interpretation

The seq16 p4p46 run is backend-stable across CPU, RDNA4 R9700, and RDNA2 V620. The persistent worst coordinate is hidden `3994`, but the large implementation-mismatch outliers found earlier do not return at the longer tested prefix length.

The next validation step should move toward longer layer coverage or logits/generation checks rather than more CPU/HIP backend comparison on this fixture.

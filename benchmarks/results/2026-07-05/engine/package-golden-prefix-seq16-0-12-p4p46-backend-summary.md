# Package Golden Prefix Seq16 0..12 Backend Summary

## Scope

Package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-reservoir65536-jobs4.ullm.d`

Fixture: `benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16`

Layers: `0..12`

Sequence length: `16`

## Results

| backend | device | run_mode | max_mse | max_mean_abs_diff | max_abs_diff | min_cosine_similarity | worst_layer | worst_token | worst_hidden |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | 0 | golden_before_each_layer | 0.000361034838 | 0.014148540 | 0.875896454 | 0.998973254 | 10 | 0 | 3456 |
| R9700 | 2 | golden_before_each_layer | 0.000361035509 | 0.014148535 | 0.875885010 | 0.998973254 | 10 | 0 | 3456 |
| V620 | 1 | golden_before_each_layer | 0.000361035509 | 0.014148535 | 0.875885010 | 0.998973254 | 10 | 0 | 3456 |
| CPU | 0 | actual_prefix | 0.003190722428 | 0.043881594 | 1.744266510 | 0.994555928 | 10 | 0 | 3456 |
| R9700 | 2 | actual_prefix | 0.003190725513 | 0.043881624 | 1.744228363 | 0.994555922 | 10 | 0 | 3456 |
| V620 | 1 | actual_prefix | 0.003190725513 | 0.043881624 | 1.744228363 | 0.994555922 | 10 | 0 | 3456 |

## CPU Actual-Prefix Layer Progression

| layer | mse | mean_abs_diff | max_abs_diff | cosine_similarity | worst_token | worst_hidden |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.000013944369 | 0.002744401 | 0.108239651 | 0.999292389 | 3 | 3994 |
| 1 | 0.000064639891 | 0.005930064 | 0.255608559 | 0.998517863 | 13 | 3994 |
| 2 | 0.000156658328 | 0.009395298 | 0.353747368 | 0.996677839 | 12 | 3994 |
| 3 | 0.000360795679 | 0.014560507 | 0.456401825 | 0.997113943 | 0 | 3994 |
| 4 | 0.000525860569 | 0.017613544 | 0.542758942 | 0.996633897 | 14 | 3994 |
| 5 | 0.000741986696 | 0.020962822 | 0.516991615 | 0.995970343 | 14 | 3994 |
| 6 | 0.001379608173 | 0.028198784 | 0.588710785 | 0.996029908 | 0 | 3456 |
| 7 | 0.001575089386 | 0.030268902 | 0.665708542 | 0.995290748 | 0 | 3994 |
| 8 | 0.001853344736 | 0.032951979 | 0.735359192 | 0.994910807 | 0 | 3456 |
| 9 | 0.002163299456 | 0.035897321 | 0.769666672 | 0.994762951 | 0 | 3456 |
| 10 | 0.002704375912 | 0.040026371 | 1.744266510 | 0.995233829 | 0 | 3456 |
| 11 | 0.003190722428 | 0.043881594 | 1.686901093 | 0.994555928 | 0 | 3456 |

## Interpretation

The 0..12 seq16 run is backend-stable across CPU, RDNA4 R9700, and RDNA2 V620. Extending from 0..8 to 0..12 exposes a new layer 10 outlier at token `0`, hidden `3456`.

This outlier is already visible in `golden_before_each_layer`, so it is not only prefix accumulation. `actual_prefix` then roughly doubles the max absolute difference at the same coordinate. The next narrow debug target is layer 10 token `0` hidden `3456`, with module-level tracing before moving to broader logits or generation checks.

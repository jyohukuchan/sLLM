# Package Golden Prefix HIP Backend Summary

## Scope

Package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-reservoir65536-jobs4.ullm.d`

Fixture: `benchmarks/golden/2026-07-04/qwen35-9b-prefix0-8-seq8`

Layers: `0..8`

## Results

| backend | device | run_mode | max_mse | max_mean_abs_diff | max_abs_diff | min_cosine_similarity |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| CPU | 0 | golden_before_each_layer | 0.000487887468 | 0.016121546 | 0.486274719 | 0.999005169 |
| R9700 | 2 | golden_before_each_layer | 0.000487887774 | 0.016121550 | 0.486297607 | 0.999005168 |
| V620 | 1 | golden_before_each_layer | 0.000487887774 | 0.016121550 | 0.486297607 | 0.999005168 |
| CPU | 0 | actual_prefix | 0.001895285663 | 0.033220127 | 0.665708542 | 0.995342680 |
| R9700 | 2 | actual_prefix | 0.001895293292 | 0.033220179 | 0.665769577 | 0.995342660 |
| V620 | 1 | actual_prefix | 0.001895293292 | 0.033220179 | 0.665769577 | 0.995342660 |

## Model-Loop Smoke

`package-self-attn-mlp-block-model-loop-smoke` with layers `[3, 7]`, sequence length `3`, and p4p46-inproj passed on:

- CPU device `0`: runtime/cache diffs all `0`
- R9700 device `2`: runtime/cache diffs all `0`, prepared q/k/RoPE/causal attention diffs in the existing small HIP range
- V620 device `1`: runtime/cache diffs all `0`, prepared q/k/RoPE/causal attention diffs in the existing small HIP range

## Interpretation

After the Conv+SiLU and self-attention input RMSNorm fixes, the p4p46 golden-prefix results are backend-stable across CPU, RDNA4 R9700, and RDNA2 V620. The remaining drift is dominated by package quantization and prefix accumulation rather than backend execution differences.

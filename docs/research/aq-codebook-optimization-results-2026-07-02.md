# AQ codebook alternating optimization results 2026-07-02

## Scope

This note records the first CPU experiment for scale-aware AQ codebook
optimization. The experiment compares the current baseline against alternating
optimization of:

```text
raw-value ~= global-scale * local-scale * codebook[codebook-index]
```

Constraints used in the experiment:

- `codebook` is rounded to FP16 after initialization and after each update.
- `local-scale` is selected from the candidate scale table, such as E4M3,
  E5M2, UE5M3, or E8M0.
- `global-scale` is rounded to FP16 or FP32 according to the run option.
- `codebook-index` is 4-bit, so each codebook has 16 values.

The implementation is `tools/run-aq-codebook-opt-experiment.py`. It uses
sampled blocks rather than full tensor conversion, and is intended as a
research tool before moving the method into the Rust/C++ quantizer.

## Algorithm

The alternating optimizer starts from the current codebook construction path:

1. sample blocks from one tensor,
2. build the initial codebook with the existing quantile/Lloyd helper,
3. choose the initial global-scale with the existing tensor-scale estimator,
4. assign `local-scale` and `codebook-index` with scale-window search,
5. update codebook by least squares for fixed assignments,
6. round codebook to FP16,
7. update global-scale by least squares for fixed assignments,
8. round global-scale to FP16 or FP32,
9. repeat, then run a final assignment pass.

This is not complete simultaneous optimization. It is a practical coordinate
descent path that includes the real storage constraints during codebook
creation.

## Result Files

- `benchmarks/results/2026-07-02/aq/2026-07-02-aq-codebook-opt-smoke-mlp-up-32k.json`
- `benchmarks/results/2026-07-02/aq/2026-07-02-aq-codebook-opt-family4-262k-it3.json`
- `benchmarks/results/2026-07-02/aq/2026-07-02-aq-codebook-opt-family4-1m-it3.json`
- `benchmarks/results/2026-07-02/aq/2026-07-02-aq-codebook-opt-mlp-up-262k-it1.json`
- `benchmarks/results/2026-07-02/aq/2026-07-02-aq-codebook-opt-mlp-up-262k-it2.json`
- `benchmarks/results/2026-07-02/aq/2026-07-02-aq-codebook-opt-mlp-up-262k-it4.json`
- `benchmarks/results/2026-07-02/aq/2026-07-02-aq-codebook-opt-mlp-up-262k-it8.json`
- `benchmarks/results/2026-07-02/aq/2026-07-02-aq-codebook-opt-family4-262k-it8.json`
- `benchmarks/results/2026-07-02/aq/2026-07-02-aq-codebook-opt-family4-262k-it8-gsfp32.json`
- `benchmarks/results/2026-07-02/aq/2026-07-02-aq-codebook-opt-scale-formats-262k-it4.json`
- `benchmarks/results/2026-07-02/aq/2026-07-02-aq-codebook-opt-scale-formats-32k-it4-exhaustive.json`

## Main Results

### E4M3 g16/g8, 262k elements, 3 iterations

Families: `mlp_up`, `attn_k`, `attn_o`, `linear_attn_out`.
Tensors: 2 per family, 8 total. Candidates:
`aq4_e4m3_g16_ts_flloyd16`, `aq4_e4m3_g8_ts_flloyd16`.

| Metric | Value |
|---|---:|
| result rows | 16 |
| mean alternating/baseline relative MSE | 0.986642 |
| min alternating/baseline relative MSE | 0.981688 |
| max alternating/baseline relative MSE | 0.990079 |
| mean baseline time per row | 0.1227 s |
| mean alternating time per row | 0.5338 s |
| whole command elapsed | 12.31 s |
| max RSS | 520,916 KiB |

The improvement was consistent across all 16 rows.

### E4M3 g16/g8, 1M elements, 3 iterations

Families: `mlp_up`, `attn_k`, `attn_o`, `linear_attn_out`.
Tensors: 1 per family, 4 total.

| Metric | Value |
|---|---:|
| result rows | 8 |
| mean alternating/baseline relative MSE | 0.987425 |
| min alternating/baseline relative MSE | 0.985830 |
| max alternating/baseline relative MSE | 0.990265 |
| mean baseline time per row | 0.5712 s |
| mean alternating time per row | 2.1935 s |
| whole command elapsed | 24.01 s |
| max RSS | 580,316 KiB |

The 1M-element result is close to the 262k result, so the 262k sample is useful
for iteration search.

### Iteration Sweep

Tensor: `model.language_model.layers.14.mlp.up_proj.weight`.
Sample: 262k elements.

| iterations | g8 ratio | g16 ratio |
|---:|---:|---:|
| 1 | 0.990664 | 0.994782 |
| 2 | 0.988146 | 0.990876 |
| 4 | 0.984604 | 0.984764 |
| 8 | 0.980538 | 0.977082 |

Eight iterations still improved BF16-error in this smoke. More iterations may
continue to help, but the next step should check model-level quality before
spending much more time on tensor-only optimization.

### E4M3 g16/g8, 262k elements, 8 iterations

Families: `mlp_up`, `attn_k`, `attn_o`, `linear_attn_out`.
Tensors: 2 per family, 8 total.

| Metric | Value |
|---|---:|
| result rows | 16 |
| mean alternating/baseline relative MSE | 0.976375 |
| min alternating/baseline relative MSE | 0.965344 |
| max alternating/baseline relative MSE | 0.983151 |
| mean baseline time per row | 0.1273 s |
| mean alternating time per row | 1.1493 s |
| whole command elapsed | 22.23 s |
| max RSS | 518,124 KiB |

This is the best current tensor-BF16-error result from the first pass.

### FP16 vs FP32 global-scale

Using the same 262k, 8-iteration family4 setup:

| global-scale dtype | mean ratio | min ratio | max ratio |
|---|---:|---:|---:|
| FP16 | 0.976375 | 0.965344 | 0.983151 |
| FP32 | 0.976359 | 0.965403 | 0.983171 |

Mean FP32-minus-FP16 relative-MSE delta was `-9.70e-08`. In this experiment,
FP16 global-scale is effectively enough.

### local-scale Format Comparison

Sample: 262k elements, 4 iterations, `mlp_up` and `linear_attn_out`, 4 tensors.
These rows use `zero_lloyd15` candidates to compare local-scale formats.

| candidate | baseline mean relative MSE | alternating mean relative MSE | mean ratio |
|---|---:|---:|---:|
| `aq4_e4m3_g16_ts_zlloyd15` | 0.005646 | 0.005430 | 0.961676 |
| `aq4_ue5m3_g16_ts_zlloyd15` | 0.005657 | 0.005461 | 0.965372 |
| `aq4_e5m2_g16_ts_zlloyd15` | 0.006339 | 0.006134 | 0.967756 |
| `aq4_e8m0_g16_zlloyd15` | 0.009578 | 0.009072 | 0.947190 |

E8M0 benefits from alternating optimization, but its absolute BF16-error is
still much worse than E4M3/UE5M3 in this test.

## Simultaneous Optimization Cost Reference

The script estimates exhaustive local assignment cost for fixed codebook and
global-scale:

```text
blocks * scale_candidates * block-size * 16
```

Baseline scale-window search uses:

```text
blocks * (2 * scale-window + 1) * block-size * 16
```

With scale-window 4, exhaustive local-scale search is approximately:

| local-scale format | scale candidates | exhaustive/baseline ops |
|---|---:|---:|
| E4M3 | 119 | 13.22x |
| E5M2 | 123 | 13.67x |
| UE5M3 | 255 | 28.33x |
| E8M0 | 255 | 28.33x |

Small 32k-element timing for one `mlp_up` tensor:

| candidate | baseline time | alternating time, 4 iters | exhaustive local assignment time |
|---|---:|---:|---:|
| `aq4_e4m3_g16_ts_zlloyd15` | 0.0365 s | 0.0861 s | 0.0464 s |
| `aq4_e5m2_g16_ts_zlloyd15` | 0.0293 s | 0.0826 s | 0.0481 s |
| `aq4_ue5m3_g16_ts_zlloyd15` | 0.0292 s | 0.0822 s | 0.0988 s |
| `aq4_e8m0_g16_zlloyd15` | 0.0286 s | 0.0794 s | 0.0892 s |

This exhaustive measurement is only the local assignment part with fixed
codebook and global-scale. A true brute-force simultaneous optimization over
FP16 codebook values is not practical. The useful path is coordinate descent:
local assignment, codebook least-squares update, global-scale least-squares
update, then repeat.

## Conclusions

- The constraint-aware alternating optimizer is practical on CPU samples.
- It consistently reduced BF16-error versus the current baseline.
- E4M3 g16/g8 with 8 iterations reduced tensor relative MSE by about 2.4% on
  the representative 262k family4 run.
- FP16 global-scale appears sufficient for these tensors.
- E8M0 remains much worse in absolute BF16-error despite a larger relative gain
  from alternating optimization.
- Full simultaneous brute force is not realistic; exhaustive local-scale search
  alone is already 13x to 28x the baseline operation count depending on
  local-scale format.

## Next Steps

1. Add activation-weighted alternating optimization.
2. Export optimized family-level codebooks and run the Rust/C++ full-tensor
   converter path.
3. Compare full-tensor BF16-error against the current family codebook artifacts.
4. Only after that, run model-level loss checks, because tensor BF16-error is
   useful but mixed-policy behavior has already shown model-level surprises.

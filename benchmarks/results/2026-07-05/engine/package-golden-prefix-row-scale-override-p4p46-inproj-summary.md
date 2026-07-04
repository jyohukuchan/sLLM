# Package Golden Prefix Row-Scale Override Summary

## Scope

- Package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-reservoir65536-jobs4.ullm.d`
- Fixture: `benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16`
- Layers: `0..12`
- Override file: `benchmarks/results/2026-07-05/engine/package-row-scale-overrides-layer10-hidden3456-p4p46-inproj.json`
- Override rows:
  - layer `10`, `linear_attn.out_proj.weight`, row `3456`, scale `1.02307179310`
  - layer `10`, `mlp.down_proj.weight`, row `3456`, scale `1.04165701172`

The no-override CPU baseline was regenerated with the same binary used for the override runs. Older artifacts are not used as the baseline in this summary.

## Aggregate Results

| Backend | Mode | Override | Max MSE | Max MSE Layer | Max Mean Abs Diff | Max Mean Layer | Max Abs Diff | Max Abs Layer | Min Cosine | Min Cos Layer |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | golden_before_each_layer | no | `0.000715637264` | `11` | `0.020423743` | `11` | `0.875896454` | `10` | `0.998661227` | `7` |
| CPU | golden_before_each_layer | yes | `0.000715637264` | `11` | `0.020423743` | `11` | `0.486213684` | `6` | `0.998661227` | `7` |
| CPU | actual_prefix | no | `0.004029775765` | `11` | `0.049407153` | `11` | `1.744266510` | `10` | `0.993128410` | `11` |
| CPU | actual_prefix | yes | `0.003994860341` | `11` | `0.049373711` | `11` | `0.967845917` | `10` | `0.993178138` | `11` |
| R9700 | golden_before_each_layer | yes | `0.000715636461` | `11` | `0.020423736` | `11` | `0.486255646` | `6` | `0.998661227` | `7` |
| R9700 | actual_prefix | yes | `0.003994862952` | `11` | `0.049373733` | `11` | `0.967796326` | `10` | `0.993178132` | `11` |

## Layer 10 Effect

| Backend | Mode | Override | Layer 10 MSE | Layer 10 Mean Abs Diff | Layer 10 Max Abs Diff | Layer 10 Cosine |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| CPU | golden_before_each_layer | no | `0.000309102871` | `0.013572958` | `0.875896454` | `0.999455637` |
| CPU | golden_before_each_layer | yes | `0.000297667653` | `0.013563661` | `0.304975510` | `0.999477328` |
| CPU | actual_prefix | no | `0.003104859460` | `0.043039161` | `1.744266510` | `0.994523513` |
| CPU | actual_prefix | yes | `0.003072910419` | `0.043027699` | `0.967845917` | `0.994575963` |
| R9700 | golden_before_each_layer | yes | `0.000297667313` | `0.013563653` | `0.304971695` | `0.999477329` |
| R9700 | actual_prefix | yes | `0.003072915173` | `0.043027737` | `0.967796326` | `0.994575954` |

## Interpretation

- The targeted two-row scale override materially reduces the layer `10`, hidden `3456` outlier.
- In `golden_before_each_layer`, the global max abs shifts from layer `10` to layer `6`.
- In `actual_prefix`, the global max abs remains layer `10`, but falls from `1.744266510` to about `0.9678`.
- The aggregate MSE and mean-absolute-diff remain dominated by layer `11`; row scaling layer `10` does not address that separate drift.
- CPU and R9700 override results are backend-stable within the usual tiny CPU/HIP differences.

## Decision

This validates row-dot compensation as a useful targeted experiment, not as a finished production mechanism. The next step is to move from smoke-only runtime row scaling to a quantizer-side row compensation design, while separately diagnosing the layer `11` aggregate drift.

## Layer 11 Follow-Up

A first self-attention-layer MLP row-scale probe was also run with:

- Override file: `benchmarks/results/2026-07-05/engine/package-row-scale-overrides-layer11-hidden3377-self-attn-mlp-p4p46-inproj.json`
- Override row: layer `11`, `mlp.down_proj.weight`, row `3377`, scale `1.218300518695`
- Report: `benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-golden-before-layer11-row-scale-override-p4p46-inproj.jsonl`

Result:

| Backend | Mode | Override | Layer 11 MSE | Layer 11 Mean Abs Diff | Layer 11 Max Abs Diff | Layer 11 Cosine |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| CPU | golden_before_each_layer | no | `0.000715637264` | `0.020423743` | `0.179061234` | `0.998779597` |
| CPU | golden_before_each_layer | row 3377 | `0.000714828336` | `0.020418806` | `0.167163849` | `0.998780983` |

The layer `11` probe is only a weak improvement. After row `3377` is scaled, the max-diff coordinate moves to hidden `3994`, so layer `11` looks like a broader multi-row or mixed-path drift rather than the same single-row pattern seen at layer `10`.

A second probe added row `3994` with a least-squares scale derived after applying row `3377`:

- Override file: `benchmarks/results/2026-07-05/engine/package-row-scale-overrides-layer11-hidden3377-3994-self-attn-mlp-p4p46-inproj.json`
- Additional row: layer `11`, `mlp.down_proj.weight`, row `3994`, scale `1.020286172534`
- Report: `benchmarks/results/2026-07-05/engine/package-golden-prefix-cpu-golden-before-layer11-two-row-scale-override-p4p46-inproj.jsonl`

Result:

| Backend | Mode | Override | Layer 11 MSE | Layer 11 Mean Abs Diff | Layer 11 Max Abs Diff | Layer 11 Cosine |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| CPU | golden_before_each_layer | rows 3377 + 3994 | `0.000714424519` | `0.020416921` | `0.179649353` | `0.998781590` |

The two-row version improves MSE only marginally and worsens max abs compared with the single-row probe. This reinforces that layer `11` should not be handled by hand-picked row scaling without a proper self-attention-layer row-dot trace.

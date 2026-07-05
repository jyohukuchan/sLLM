# Qwen Hidden3994 Named Policy Accepted Package Recipe

## Purpose

Rebuild the accepted hidden3994 package candidate with explicit, reproducible
inputs. This recipe keeps the existing layer6/layer10 row3456 manifest
compensation and uses the named quantizer policy
`qwen35_9b_p4p46_hidden3994_v1`.

The named policy is equivalent to `p4p46_inproj` plus two exact layer8 high AQ
tensor promotions:

- `model.language_model.layers.8.linear_attn.in_proj_qkv.weight`
- `model.language_model.layers.8.mlp.up_proj.weight`

The generic `p4p46_inproj` policy remains unchanged.

## Quantizer Plan

```bash
target/debug/ullm-quant direct-package \
  /home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B \
  /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-qwen35-hidden3994-v1-row-scale-layer6-layer10.ullm.d \
  --aq-policy qwen35_9b_p4p46_hidden3994_v1 \
  --row-scale-overrides-json benchmarks/results/2026-07-05/engine/package-row-scale-overrides-layer6-layer10-hidden3456-p4p46-inproj.json \
  --tensor-scale-estimator reservoir \
  --tensor-scale-reservoir-size 65536 \
  --jobs 64
```

Observed build:

- selected tensors: `255`
- passthrough tensors: `520`
- codebooks: `14`
- failures: `0`
- total file bytes: `9127853385`
- package summary: `ullm-quant-direct-package-fullpkg-qwen35-9b-qwen35-hidden3994-v1-row-scale-layer6-layer10-jobs64.json`
- independent verify: `ullm-quant-direct-package-fullpkg-qwen35-9b-qwen35-hidden3994-v1-row-scale-layer6-layer10-jobs64-verify.log`

## Acceptance Checks

```bash
python3 tools/run-qwen-prefix-smoke-matrix.py \
  --engine-bin target/debug/ullm-engine \
  --package /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer10.ullm.d \
  --fixture tokens1=benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16 \
  --fixture tokens101=benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16-tokens101-116 \
  --fixture tokens201=benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16-tokens201-216 \
  --fixture tokens301=benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16-tokens301-316 \
  --fixture tokens401=benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16-tokens401-416 \
  --condition baseline \
  --condition qwen35-hidden3994-policy,package=/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-qwen35-hidden3994-v1-row-scale-layer6-layer10.ullm.d \
  --output-dir benchmarks/results/2026-07-05/engine/qwen-prefix-smoke-matrix-qwen35-hidden3994-policy-cpu-five-fixture \
  --summary-json benchmarks/results/2026-07-05/engine/qwen-prefix-smoke-matrix-qwen35-hidden3994-policy-cpu-five-fixture/summary.json \
  --markdown benchmarks/results/2026-07-05/engine/qwen-prefix-smoke-matrix-qwen35-hidden3994-policy-cpu-five-fixture/summary.md \
  --device-index 0 \
  --chunk-bytes 1048576 \
  --layer-start 0 \
  --layer-end 12 \
  --rotary-dim 64 \
  --rope-base 10000000 \
  --position-offset 0 \
  --run-mode actual_prefix
```

Then summarize and gate:

```bash
python3 tools/summarize-qwen-prefix-smokes.py \
  --matrix-summary-json benchmarks/results/2026-07-05/engine/qwen-prefix-smoke-matrix-qwen35-hidden3994-policy-cpu-five-fixture/summary.json \
  --summary-json benchmarks/results/2026-07-05/engine/qwen-prefix-qwen35-hidden3994-policy-cpu-five-fixture-summary.json \
  --markdown benchmarks/results/2026-07-05/engine/qwen-prefix-qwen35-hidden3994-policy-cpu-five-fixture-summary.md

python3 tools/evaluate-qwen-prefix-candidate-gates.py \
  --summary-json benchmarks/results/2026-07-05/engine/qwen-prefix-qwen35-hidden3994-policy-cpu-five-fixture-summary.json \
  --output-json benchmarks/results/2026-07-05/engine/qwen-prefix-qwen35-hidden3994-policy-cpu-five-fixture-gates.json \
  --markdown benchmarks/results/2026-07-05/engine/qwen-prefix-qwen35-hidden3994-policy-cpu-five-fixture-gates.md \
  --baseline-condition baseline \
  --candidate-condition qwen35-hidden3994-policy \
  --max-fixture-worsen 0.001 \
  --min-median-improvement 0.005 \
  --min-fixture-count 5
```

Accepted result from the completed run:

- CPU five-fixture gate: `accept`
- R9700 five-fixture gate: `accept`
- V620 five-fixture gate: `accept`

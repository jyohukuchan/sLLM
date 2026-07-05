# Qwen Hidden3994 Quantizer Policy Branch v0.1

## Purpose

Manifest row-scale candidates did not produce an accepted hidden3994 fix.
The next branch is quantizer policy, starting with dry-run cost checks before building another full package.

## Current Policy Boundary

`ullm-quant` originally assigned AQ formats at tensor-family granularity.

- policy resolution: `resolve_aq_policy`
- family detection: `family_for_tensor`
- format assignment: `quant_assignment`
- direct package conversion boundary: `run_one_direct_package_convert`

There is still no row-level or activation-stat input in the current quantizer
path. A repeatable exact-name override, `--aq-high-tensor <TENSOR_NAME>`, was
added as the smallest narrower policy branch before building another package.

## Dry-Run Candidates

Baseline:

- command output: `qwen-hidden3994-policy-p4p46-inproj-baseline-dry-run.txt`
- plan: `qwen-hidden3994-policy-p4p46-inproj-baseline-plan.json`
- high families: `attn_o`, `attn_v`, `linear_attn_a`, `linear_attn_b`, `linear_attn_out`, `linear_attn_z`
- supported tensors: `255`
- high tensors: `114`
- low tensors: `141`
- estimated output bytes: `9121922016`

Custom `p4p46 + mlp_up`:

- command output: `qwen-hidden3994-policy-custom-p4p46-plus-mlp-up-dry-run.txt`
- plan: `qwen-hidden3994-policy-custom-p4p46-plus-mlp-up-plan.json`
- high families: baseline p4p46 high families plus `mlp_up`
- supported tensors: `255`
- high tensors: `147`
- low tensors: `108`
- estimated output bytes: `9225731040`
- estimated output increase: `103809024` bytes

Targeted `p4p46 + layer8 mlp.up_proj.weight`:

- command output: `qwen-hidden3994-policy-p4p46-plus-layer8-mlp-up-tensor-dry-run.txt`
- plan: `qwen-hidden3994-policy-p4p46-plus-layer8-mlp-up-tensor-plan.json`
- high tensors: baseline p4p46 high tensors plus `model.language_model.layers.8.mlp.up_proj.weight`
- supported tensors: `255`
- high tensors: `115`
- low tensors: `140`
- estimated output bytes: `9125067744`
- estimated output increase over baseline: `3145728` bytes
- exact override check: layer8 `mlp.up_proj.weight` is high, while layer9 `mlp.up_proj.weight` remains low.

## Interpretation

Raising all `mlp_up` tensors is cheap in estimated package size, but it is broad.
It is not activation-aware and may repeat the same problem seen with `p4p65`: family-level changes can improve one path while regressing another fixture.

The next policy experiment should be one of:

1. Build `p4p46 + mlp_up` only if a broad family probe is acceptable.
2. Build and smoke the narrower tensor override package.
3. Add row-aware or activation-aware policy input instead of family-wide high assignment.

Given the weak layer8-up6340 result, option 2 or 3 is more aligned with the evidence than a broad family-wide package.

## Targeted Layer8 MLP Up Package

Built package:

- quantizer package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-layer8-mlp-up-high-reservoir65536-jobs64.ullm.d`
- row-scale manifest package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-layer8-mlp-up-high-row-scale-layer6-layer10.ullm.d`
- package summary: `ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-layer8-mlp-up-high-reservoir65536-jobs64.json`
- verify log: `ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-layer8-mlp-up-high-reservoir65536-jobs64-verify.log`
- row-scale package build: `qwen-targeted-high-tensor-row-scale-layer6-layer10-package-build.json`

Build and verify:

- selected tensors: `255`
- passthrough tensors: `520`
- codebooks: `13`
- conversion failures: `0`
- build wall time: `1:34.98`
- max RSS: `3734884` KiB
- independent verify: `255` quantized tensors and `520` passthrough tensors, exit `0`

Prefilter:

- matrix: `qwen-prefix-smoke-matrix-targeted-high-layer8-mlp-up-prefilter/summary.json`
- summary: `qwen-prefix-targeted-high-layer8-mlp-up-prefilter-summary.json`
- gate: `qwen-prefix-targeted-high-layer8-mlp-up-prefilter-gates.json`
- tokens1: `0.645338058 -> 0.627647400`
- tokens401: `0.959306717 -> 0.974622726`
- decision: `reject`, max regression `0.0153160095`

The candidate is useful evidence but not a solution: exact layer8 `mlp.up_proj.weight`
high promotion improves tokens1 and worsens tokens401, so it preserves the same
cross-fixture sign conflict.

## Targeted Layer8 QKV Package

Built package:

- package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-layer8-qkv-high-row-scale-layer6-layer10.ullm.d`
- plan: `qwen-hidden3994-policy-p4p46-plus-layer8-qkv-tensor-plan.json`
- package summary: `ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-layer8-qkv-high-row-scale-layer6-layer10-jobs64.json`
- verify log: `ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-layer8-qkv-high-row-scale-layer6-layer10-jobs64-verify.log`

Build and verify:

- selected tensors: `255`
- passthrough tensors: `520`
- codebooks: `13`
- conversion failures: `0`
- build wall time: `1:40.36`
- max RSS: `3743064` KiB
- independent verify: `255` quantized tensors and `520` passthrough tensors, exit `0`

Prefilter:

- matrix: `qwen-prefix-smoke-matrix-targeted-high-layer8-qkv-prefilter/summary.json`
- summary: `qwen-prefix-targeted-high-layer8-qkv-prefilter-summary.json`
- gate: `qwen-prefix-targeted-high-layer8-qkv-prefilter-gates.json`
- tokens1: `0.645338058 -> 0.651521683`
- tokens401: `0.959306717 -> 0.919565201`
- decision: `reject`, max regression `0.00618362427`

This candidate is also not a solution by itself. It moves the hard fixture in
the desired direction while regressing tokens1, making combined qkv+MLP-up a
more targeted follow-up than broad family promotion.

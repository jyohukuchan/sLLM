# Qwen Hidden3994 Quantizer Policy Branch v0.1

## Purpose

Manifest row-scale candidates did not produce an accepted hidden3994 fix.
The next branch is quantizer policy, starting with dry-run cost checks before building another full package.

## Current Policy Boundary

`ullm-quant` currently assigns AQ formats at tensor-family granularity.

- policy resolution: `resolve_aq_policy`
- family detection: `family_for_tensor`
- format assignment: `quant_assignment`
- direct package conversion boundary: `run_one_direct_package_convert`

There is no row-level or activation-stat input in the current quantizer path.
The smallest existing policy experiment is therefore a custom family-level policy.

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

## Interpretation

Raising all `mlp_up` tensors is cheap in estimated package size, but it is broad.
It is not activation-aware and may repeat the same problem seen with `p4p65`: family-level changes can improve one path while regressing another fixture.

The next policy experiment should be one of:

1. Build `p4p46 + mlp_up` only if a broad family probe is acceptable.
2. Add a narrower tensor override policy, such as high-format only for selected `model.language_model.layers.N.mlp.up_proj.weight` tensors.
3. Add row-aware or activation-aware policy input instead of family-wide high assignment.

Given the weak layer8-up6340 result, option 2 or 3 is more aligned with the evidence than a broad family-wide package.

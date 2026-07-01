# aq policy isolation plan v0.1

## Purpose

The full inproj248 project-text loss smoke changed the current aq policy
baseline. Earlier 22/44-module smokes made p4p6 look like the safest mixed
policy, but the full 248-module run ranked:

1. all-g16
2. all-g8
3. p4p6
4. p4p46
5. p4p65

This plan isolates why mixed policies degrade before any aq precision policy is
treated as stable.

## Current Evidence

Primary result:

- summary:
  `benchmarks/results/2026-07-01/aq/2026-07-01-aq-module-loss-summary-projecttext32-inproj248-all-r9700-qwen35-9b-s256.json`
- scope:
  248 cumulative modules, 32 project-text prompts, sequence length 256
- runtime:
  `2:23:58`, max RSS `16371476 KiB`

Token-weighted loss deltas:

| variant | loss delta |
| --- | ---: |
| all-g16 | -0.000307545 |
| all-g8 | -0.000161491 |
| p4p6 | +0.005253904 |
| p4p46 | +0.005814455 |
| p4p65 | +0.009358883 |

Scope comparison:

- `benchmarks/results/2026-07-01/aq/2026-07-01-aq-module-loss-scope-comparison-projecttext32-qwen35-9b.json`

Mixed-policy delta minus all-g16:

| scope | p4p6 | p4p46 | p4p65 |
| --- | ---: | ---: | ---: |
| inproj22 | +0.000928007 | +0.001949616 | +0.001512036 |
| inproj44 | +0.000526816 | +0.001080275 | +0.002476983 |
| inproj248 | +0.005561449 | +0.006122001 | +0.009666428 |

## Working Hypotheses

1. Some g8-promoted attention or linear-attention families are harmful when
   applied across all layers, even though they help tensor MSE.
2. The problem is not simply "g8 is worse", because all-g8 beat every mixed
   policy in the full inproj248 run.
3. A precision mismatch between promoted attention families and g16 MLP families
   may be worse than uniformly using g16 or g8.
4. The Python loss smoke is too slow for broad policy search because it
   re-quantizes every selected weight for every variant.

## Required Tooling Change

Before running many full-scope policy tests, add one of these evaluation paths:

1. Preferred: load Rust-converted `.ullm.d` prototype tensors into the Python
   model for loss evaluation.
2. Implemented short-term path: add a disk cache keyed by model path,
   activation stats path, tensor name, shape, dtype, variant settings,
   scale window, codebook sample cap, and seed so
   `run-aq-module-loss-smoke.py` can reuse quantized tensors.
3. Fallback only for narrow smokes: keep the current Python re-quantization path
   and reduce scope or prompt count.

The cache/loader path should record:

- source BF16 tensor name and shape,
- quantized variant,
- codebook artifact path and hash if available,
- scale format, group size, scale window, tensor-scale estimator,
- relative MSE from verification,
- tool git commit.

## Experiment Order

### Phase 1: One-Family Promotions

Use all-g16 as the baseline. Promote exactly one family to g8 at a time over
the inproj248 selection.

Priority families:

1. `linear_attn_out`
2. `attn_o`
3. `attn_v`
4. `attn_k`
5. `linear_attn_qkv`
6. `linear_attn_z`
7. `linear_attn_a`
8. `linear_attn_b`
9. `mlp_up`
10. `mlp_gate`
11. `mlp_down`

Record each row as:

- policy id: `promote_<family>`
- promoted family byte size,
- token-weighted loss delta,
- delta minus all-g16,
- wall time and max RSS.

### Phase 2: Minimal Pair Policies

After Phase 1, test only pairs that are justified by Phase 1.

Initial candidates:

- `linear_attn_out + attn_o`
- `attn_k + attn_o + attn_v`
- `linear_attn_a + linear_attn_b + linear_attn_z`
- `p4p6 + mlp_up`
- `p4p6 + mlp_up + mlp_gate + mlp_down`

The last two test whether the mixed-policy loss penalty comes from precision
mismatch between attention and MLP blocks.

### Phase 3: Full Policy Confirmation

Only after a Phase 1/2 policy beats or matches all-g16:

- rerun on the 32-prompt project-text set,
- add a second prompt corpus,
- run sequence length 512 if VRAM allows,
- compare against direct-package byte size and tensor MSE.

## Decision Rules

- Do not choose a policy from tensor MSE alone.
- Treat all-g16 as the conservative baseline until a full-scope loss run beats
  it or is statistically indistinguishable with a meaningful size/performance
  benefit.
- Treat negative loss deltas as noise unless they reproduce on another corpus.
- Any policy worse than all-g16 by more than `+0.001` token-weighted loss delta
  should be rejected or narrowed.

## Immediate Next Step

Use `tools/run-aq-module-loss-smoke.py --quantized-cache-dir <dir>` for Phase 1
so repeated policies can reuse quantized tensors. Then run Phase 1 for
`linear_attn_out`, `attn_o`, `attn_v`, and `mlp_up` first.

The current cache is a short-term Python/PyTorch cache. It is not a substitute
for a Rust `.ullm.d` loader path, but it avoids re-quantizing identical
module/variant/settings combinations across policy experiments.

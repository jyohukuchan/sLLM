# Qwen layer 8 attention QKV V845 cell-delta experiment

Artifacts:

- `package-cell-delta-overrides-layer8-qkv-v845-col3994-p4p46-inproj.json`
- `package-cell-delta-overrides-layer8-qkv-v845-col3994-lsfit-p4p46-inproj.json`
- `package-golden-prefix-cpu-actual-prefix0-12-rotary64-manifest-row-scale-layer6-layer10-cell-delta-layer8qkv-v845col3994-p4p46-inproj.jsonl`
- `package-golden-prefix-cpu-actual-prefix0-12-rotary64-manifest-row-scale-layer6-layer10-cell-delta-layer8qkv-v845col3994-lsfit-p4p46-inproj.jsonl`

## Target

The layer `8`, token `7`, hidden `3994` attention activation-path error is
driven by attention projection input feature `845`.

From the tracked row-dot trace, feature `845` maps to the V slice of
`linear_attn.in_proj_qkv.weight`:

- q rows: `0..2047`
- k rows: `2048..4095`
- v rows: `4096..8191`
- V feature `845` row: `4096 + 845 = 4941`

For `linear_attn.in_proj_qkv.weight[4941,3994]`:

| item | value |
| --- | ---: |
| source weight | 0.010986328125 |
| package weight | 0.008613099344 |
| weight error | -0.002373228781 |
| source-restore delta | 0.002373228781 |

At token `7`, this one cell contributes `-0.097302380` to the row-dot error,
while the full row package-vs-source dot error is `-0.065318765`.

The all-token row-dot least-squares delta for the same cell is
`0.001503075294`. It improves row-dot RMSE from `0.082291864` to
`0.057757336`, but is weaker than source restoration.

## Full Prefix Result

Run settings:

- package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer10.ullm.d`
- fixture: `benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16`
- run mode: `actual_prefix`
- layers: `0..12`
- `rotary_dim=64`
- CPU backend

| variant | overall max_abs | max layer | max token/hidden | layer 11 max_abs | layer 11 token/hidden |
| --- | ---: | ---: | --- | ---: | --- |
| baseline layer6/layer10 row-scale | 0.645338058 | 11 | token 7 / hidden 3994 | 0.645338058 | token 7 / hidden 3994 |
| qkv V845 cell source-restore | 0.627647400 | 7 | token 0 / hidden 3994 | 0.619235992 | token 7 / hidden 3994 |
| qkv V845 cell LS fit | 0.628797531 | 11 | token 7 / hidden 3994 | 0.628797531 | token 7 / hidden 3994 |

Layer detail:

| variant | layer | max_abs | mean_abs | mse | max token/hidden |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 8 | 0.578010559 | 0.032954554 | 0.001845195 | token 3 / hidden 3994 |
| qkv V845 cell | 8 | 0.588329315 | 0.032803790 | 0.001830218 | token 3 / hidden 3994 |
| qkv V845 cell LS fit | 8 | 0.584562302 | 0.032855703 | 0.001835375 | token 3 / hidden 3994 |
| baseline | 11 | 0.645338058 | 0.043868927 | 0.003148948 | token 7 / hidden 3994 |
| qkv V845 cell | 11 | 0.619235992 | 0.043696373 | 0.003124643 | token 7 / hidden 3994 |
| qkv V845 cell LS fit | 11 | 0.628797531 | 0.043756262 | 0.003133063 | token 7 / hidden 3994 |

Layer `8`, hidden `3994` token detail:

| variant | token | output diff | attention output | MLP output | delta diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 3 | -0.578010559 | 0.328756571 | 0.160099775 | -0.136144638 |
| qkv V845 cell | 3 | -0.588329315 | 0.321091354 | 0.157445222 | -0.146463394 |
| baseline | 7 | 0.296178818 | 0.999587059 | 0.077883139 | 0.452468872 |
| qkv V845 cell | 7 | 0.281913757 | 0.991616368 | 0.071586035 | 0.438203812 |

## Interpretation

- This is the first sparse cell-delta experiment in this batch that improves
  the full actual-prefix max.
- The previous MLP cell corrections improved layer `8` token `7` locally but
  worsened full-prefix max. In contrast, correcting the attention V projection
  row reduces the layer `11` hidden `3994` chain enough that the remaining
  overall max moves back to layer `7`.
- The layer `8` local max at token `3` worsens slightly, so this is still not a
  finished correction. It is a strong localization result: the most useful next
  target is the attention-side `linear_attn.in_proj_qkv` V row around
  `row=4941`, especially column/group `3994`.
- The LS-fit delta is more conservative and reduces the layer `8` local
  worsening, but it leaves a larger layer `11` max than source restoration.
  For the current full-prefix objective, source restoration is the better
  single-cell probe.
- A promotion candidate should not be a single smoke-only cell override. The
  next step is a controlled row/group compensation or quantizer policy experiment
  for this qkv V row, evaluated against full-prefix hidden error.

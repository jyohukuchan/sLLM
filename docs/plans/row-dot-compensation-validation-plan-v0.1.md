# Row-Dot Compensation Validation Plan v0.1

## 前回の要点

- Qwen3.5-9B p4p46-inproj packageで、Qwen3.5参照実装との差分を `package-golden-prefix-smoke` で検証した。
- Conv1d後のSiLU欠落と、self-attention q/k/v投影前のinput RMSNorm欠落は修正済み。
- 修正後、`seq8 / 0..8` と `seq16 / 0..8` はCPU/R9700/V620でbackend-stableになった。
- `seq16 / 0..12` へ広げると、layer `10`, token `0`, hidden `3456` が新しい最大outlierになった。
- p4p46-inprojはp4p6/p4p65-inprojより良いが、layer 10 hidden `3456` のoutlierは消えない。

## 今回の変更点

- module traceとrow-dot sensitivity分析で、layer 10 hidden `3456` の誤差は主に最終projection row dotにあると分かった。
- p4p46-inprojのlayer 10 hidden `3456`:
  - `mlp_down_proj`: original max abs row-dot error `0.613532790`
  - `attention_out_proj`: original max abs row-dot error `0.169056547`
- 16-token trace上では、単純なrow scaleでrow-dot誤差が大きく縮む見込みがある。
  - `mlp_down_proj`: optimal scale `1.04165701172`, scaled max abs `0.014996152`
  - `attention_out_proj`: optimal scale `1.02307179310`, scaled max abs `0.006276145`
- 既存trace横断でも同種の候補が複数出た。
  - layer `6`, hidden `3456`, `mlp_down_proj`: original max abs `0.442788575`, scaled max abs `0.026397223`
  - layer `6`, hidden `3456`, `attention_out_proj`: original max abs `0.228395332`, scaled max abs `0.009216717`

## 次の行動

最小実験は、`package-golden-prefix-smoke` に検証専用のrow scale overrideを追加し、layer `10` の `linear_attn.out_proj[3456]` と `mlp.down_proj[3456]` だけをスケールして `seq16 / 0..12` の `golden_before_each_layer` と `actual_prefix` を再実行する。

## Scope

対象:

- package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-reservoir65536-jobs4.ullm.d`
- fixture: `benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16`
- primary backend: CPU `0`
- validation backend: R9700 `2` after CPU resultが良い場合
- first rows:
  - `model.language_model.layers.10.linear_attn.out_proj.weight`, row `3456`, scale `1.02307179310`
  - `model.language_model.layers.10.mlp.down_proj.weight`, row `3456`, scale `1.04165701172`

対象外:

- package file formatの変更
- permanent runtime APIの変更
- quantizer本体のpolicy変更
- all-layer/all-row自動補正
- logits/generation評価

## Implementation Plan

### T1: Override schema

追加するJSON file形式:

```json
{
  "schema_version": "package-row-scale-overrides-v0.1",
  "overrides": [
    {
      "layer_index": 10,
      "tensor_suffix": "linear_attn.out_proj.weight",
      "row_index": 3456,
      "scale": 1.02307179310
    },
    {
      "layer_index": 10,
      "tensor_suffix": "mlp.down_proj.weight",
      "row_index": 3456,
      "scale": 1.04165701172
    }
  ]
}
```

Validation rules:

- `schema_version` must match exactly.
- `layer_index`, `row_index`, and `scale` are required.
- `scale` must be finite and positive.
- `tensor_suffix` must be one of a narrow allowlist at first:
  - `linear_attn.out_proj.weight`
  - `mlp.down_proj.weight`
  - optionally later: `self_attn.o_proj.weight`

### T2: Smoke-only matrix row scaling

Add a helper near the package golden-prefix smoke code:

- copy the selected runtime matrix buffer to host
- decode as little-endian f32
- multiply the selected row slice by `scale`
- encode to little-endian f32
- copy back to the same runtime buffer
- synchronize stream

This is intentionally smoke-only and expensive. A layer 10 MLP down matrix is about 192 MiB f32, which is acceptable for targeted validation but not for production runtime.

### T3: Wire linear-attention path first

Apply row overrides only inside `package_linear_attn_mlp_block_sequence_run`:

- after materializing `out_matrix`, apply matching `linear_attn.out_proj.weight` row overrides
- after materializing `down_matrix`, apply matching `mlp.down_proj.weight` row overrides
- include the applied override list in `module_contribution` details or top-level JSON detail

Do not change self-attention path in the first patch unless needed for compilation symmetry.

### T4: CLI argument

Add one optional final positional argument to `package-golden-prefix-smoke`:

```text
[ROW_SCALE_OVERRIDES_JSON]
```

Existing commands remain compatible because the new argument is final and optional.

### T5: Validation matrix

Required:

- CPU p4p46-inproj, `seq16 / 0..12`, `golden_before_each_layer`, no override
- CPU p4p46-inproj, `seq16 / 0..12`, `golden_before_each_layer`, layer 10 two-row override
- CPU p4p46-inproj, `seq16 / 0..12`, `actual_prefix`, no override
- CPU p4p46-inproj, `seq16 / 0..12`, `actual_prefix`, layer 10 two-row override

Pass criteria:

- layer 10 `golden_before_each_layer` max abs should fall materially below `0.875896454`
- layer 10 `actual_prefix` max abs should fall materially below `1.744266510`
- other layers must not regress materially
- MSE and mean abs should not increase enough to suggest overfitting one coordinate

Optional:

- R9700 p4p46-inproj same two runs if CPU improves
- add layer 6 hidden `3456` overrides if layer 10 experiment is positive

## Risks

- The optimal row scale is fit on a tiny activation sample. It may overfit `seq16` token ids.
- Scaling an output row can improve the hot coordinate while hurting other tokens/prompts.
- Copying full matrices host-side is slow and should not become a production path.
- A permanent solution may need quantizer-side row handling, not runtime override.

## Expected Decision

If the smoke-only override improves layer 10 without broad regressions, the next design step is a quantizer-side row compensation experiment for selected sensitive rows. If it only improves the fitted coordinate while hurting aggregate metrics, abandon row scaling and investigate higher-bpp or per-family treatment for `mlp.down_proj` / `linear_attn.out_proj`.

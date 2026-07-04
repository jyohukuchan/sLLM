# Quantizer Row Compensation Plan v0.1

## 前回の要点

- `package-golden-prefix-smoke` にsmoke-only row scale overrideを追加した。
- Layer `10` の `linear_attn.out_proj[3456]` と `mlp.down_proj[3456]` は、row scaleで大きく改善した。
  - CPU `actual_prefix 0..12`: `max_abs_diff` が `1.744266510` から `0.967845917` へ低下。
  - R9700でも同等の結果を確認済み。
- Layer `11` はself-attention層で、単一row scaleでは弱い改善にとどまった。
  - row `3377`: `max_abs_diff` が `0.179061234` から `0.167163849` へ低下。
  - rows `3377 + 3994`: MSEはわずかに改善したが、`max_abs_diff` は `0.179649353` に悪化。
- `analyze-golden-prefix-module-contribution.py` にcomponent scale fitを追加し、layer `10` がlayer `11` より明確にscale-likeであることを確認した。

## 今回の変更点

row compensationをquantizer側へ進める前に、実装形態を分けて考える。

現在のruntime overrideは、materialized f32 matrixの特定rowを実行前に直接スケールしている。これは検証には有効だが、production packageではそのまま使えない。

quantizer/package側の選択肢は3つある。

1. **Manifest row-scale metadata**
   - AQ payloadは現状のまま保持する。
   - manifestにrow scale補正を追加する。
   - loader/materialize/fused-dequantで、該当rowに乗算する。
   - 最も正直で、補正の有無を追跡しやすい。
   - package schemaとruntime loaderの変更が必要。

2. **Pre-scaled quantization**
   - quantizerが対象rowをスケールした値として量子化する。
   - runtime変更なしで補正済みrowを読める。
   - ただし、元weightへの忠実な量子化ではなくなる。
   - codebook/scopeやtensor metricsの意味が曖昧になりやすい。

3. **Hybrid row override tensor**
   - AQ tensorは現状のまま保持する。
   - selected rowsだけ小さな補正tensorとして別payloadに持つ。
   - materialize時にrow差し替え、またはfused dequantでrow patchを適用する。
   - 精度は高いが、実装とpackage表現が複雑になる。

## 次の行動

最初にやるべきなのは、production実装ではなく、manifest row-scale metadataのprototypeである。

理由:

- runtime overrideで効果が確認済みなので、同じ意味論をpackage schemaに移すのが最も検証しやすい。
- pre-scaled quantizationは、weight忠実度と補償効果が混ざり、失敗時の原因切り分けが難しい。
- hybrid row override tensorは有望だが、row scaleで十分かどうかを確認した後でよい。

## Prototype Scope

対象はlayer `10` の2行だけに限定する。

- `model.language_model.layers.10.linear_attn.out_proj.weight`
  - row `3456`
  - scale `1.02307179310`
- `model.language_model.layers.10.mlp.down_proj.weight`
  - row `3456`
  - scale `1.04165701172`

対象外:

- layer `11`
- 自動row探索
- generation/logits評価
- fused dequant kernel対応
- pre-scaled quantization
- row offset/additive correction

## Proposed Schema

Package manifestに任意のtop-level fieldを追加する。

```json
{
  "row_scale_overrides": {
    "schema_version": "row-scale-overrides-v0.1",
    "entries": [
      {
        "tensor_name": "model.language_model.layers.10.linear_attn.out_proj.weight",
        "row_index": 3456,
        "scale": 1.02307179310,
        "source": "golden-prefix-row-dot-sensitivity"
      }
    ]
  }
}
```

Validation rules:

- `tensor_name` must match a quantized matrix tensor in the manifest.
- `row_index < rows`.
- `scale` must be finite and positive.
- Only allow matrix tensors with row-major materialization.
- Reject duplicate `(tensor_name, row_index)`.

## Implementation Steps

### T1: Manifest data model

- Add optional `row_scale_overrides` to the package manifest structs used by loader/runtime.
- Add serde defaults so existing packages continue to load.
- Add validation helper:
  - missing tensor is an error
  - row out of range is an error
  - unsupported tensor encoding is an error

### T2: Loader/materialize path

- Apply row scale immediately after AQ materialization to f32 runtime buffer.
- Reuse the same host-copy implementation currently used by `package-golden-prefix-smoke`.
- Keep this path behind manifest metadata, not CLI-only smoke arguments.

### T3: Quantizer emission path

- Add optional quantizer CLI argument:

```text
--row-scale-overrides-json PATH
```

- The file should use the same logical schema as the smoke override JSON, but require full `tensor_name`.
- `ullm-quant` writes the metadata into the generated package manifest.
- The quantized payload itself remains unchanged.

### T4: Validation matrix

Required:

- Load old p4p46 package without manifest row scale: must remain unchanged.
- Generate or patch a p4p46 package manifest with layer `10` row scale metadata.
- Run `package-golden-prefix-smoke` without CLI row override:
  - CPU `golden_before_each_layer 0..12`
  - CPU `actual_prefix 0..12`
  - R9700 `golden_before_each_layer 0..12`
  - R9700 `actual_prefix 0..12`

Pass criteria:

- Results match the current smoke-only row override within CPU/HIP tolerance.
- Existing packages without `row_scale_overrides` remain byte/load compatible.
- Layer `10` max abs improvement is preserved.

## Risks

- Row scale compensation changes the effective model weights, so it must be treated as a calibrated approximation, not faithful quantization.
- Prompt-local overfitting is still possible. The current evidence is only `seq16` prefix validation.
- Manifest metadata means fused dequant kernels must eventually implement the same row multiplier.
- Too many row overrides would complicate package loading and reduce confidence in the quantization policy itself.

## Decision Gate

Proceed to manifest row-scale metadata only if layer `10` is the initial target and the implementation is kept optional. Do not generalize to layer `11` until a self-attention-specific row-dot trace exists.

## 2026-07-05 Prototype Result

Implemented the first manifest metadata prototype:

- Engine package manifest parsing accepts optional `row_scale_overrides`.
- `materialize_selected_aq4_matrix` applies matching row scales after AQ4 dequantization and before returning the runtime matrix.
- `ullm-quant --row-scale-overrides-json PATH` attaches validated metadata to direct package manifests.
- Existing packages without metadata remain load-compatible; the no-metadata p4p46 package still runs with `row_scale_overrides=none`.

Validation package:

- `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer10.ullm.d`
- Created as a hardlink copy of the p4p46-inproj package with only `manifest.json` changed.

CPU current-binary comparison, no metadata vs manifest metadata:

| mode | max MSE before | max MSE after | max abs before | max abs after |
| --- | ---: | ---: | ---: | ---: |
| `golden_before_each_layer` | `0.000740506879` | `0.000740506879` | `0.875896454` | `0.508314133` |
| `actual_prefix` | `0.004141662294` | `0.004106469453` | `1.744266510` | `0.967845917` |

Layer `10` direct effect:

| mode | layer 10 max abs before | layer 10 max abs after |
| --- | ---: | ---: |
| `golden_before_each_layer` | `0.875896454` | `0.304975510` |
| `actual_prefix` | `1.744266510` | `0.967845917` |

R9700 manifest metadata validation:

| mode | max MSE | max mean abs diff | max abs diff | min cosine similarity |
| --- | ---: | ---: | ---: | ---: |
| `golden_before_each_layer` | `0.000740507114` | `0.020715803` | `0.508314133` | `0.998585695` |
| `actual_prefix` | `0.004106476000` | `0.050080222` | `0.967796326` | `0.992982658` |

Conclusion:

- Manifest metadata reproduces the layer `10` max-abs improvement without using the smoke CLI override.
- The improvement is backend-stable on CPU and R9700.
- Aggregate MSE is still dominated by later layer drift, so layer `11` remains a separate debugging track.

## 2026-07-05 Layer 6 + Layer 10 Probe

Added layer `6`, hidden `3456` row-scale metadata using the existing row-dot sensitivity result:

- `model.language_model.layers.6.linear_attn.out_proj.weight[3456]`, scale `1.032273364777375`
- `model.language_model.layers.6.mlp.down_proj.weight[3456]`, scale `1.036585679248007`

Kept the layer `10`, hidden `3456` metadata from the prototype:

- `model.language_model.layers.10.linear_attn.out_proj.weight[3456]`, scale `1.0230717930961908`
- `model.language_model.layers.10.mlp.down_proj.weight[3456]`, scale `1.0416570117172528`

Validation package:

- `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer10.ullm.d`
- Created as a hardlink copy of the p4p46-inproj package with only `manifest.json` changed.

Summary artifact:

- `benchmarks/results/2026-07-05/engine/package-golden-prefix-manifest-row-scale-layer6-layer10-p4p46-inproj-summary.md`

Comparison against the layer `10`-only metadata package:

| run | backend | layer10-only max abs | layer6+10 max abs |
| --- | --- | ---: | ---: |
| `golden_before_each_layer` | CPU | `0.508314133` | `0.508314133` |
| `actual_prefix` | CPU | `0.967845917` | `0.891334534` |
| `golden_before_each_layer` | R9700 | `0.508314133` | `0.508314133` |
| `actual_prefix` | R9700 | `0.967796326` | `0.891326904` |

Interpretation:

- Layer `6` metadata removes the hidden `3456` actual-prefix drift chain that previously survived into layer `11`.
- The improvement is real but modest; the dominant actual-prefix coordinate moves to hidden `3994`.
- Row-scale metadata is useful for isolated scale-like rows, but a broader policy still needs a selection rule and must handle layer `11` attention-input drift separately.

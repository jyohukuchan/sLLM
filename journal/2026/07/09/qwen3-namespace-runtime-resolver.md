# Qwen3 Namespace Runtime Resolver

## 前回の要点

- SQ8_0とvLLM FP8の同一モデル比較には、Qwen3-14B-FP8のuLLM packageとSQ8_0 artifactが必要。
- Qwen3-14B-FP8 safetensorsは `model.*` namespace、既存uLLM runtimeは `model.language_model.*` namespaceを前提にしていた。
- そのため、package生成前にtensor namespace互換が必要だった。

## 今回の変更点

- `crates/ullm-engine/src/qwen3_names.rs` を追加し、Qwen3の `model.*` / `model.language_model.*` aliasを共通化した。
- package quantized selector、passthrough selector、SQ8_0 artifact selectorにexact lookup優先のfallback aliasを追加した。
- `manifest-all` / `manifest-self-attn` のlayer検出が `model.layers.*` を受けられるようにした。
- unit testで package selector、SQ selector、manifest layer detection、共通alias helperを検証した。

## 次の行動

- Qwen3-14B-FP8 packageをbounded memoryで生成し、`inspect-package` と `manifest-all` 40層検出を確認する。
- そのpackageに対してSQ8_0 artifactを接続し、smoke shapeのuLLM rowを取る。

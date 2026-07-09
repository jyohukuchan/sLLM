# SQ8_0 vs vLLM FP8 Same-Model Readiness Audit

## 前回の要点

- 現在の比較は `benchmarks/results/2026-07-09/sq8-vllm-fp8-comparison/results.jsonl` に集約されている。
- 含まれる行は、
  - uLLM: `Qwen3.5-9B` `SQ8_0` `pp16/tg8/b1`
  - vLLM: `Qwen3-14B-FP8` `pp16/tg8/b1`（smoke）
  - vLLM: `Qwen3-14B-FP8` `pp512/tg128/b1`（representative）
- いずれも結果は取得できており、実行自体は成功している。
- ただし、比較対象モデルは同一ではなく、same-model結論を出す前提が未充足。

## 今回の変更点

- Same-model化監査として、未充足項目を明確化するための観点を新規Markdownで整理。
- Qwen3-14B-FP8側のHF設定を固定情報として明記。
  - `model_type=qwen3`, `architectures=Qwen3ForCausalLM`
  - `hidden_size=5120`, `num_hidden_layers=40`, `num_attention_heads=40`, `num_key_value_heads=8`, `head_dim=128`
  - `torch_dtype=bfloat16`, `quantization_config {fmt=e4m3, weight_block_size=[128,128]}`
  - `HF path`: `/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3-14B-FP8`
- 現時点で見つかる `.ullm.d` は `Qwen3.5-9B` 系のみで、`Qwen3-14B` のuLLM packageは未確認/未作成であることを明文化。
- Runtimeとsafetensorsのテンソル名不一致を同一モデル化の阻害要因として明示。
  - 現在のuLLM runtime前提
    - `QWEN3_EMBED_TOKENS_TENSOR = model.language_model.embed_tokens.weight`
    - layer: `model.language_model.layers.*`
  - Qwen3-14B-FP8 safetensors実体
    - `model.embed_tokens.weight`
    - `model.layers.*`
  - 対応方針は、変換時リネーム（最も直接的）かruntime側のprefix吸収のどちらか。
- Runtime側のprefix吸収を実装した。
  - package quantized tensor selectorは `model.language_model.*` 要求から `model.*` manifestを引ける。
  - passthrough selectorは `model.language_model.embed_tokens.weight` / `model.language_model.norm.weight` から `model.embed_tokens.weight` / `model.norm.weight` を引ける。
  - SQ8_0 artifact selectorは `model.language_model.layers.*` 要求から `model.layers.*` artifact entryを引ける。
  - `manifest-all` と `manifest-self-attn` のlayer検出は `model.layers.*` に対応した。
- dry-runで、同一モデル化に必要なmetadata読み取りはpayloadを全量materializeせず進められる見込みを確認。
  - `tools/build-sq-fp8-w8a16-artifact.py --dry-run`: FP8対象 `281`、passthrough `442`、compact resident estimate `15557220864` bytes
  - 修正前の `ullm-quant --dry-run`: total tensor `723`、supported tensor `280`、passthrough `443`
  - 修正後の `ullm-quant --dry-run`: total tensor `723`、supported tensor `0`、passthrough `723`
- `ullm-quant` の修正後分類では、Qwen系FP8の補助テンソル `*.weight_scale_inv` は `family=other` / `action=passthrough` になる。
- この結果、`Qwen3-14B-FP8` を現在のAQ4 direct package converterで再量子化してuLLM package化する経路は採らない。source matrixが既に `F8_E4M3` で、AQ4 converterはBF16/F16/F32 source matrixを対象にしているため。
- `ullm-quant` に passthrough filter を追加し、Qwen3-14B-FP8 sourceからBF16-onlyの薄い `.ullm.d` package shellを作れるようにした。
  - 実生成物: `/tmp/ullm-qwen3-14b-fp8-bf16-thin.ullm.d`
  - `quantized_tensors=0`, `passthrough_tensors=163`, `codebooks=0`
  - `F8_E4M3` weight本体と `*.weight_scale_inv` は薄いpackageにはコピーしない。
- `ullm-engine` の `manifest-all` / explicit layer kind検出を、薄いpackageの passthrough `self_attn.q_norm.weight` / `self_attn.k_norm.weight` でも動くようにした。
  - 実packageで `layer_count=40`, `self_attention_count=40`, `contiguous_layer_indices=true` を確認。
- 同じQwen3-14B-FP8 sourceからlayer0用SQ8_0 sidecar artifactを作成した。
  - 実生成物: `/tmp/ullm-qwen3-14b-fp8-layer0-sq8-artifact`
  - `fp8_tensor_count=7`, `passthrough_tensor_count=716`
- 薄いpackage + layer0 SQ8_0 artifactで `sq-fp8-token-ids-logits-smoke` が `verified=true` まで到達した。
  - これはsame-modelの最小接続確認であり、まだ40-layer throughput rowではない。
- Same-model rowの必要条件を列挙
  - FP8/SQ8_0 package import（短期はBF16 thin package + SQ8_0 sidecar overlay、長期はnative `.ullm.d` SQ tensor統合）
  - SQ8_0 artifact生成/import（layer0は確認済み、40層は未完了）
  - tensor-name互換の解消
  - 40-layer `manifest-all` rowの整備
  - prompt guard bundleまたは同等のbehavioral guard
  - vLLMと workload shapeの完全一致（少なくとも `pp/tg/b1` と長さ・シード条件）

## 次の行動

- layer0で通った thin package + SQ8_0 sidecar overlay 経路を、40層artifactへ拡張する。
- 生成済みQwen3-14B thin packageを使って `40-layer manifest-all` のuLLM行を追加する。
- tensor-nameの整合（`model.*`/`model.language_model.*`）はruntime側で吸収済み。次は実packageで自動検証する。
- 同条件（`pp16/tg8/b1` を含む）でvLLM smoke/代表bothを再実行し、初めて同一モデルsame-model throughputとして扱う。
- その時点で、比較結論の表記を「same-model throughput conclusion」として更新できるか判定する。

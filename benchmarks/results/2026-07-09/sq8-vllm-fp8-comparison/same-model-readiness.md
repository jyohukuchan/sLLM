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
  - `rope_theta=1000000`
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
  - これはsame-modelの最小接続確認。
- 同じsourceから40層SQ8_0 sidecar artifactを作成した。
  - 実生成物: `/tmp/ullm-qwen3-14b-fp8-full-sq8-artifact`
  - `fp8_tensor_count=281`, `passthrough_tensor_count=442`
  - artifact sizeは約 `14G`
- Qwen3-14B-FP8の40-layer `manifest-all` uLLM行を `results.jsonl` に追加した。
  - preliminary行: `rotary_dim=32`, `rope_base=10000000`
  - config一致行: `rotary_dim=128`, `rope_base=1000000`
  - config一致 smoke: `ullm-r9700-qwen3-14b-fp8-sq8-smoke-pp16-tg8-b1-rope128-theta1e6`
  - config一致 representative: `ullm-r9700-qwen3-14b-fp8-sq8-rep-pp512-tg128-b1-rope128-theta1e6`
- config一致行はどちらも `status=ok` / `verified=true` / `sq_execution_mode=direct_fp8_dequant_matvec`。
- config一致行へself-behavioral prompt-suite smoke guardを添付した。
  - prompt-suite summary:
    `benchmarks/results/2026-07-09/sq8-vllm-fp8-comparison/qwen3-14b-sq8-prompt-suite-smoke-rope128-theta1e6/summary.json`
  - guard bundle:
    `benchmarks/results/2026-07-09/sq8-vllm-fp8-comparison/qwen3-14b-sq8-prompt-suite-smoke-rope128-theta1e6/guard-self-behavioral/guard-bundle-summary.json`
  - `quality.prompt_suite_regression_status=passed`, `acceptance_mode=behavioral`,
    `scope=self_behavioral_prompt_suite_smoke`
  - これは同じsummaryをreference/candidateにした自己比較なので、prompt-suite/guard配線確認であり、外部参照による品質確認ではない。
- Same-model rowの必要条件を列挙
  - FP8/SQ8_0 package import（短期はBF16 thin package + SQ8_0 sidecar overlay、長期はnative `.ullm.d` SQ tensor統合）
  - SQ8_0 artifact生成/import（40層 artifact は完了）
  - tensor-name互換の解消
  - 40-layer `manifest-all` rowの整備（smoke / representative とも完了）
  - prompt guard bundleまたは同等のbehavioral guard（self-behavioral smokeは添付済み、非自己比較またはoutput-health評価は未完了）
  - vLLMと workload shapeの完全一致（`pp16/tg8/b1` と `pp512/tg128/b1` は完了）

## 次の行動

- Qwen3-14B-FP8同一モデル行へ、非自己比較のbehavioral guardまたはoutput-healthを評価するprompt suiteを追加する。
- uLLM行は現在token-id model-loop / final logits込み / `prefill_real_batch=false` / `decode_real_batch=false` なので、vLLM serving benchmarkと同格の最終性能結論にはしない。
- 次の比較昇格には、real-batchまたはserver-style uLLM pathで同じ `pp/tg/b1` を再測定する。

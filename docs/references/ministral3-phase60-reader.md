# Ministral 3 Phase 60 reader

Phase 60のidentity、semantic、GGUF相互運用を再確認するためのread-only資料一覧である。外部実装のsource expressionはcopyせず、
公式metadataと観測可能な数値／layout contractからsLLM側のtyped contractを独立実装する。

## Artifact

- safetensors source: `mistralai/Ministral-3-3B-Instruct-2512-BF16`
  revision `b6d637bef2393152b3da2b2fde72eecdee30557e`
- production GGUF: `mistralai/Ministral-3-3B-Instruct-2512-GGUF`
  revision `eb599d408350ea2bb60452cb86be7c7b2fc28227`
- Mistral product docs: <https://docs.mistral.ai/models/ministral-3-3b-25-12>

## Semantic reference

- Hugging Face Transformers commit `3e9d3e50e71442a3173bdf01cd45ba5833533efe`
  - `configuration_ministral3.py` SHA-256
    `92438408b796088aba2696d62ddaeeb6d2a4c036e3bbacdcab7bcee2fb08a097`
  - `modeling_ministral3.py` SHA-256
    `fecc874d4e21cfb6771f3af13f32e634ae4aec6f3f3145d9e6a6f8b3eb0ac6e5`
  - `modeling_rope_utils.py` SHA-256
    `9a10115e9b7d80a4015324b7da06f67c42166c43bee2959a2b5525753bc1732e`
- YaRN inverse-frequency、split-half rotation、RoPE後Q-only long-position scale、GQA mapping、Direct RMSNorm、SwiGLUの
  stage orderだけを照合する。

## GGUF reference

- llama.cpp `b10453` commit `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`
- canonical spelling `mistral3`、236 text tensor naming、Q/K permutation、YaRN metadata、attention temperature metadataの
  相互運用cross-checkだけに使う。
- production fileはMistral公式GGUFを直接reviewする。sLLM converter由来、またはllama.cppからのimportとは主張しない。

[対応する計画](../plans/archive/2026/08/21-31/phase60-ministral3-3b-production.md)

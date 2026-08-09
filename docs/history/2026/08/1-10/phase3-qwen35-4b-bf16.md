# Phase 3 Qwen3.5-4B BF16 text生成履歴

## 2026-08-04

- 正本の開発順序とGit外`sLLM.md`を照合し、Phase 3の完了点がQwen3.5-4B BF16のtext-only CLI生成とG3までを含むことを再確認した。
- model lock・RMSNorm・G2・P0までの既存案をPhase 3全体の完了点にせず、public runtimeとmodel-bound最小数値経路を作るStage A子計画へ位置付けた。
- Phase 3全体をStage A model-bound最小経路、Stage B model I/O/frontend、Stage C baseline operator、Stage D model graph/state、Stage E CLI/G3へ分割した。
- exact `gfx1030`/`gfx1201`の同一immutable candidateでfull model G3をPASSするまでPhase 3を完了扱いにしないgateを追加した。
- vision、MTP、Qwen3.5-2B/9B、OpenAI API、最適化、quantizationをPhase 3から除外し、正本の開発順序を維持した。
- 固定llama.cpp/vLLMのfull-model reader調査を行い、hybrid layer schedule、full-attention KV、linear recurrent/conv state、tensor分類、tokenizer/CLI、G3 evidence順序を[reader記録](../../../../references/qwen3.5-phase3-full-model-reader.md)へ固定した。main agentが両local checkoutの完全SHAを再確認した。
- 固定cacheと固定vLLM/llama.cppを再照合し、full-attention Q/gateのhead-wise packing、text-only MRoPE、GDN projection・convolution・recurrent update、BF16入力/weight・FP32 accumulationの契約を確定した。
- Phase 3 text-onlyのstateをconvolution BF16 `[3, 8192]`、recurrent F32 `[32, 128, 128]`、full-attention KV FP16 `[4, T, 256]`へ固定し、request-local lifetimeとprefill/decode共通transitionを要求した。
- config EOS 248044とchat-template EOS 248046の差異は、停止集合`[248046, 248044]`、生成tokenだけの判定、stop tokenのvisible output除外、reportへの停止identity保持として解決した。GPU toleranceとG3 goldenは実装後の独立evidence gateとして残した。
- B1 tokenizer依存readerでlocal crate cacheと固定tokenizer metadataを監査し、`tokenizers =0.21.4`のdefault featureを無効化して`onig`だけを使い、任意Jinjaではなくtyped Qwen3.5 text-only rendererを実装する方針を固定した。停止policyのversioned lock/schema/API化と、全依存のroot lock・license・MSRV offline evidenceをB1前提とした。

[対応する計画](../../../../plans/active/2026/08/1-10/phase3-qwen35-4b-bf16.md)

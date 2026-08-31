# DiffusionGemma Phase 59 reader

## 固定source

- artifact: `google/diffusiongemma-26B-A4B-it` revision
  `f7f5b7f5fa82ffc52addd066915886d497f5517b`
- Google JAX semantic source: `google-deepmind/gemma` commit
  `7b785991bd78626c73b317eb43fdbb6c292f7b9c`
- Transformers semantic source: commit `42ca97014c85d71a88ad60d55f08cb9fb4d26e2c`
- Diffusers scheduler source: commit `c1bf18c92c6285334adcaac7e75ef8946a227f49`

## 読取境界

- config／index／model card／generation／processor／scheduler metadataはidentity、shape、生成意味のprimary dataとして使う。
- official JAX、Transformers、Diffusersはmode切替、mask、self-conditioning、sampling順序を照合するno-copy referenceである。
- vLLM等の第三者実装は性能／対応状況の参考に限り、source expression、control flow、kernelをreuseしない。
- full shard payloadを取得していない段階ではLFS SHA-256をlocal payload hashとして扱わない。

## 固定semantic

- causal encoderがpromptと確定済みcanvasをKV cacheへ追加する。
- decoderは256-token canvas内をbidirectionalに扱い、encoder KVへcross-attendする。
- self-conditioningは前stepの分布からembedding expectationを作り、次stepへ渡す。
- generationはuniform vocabulary noiseから開始し、entropy-bounded acceptance、残りのrenoise、temperature schedule、
  confidence＋stability early stopを適用する。
- diffusion生成はAR token-by-token generationと別contractであり、未対応時にARへsilent fallbackしない。

[対応する計画](../plans/archive/2026/08/21-31/phase59-diffusion-gemma-foundation.md)

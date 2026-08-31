# Phase 59: DiffusionGemma foundation

## 2026-08-31: scope固定

- 計画上の未着手`Gemma4 [Diffusion]`を、公式に公開済みの`google/diffusiongemma-26B-A4B-it`として開始した。
- artifact revisionは`f7f5b7f5fa82ffc52addd066915886d497f5517b`、Apache-2.0へ固定した。
- configは3,469 bytes／SHA-256 `13b11d2fe87302cc2332c64eb9eb4ac305d9b8a123ffe9c5cb5b1920fc70c506`、
  indexは104,650 bytes／SHA-256 `6e33e8465d55fe6c7bc0a5453c7a4b341e6467d032c6ded82aaf439f61dac69a`である。
- official BF16は11 shard、1,047 tensor、index payload 51,647,562,456 bytes、shard file合計51,647,701,024 bytesで、
  単一32 GiB GPUに収まらないためfull-model productionとは分離した。
- official descriptionに従い、256-token canvasのuniform-state diffusion、causal incremental prefill、bidirectional denoising、
  self-conditioning、entropy-bounded refinementを新しいsemantic境界として扱う。

## 2026-08-31: identity／header／capacity foundation

- strict configはtext 30 layer、hidden 2,816、5 sliding＋1 full、128 expert／top-8、context 262,144、vision 27 layer、
  canvas 256、special tokenをallocation前に照合する。
- exact index catalogは1,047 tensorをdecoder text 657、vision 355、projector 1、diffusion固有34へ分類し、11 shard assignment、
  total parameters 25,823,778,864、payload 51,647,562,456 bytesを固定する。
- 11 shardの8-byte safetensors length field＋JSON headerだけをbounded range取得した。prefix合計138,568 bytes、payload取得0で、
  全1,047 tensorのBF16 dtype、shape byte数、relative／absolute range、contiguous coverageをindexへ照合した。
  header catalog digestは`fd2cdedb367cd6c9aa52af6463e73baff3df52477b9cc3d61b9c6c4213cdc86f`である。
- shard file合計51,647,701,024 bytes = header 138,568 + payload 51,647,562,456が成立した。capacity admissionは
  file合計を使い、単一32 GiBへfull modelを登録しない。

## 2026-08-31: diffusion semantic／graph boundary

- outer graphはpromptをcausal encoder KVへ追加し、2 canvas目以降は直前の確定256 tokenを一度だけappendする。
  decoderはencoder KVをread-onlyに参照し、current canvas K/Vをconcatしてbidirectional attentionを行うがKV cacheを更新しない。
- uniform random canvas、reverse 48-step temperature、lowest-entropy cumulative acceptance、非accept位置のuniform renoise、
  previous processed-logit self-conditioning、argmax history＋mean entropy adaptive stopを明示RNGのhost oracleへ分離した。
- 公式実装と同じく各refinementで全canvas分のuniform random canvasを先に生成し、accept位置を含む全位置分だけRNGを進める。
  entropy-boundの累積和と差分もFP32で固定し、seed streamとthreshold近傍のaccept maskを一致させた。
- canvas publicationはsampled／renoised working canvasではなく、最終decoder stepのargmax canvasだけをcommitする。
- 1／31／32／33／255／256／257、tie、非finite、range、算術overflowを境界testへ含め、通常AR token loopへのfallbackを持たせない。

## 2026-08-31: GGUF／WebUI boundary

- `diffusion-gemma`をsLLMのwrite-disabled foundation keyとしてparserに追加した。fixed llama.cpp revisionにはmerged architectureがなく、
  open proposalはspelling cross-checkだけに使いupstream canonicalとは主張しない。
- exact config／indexからtyped source mappingをdry-runするが、full payload未取得のためwriter、file type、tensor transform、output hashを
  生成しない。
- model libraryはarchitectureを認識する一方、Apache-2.0、51,647,701,024-byte admission、production loader未対応を理由付きで
  灰色表示し、dynamic registration callbackと通常生成へ渡さない。

## 2026-08-31: validation／integration review

- exact official metadata、1,047-tensor GGUF mapping、11-shard bounded header geometryを固定revisionのlocal oracleへ再照合し、
  3件すべてPASSした。GGUF parser／writer contract 12件、model library 15件もPASSした。
- integration reviewで検出したfull-canvas RNG消費、FP32 entropy cumulative、public writerのwrite-disabled境界、旧Phase 57記述の
  帰属を修正し、変更箇所だけを再レビューして残存correctness blocker 0を確認した。
- workspace test／clippy、format、JSON、Markdown linkの最終結果を完了条件に含める。GPU full-model PASSと性能値は主張しない。

[対応する計画](../../../../plans/archive/2026/08/21-31/phase59-diffusion-gemma-foundation.md)

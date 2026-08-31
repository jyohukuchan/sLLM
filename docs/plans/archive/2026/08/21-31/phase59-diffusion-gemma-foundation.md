# Phase 59: DiffusionGemma foundation

> 状態: 完了（foundation）
> 作成日: 2026-08-31

## 目的

計画済みのGemma 4 Diffusionを、公式DiffusionGemma 26B-A4B instruction checkpoint、discrete diffusion生成、
causal encoder／bidirectional decoder、self-conditioning、block refinementのtyped contractとして追加する。
既存Gemma 4 MoEのreuse可能範囲とdiffusion固有境界を分離し、model-free operatorやmetadataだけをfull-model対応へ読み替えない。

## 固定対象

- artifact: `google/diffusiongemma-26B-A4B-it` revision
  `f7f5b7f5fa82ffc52addd066915886d497f5517b`、Apache-2.0。
- official semantic implementation: `google-deepmind/gemma` commit
  `7b785991bd78626c73b317eb43fdbb6c292f7b9c`。
- Transformers reference: commit `42ca97014c85d71a88ad60d55f08cb9fb4d26e2c`。Diffusers reference: commit
  `c1bf18c92c6285334adcaac7e75ef8946a227f49`。いずれもsemantic照合だけに使い、sourceをcopyしない。
- backbone: Gemma 4 26B-A4B、30 layer、hidden 2,816、128 expert／top-8、5 sliding＋1 full schedule、vocab 262,144、
  context 262,144、canvas 256、vision 27 layer／soft token 280。
- generation default: uniform-state diffusion、maximum 48 denoising step、temperature 0.8→0.4、entropy bound 0.1、
  confidence threshold 0.005、stability threshold 1。scheduler metadataはblock length 32を保持する。
- official BF16は11 shard、1,047 tensor、index payload 51,647,562,456 bytes、shard file合計51,647,701,024 bytesで、
  KV／workspace前に単一32 GiB GPUへ収まらない。foundationではfull-model resident／generationをPASSとして主張しない。

## 受入条件

1. revision、license、support files、11 shardのsize／LFS identity、index、processor、tokenizer、generation、scheduler設定を固定する。
2. strict typed config／indexでencoder／decoder mode、Gemma 4 topology、canvas、special token、vision、tensor family、capacityを
   allocation前にfail-closeする。full payload未取得時はHub identityとlocal SHA-256を混同しない。
3. causal incremental-prefillとbidirectional canvas attention、encoder KV cross-attention、self-conditioning、canvas commitの
   container-neutral graph／state transitionを固定し、AR token loopへ暗黙fallbackしない。
4. uniform random initialization、temperature schedule、entropy-bounded acceptance、renoising、confidence＋stability early stop、
   multi-canvas boundaryを独立oracleへ固定する。1／31／32／33／255／256／257と非finite／overflowを含める。
5. 既存Gemma 4 MoE／attention／vision contractとの共有点を明示し、bidirectional／cross-attention／self-conditioningに必要な
   新規semantic opだけをversionedに追加する。
6. canonical GGUF metadata／tensor mapping／write-disabled dry-runを追加する。full source bytesなしにderived GGUFや
   tensor byte一致を主張しない。
7. model library／WebUIはarchitecture、Apache-2.0、必要resident bytes、production loader未対応を灰色表示し、通常生成へ登録しない。
8. model-free refinement operatorを追加する場合はhost oracleとexact `gfx1030`／`gfx1201`でHIP-only、fallback 0、
   nonfinite fail-close、cleanup 0を確認する。
9. affected test／clippy／format、integration review、model lock、GGUF、runtime、provenance、compatibility、main plan、historyを同期する。

## 後段production条件

- 対象GPUへ収まるreviewed artifact、またはmulti-GPU／partial residencyを明示した別計画を必要とする。
- fixed promptのreference canvas／token列、seeded RNG、early-stop、stop、CLI／API／WebUI、metrics、cancel／recovery、
  load／unload、clean shutdownを同じartifactとexact GPUで確認する。
- 通常AR Chat Completionとのstreaming差、canvas内token訂正、usage／finish reason／partial publicationをAPI profileへ明示する。

## 非対象

- 51.6 GB BF16を単一32 GiB GPUへ無理にresident化すること。
- community quantizationの未検証production採用、CPU full-model fallback、Transformers／Diffusers／vLLM sourceのcopy／port。
- multimodal full execution、256K actual、multi-GPU、performance claim。

[対応する履歴](../../../../../history/2026/08/21-31/phase59-diffusion-gemma-foundation.md)

# Phase 60: Ministral 3 3B text production

> 状態: 完了（公式GGUFのRoPE pairingを修正し、参照生成・実GPU・resident性能を再確認）
> 作成日: 2026-08-31

## 目的

計画の`[その他]`枠へ、単一32 GiB GPUで実用できる公開済みの小型dense architectureとしてMinistral 3 3Bを追加する。
既存Qwen／Gemmaのdense実行資産を再利用しつつ、Ministral固有のYaRN RoPE、Tekken tokenizer／chat、text／vision構成境界を
typed contractへ固定し、text-onlyのCLI／API／WebUI production経路まで統合する。

## 固定対象

- artifact: [`mistralai/Ministral-3-3B-Instruct-2512-BF16`](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16)
  revision `b6d637bef2393152b3da2b2fde72eecdee30557e`、Apache-2.0、public／ungated。
- production GGUF: [`mistralai/Ministral-3-3B-Instruct-2512-GGUF`](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF)
  revision `eb599d408350ea2bb60452cb86be7c7b2fc28227`の`Ministral-3-3B-Instruct-2512-BF16.gguf`、
  6,866,745,504 bytes、LFS SHA-256 `17ef932bea952e007f9dad63151da5699132ec513d1033d618df7382e24aa3ee`。
  text-only productionはこの公式GGUFを直接reviewし、mmprojを要求しない。
- official product contract: [Mistral Docs Ministral 3 3B](https://docs.mistral.ai/models/ministral-3-3b-25-12)。
- model: outer `mistral3`、text `ministral3`、26 layer、hidden 3,072、FFN 9,216、32 Q head／8 KV head、head dim 128、
  vocab 131,072、tied embedding、YaRN 16x、original context 16,384、advertised context 262,144。
- vision metadata: Pixtral、24 layer、hidden 1,024、16 head、head dim 64、patch 14、image size 1,540、projector biasなし。
  Phase 60のproduction実行はtext-onlyとし、vision weightをtext graphへ暗黙接続しない。
- official BF16 index: 2 shard、458 tensor、index上4,251,743,232 parameter、physical header上3,849,090,048 element、
  payload 7,698,180,096 bytes。差はtied embeddingの論理出力側二重計上として保持する。
  `consolidated.safetensors`は同一checkpointの別配置であり、2-shard sourceと同時に常駐させない。

## 固定した受入条件

1. revision、license、support files、2 shardのsize／LFS SHA-256、index、tokenizer、chat template、generation設定を固定する。
2. strict typed config／index／bounded header catalogでtext、vision、projectorを分類し、dtype、shape、range、parameter、byte、
   shard assignment、capacityをallocation前にfail-closeする。
3. GQA dense SwiGLU／RMSNorm／tied embeddingとYaRN 16xのposition scalingを独立host oracleへ固定し、plain RoPEへfallbackしない。
4. canonical GGUF architecture／metadata／tensor mappingを固定し、full source bytesを取得するまではdry-runだけとする。
5. 公式BF16 GGUF full bytes取得後はLFS SHA-256、header／catalog、resident byte admission、text-only load、greedy fixed promptの
   official reference token列を固定する。公式GGUFをsLLM derived outputとは主張しない。
6. 通常CLI、OpenAI API buffered／SSE、dynamic model library、default WebUI、metrics、load／unload、cancel／recovery、
   clean shutdownを同じartifact identityで確認する。
7. model libraryはreviewed identityだけを登録し、unsupported architecture、容量不足、vision request、identity driftを理由付きで拒否する。
8. exact local GPU証拠はCPU fallback 0、selected GPU dispatch、numerical oracle、cleanup 0を必須とし、host testをGPU PASSへ数えない。
9. affected test／clippy／format、integration review、model lock、GGUF、runtime、provenance、compatibility、main plan、historyを同期する。

## 境界値

- token／context: 0／1、16,383／16,384／16,385、262,143／262,144／262,145。
- tensor／head: layer 0／25／26、Q head 31／32、KV head 7／8、head dim 127／128／129、非aligned token 3／17／33。
- artifact: 1／2／3 shard、457／458／459 tensor、index parameter／physical elementの区別、payload／header／file byteの両側、
  duplicate／unknown／overflow／nonfinite。

## 非対象

- Phase 60内のimage入力、Pixtral vision forward、OCR／document Q&A、tool-use品質保証、262K full-context性能保証。
- community quantizationの未検証採用、`consolidated.safetensors`と2-shard sourceの混在、公式GGUFをderived扱いすること、
  CPU full-model fallback。
- Llama 4、Mistral Small／Large／MoE、multi-GPU、性能優位の事前主張。

## 2026-08-31完了時点

- 公式GGUFのstrict identity、236 text weightのresident plan、Tekken frontend、499-node／105-alias graph、
  26層FP16 KV executor、通常CLI、OpenAI buffered／SSE、dynamic model library、WebUI alias経路まで実装した。
- terminal BF16 logitsを任意取得する診断経路と固定llama.cpp F32 logits比較を追加した。position 0では近い一方、position 1以降で
  KLDが`0.14`以上へ増えたことから、公式GGUFでQ/Kが既にhead permutation済みなのにsplit-half RoPEを再適用した誤りへ特定した。
  ABI v1のsplit-halfは互換用に残し、production graphはadjacent-pair v2へ移した。追加workspaceやdtype変更はない。
- 修正後のexact `gfx1030`／`gfx1201`通常CLIは、ともにraw `Hello`を固定llama.cppと同じ
  `[1307,1278,4304,1033]`（` of the world!`）として生成した。以前に反復誤生成したchat caseも`4`を返した。
  両targetともHIP-only、fallbackなし、394／394 dispatchである。
- common-prefix 3行のtop-1は両targetで固定llama.cppと全一致した。gfx1030のKLDは
  `0.000234788／0.000271453／0.000188792`、gfx1201 incrementalは
  `0.000234788／0.000317793／0.000206496`である。full-prefillは両targetでBF16 logitsがbit一致し、
  gfx1201のM=1 decodeだけprovider／targetの演算順によりbit差を持つがtop-1は維持する。
- 513-token prefill＋8 decode、2 warmup＋5 measured、resident weight常駐、request allocation／logit readback除外の中央値は、
  gfx1030が`138.29 tok/s`／`18.34 tok/s`、gfx1201が`1351.30 tok/s`／`19.29 tok/s`だった。
  gfx1201 baseline強制は`11.66 tok/s`／`6.76 tok/s`へ悪化し、品質も一様に改善しないため採用しない。
  gfx1030 prefillと両target decodeの性能残差は後続最適化候補であり、Phase 60の品質完了とは分離する。
- host C ABIとexact `gfx1030`／`gfx1201`の実HIP C ABIは旧split-half v1と新adjacent-pair v2の双方を独立数値oracleでPASSし、
  full-model evidenceもresident／request cleanup 0を確認した。vision forward、OCR／document Q&A、262K性能保証は引き続き非対象とする。

[対応する履歴](../../../../../history/2026/08/21-31/phase60-ministral3-3b-production.md)

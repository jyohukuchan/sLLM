# Phase 60: Ministral 3 3B text production

> 状態: 一時停止（text production経路は実装済み、参照生成との品質一致は未達）
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

## 2026-08-31停止時点

- 公式GGUFのstrict identity、236 text weightのresident plan、Tekken frontend、499-node／105-alias graph、
  26層FP16 KV executor、通常CLI、OpenAI buffered／SSE、dynamic model library、WebUI alias経路まで実装した。
- exact `gfx1030`と`gfx1201`で同じ公式GGUFを実行し、HIP-only、fallbackなし、request state／workspace cleanup 0を確認した。
  `Hello`のgreedy 4-token出力は両targetともtoken `[1307,1278,3950,1044]`（` of the day,`）で一致した。
- 同じtokenizer入力を固定llama.cppへ与えた参照列は`[1307,1278,4304,1033]`（` of the world!`）である。
  最初の2 tokenは一致し3 token目からずれる。tokenizer／chat templateはtoken列一致、行列積baseline化でも結果不変、
  両GPUで誤列が一致するため、target固有Attention差ではなく共有model execution数値境界の未解決問題として残す。
- production品質の受入条件5、6、8はこの不一致により未達である。architectureを対応済みへ昇格せず、計画はarchiveしない。
  再開時はhead dim 128のFP16 KV Attention逐次oracleに加え、layer境界とterminal logitsのF32/BF16差を先に比較する。
- ユーザーの停止指示により、この区切り以降のarchitecture追加を自動開始しない。

[対応する履歴](../../../../../history/2026/08/21-31/phase60-ministral3-3b-production.md)

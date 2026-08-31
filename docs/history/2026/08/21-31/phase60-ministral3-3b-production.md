# Phase 60: Ministral 3 3B text production

## 2026-08-31: target／scope固定

- 計画の`[その他]`枠は、Apache-2.0、public／ungated、単一32 GiBへ収まる公式
  `mistralai/Ministral-3-3B-Instruct-2512-BF16`を先に扱う。
- revisionは`b6d637bef2393152b3da2b2fde72eecdee30557e`へ固定した。公式APIの観測時点で26,156 downloadsであり、
  Mistral公式docsはedge／local向け3B text＋vision modelとして掲載している。
- indexは45,577 bytes／SHA-256 `7829dcf0040e34f1172b401563fcbb27cc3c5a0244ef01e6af18b7a64d63a81e`、
  configは1,579 bytes／SHA-256 `c89d1a0b4f237d2892ce911d1fe03e9e5a4834579f7149ebc715a4c3fa564214`である。
- official indexは2 shard、458 tensor、4,251,743,232 parameter、payload 7,698,180,096 bytesを宣言する。
  bounded headerのphysical elementは3,849,090,048であり、差402,653,184はtied embeddingを論理出力側へ二重計上した値として
  混同せず保持する。
  32 GiBへ収まるため、identity-only foundationではなくtext-only production統合をPhase 60の完了条件とした。
- vision topologyとweightはartifact identityへ含めるが、text graphとimage inputへ暗黙接続せず、vision productionを別条件へ分離する。
- production inputはMistral公式GGUF repository revision `eb599d408350ea2bb60452cb86be7c7b2fc28227`のBF16 text GGUFへ固定した。
  file sizeは6,866,745,504 bytes、LFS SHA-256は`17ef932bea952e007f9dad63151da5699132ec513d1033d618df7382e24aa3ee`である。
  Mistral公式artifactとして直接reviewし、sLLM converter由来とは扱わない。

## 2026-08-31: bounded header identity

- shard file sizeは4,967,581,832／2,730,659,224 bytes、header prefixは47,240／13,720 bytesで、payload取得0の
  bounded rangeだけを取得した。header tensor 353／105件はindex 458件とmissing／extra／assignment差0で一致した。
- 全458 tensorはBF16で、physical element 3,849,090,048、payload 7,698,180,096 bytes、gap／overlap 0を確認した。
  textは236 tensor、visionは218、projectorは4である。
- header catalog digestは`81f2cea3da0101288d73ae936e8afd7ec5b2760af6e51c846d093b6036e07828`へ固定した。

## 2026-08-31: YaRN semantic boundary

- 128次元split-half RoPEのYaRN inverse-frequency blendをfactor 16、original context 16,384、theta 1,000,000、
  beta fast／slow 32／1としてmodel-free FP32 oracleへ固定した。correction rampは20..37である。
- 通常YaRNとは別に、RoPE後Qだけへ`1 + 0.1 × ln(1 + floor(position / 16384))`を適用する。
  16,383では1、16,384では約1.0693147、262,143では約1.2772589となり、K／Vやplain RoPEへ共有しない。
- 32 Q head／8 KV headの4:1 GQA mappingと、0／1／16,383／16,384／32,768／262,143／262,144、
  head dim 127／128、nonfiniteをhost testへ含めた。GPU PASSはまだ主張しない。

## 2026-08-31: 公式BF16 GGUF実体検証

- `Ministral-3-3B-Instruct-2512-BF16.gguf`の全6,866,745,504 bytesを取得し、full-file SHA-256
  `17ef932bea952e007f9dad63151da5699132ec513d1033d618df7382e24aa3ee`が公式LFS identityと一致した。
- strict `VerifiedGguf`でcanonical architecture `mistral3`、alignment 32、236 text tensor、BF16 matrix 183件、
  F32 norm 53件、tied `output.weight`不在を実ファイルから確認した。raw metadata digestは
  `sha256:7e16085724a92d35c80e29982ff663860fc95b6a054fcbf57b0f28f881cd5f0e`、raw tensor catalog digestは
  `sha256:f40ed89f4535224c30c8a0c03a7a167435adcb06e909af07f33fd66f25dee95a`である。
- 458 source tensorのtext／vision／projector分類、Q／K permutation、norm変換を含むcanonical dry-run mapping digestは
  `b4c4061c4f9932c51fef2a8b01d1ae96a99b4c701ae1ece4869b852c46333da9`へ固定した。dry-runはsource payloadの変換・
  書込みを実行したという証拠には数えない。
- 同一のopen file descriptor上でGGUF parse後にfull-file hashを行うproduction admission入口を追加し、symlink／file swapを
  別identityとして受け入れない。実体fixture検証は2/2 PASSで、CPU fallbackやGPU PASSの主張は含まない。

## 2026-08-31: text resident weight plan

- 公式GGUFの236 text tensorを26-layer dense graphの236 consumerへ一対一で対応させた。Q／Kは公式GGUF内で既にcanonical
  permutation済みのため再変換せず、vision／projector／unknown／duplicate／missingをproduction planから拒否する。
- 公式GGUFの53 normはF32だが、全値が有限かつF32下位16 bit 0で、source BF16値を完全保持していることを実体から確認した。
  現行RMSNorm ABIはBF16 scaleを受け取るため、load時に無損失でBF16へ戻す。全236 resident weightはBF16となり、常駐量は
  6,858,012,672 bytesである。F32を丸めて品質を変えるfallbackは許可しない。
- packed planは16 MiB以下のbounded chunk、tied embedding、公式repo／revision／full-file SHA、変換後のvirtual norm rangeを固定する。
  通常focused test 33件とcore strict clippyに加え、公式実体からnorm変換と全planを構築するfixture testがPASSした。
- `ministral3-official-gguf-model-lock-v1`を追加し、restricted-JCS fingerprint
  `sha256:8a8701bb8e7838bbc87575bea3339a1884d83a0bcd4cc226f6c83e4c3f70759a`へsource／production／architecture／frontend
  identityを結合した。

## 2026-08-31: official GGUF frontend

- production frontendは公式GGUF内の標準`tokenizer.ggml.*` metadataからTekken tokenizerを厳格再構成する。
  sLLM独自extensionは要求せず、token 131,072件、merge 269,443件、token type、special ID、BPE、Split regex、
  ByteLevel、BOS post-processor／decoderを固定し、unknown key、型、長さ、merge形式のdriftを拒否する。
- source repositoryのchat templateは11,912 bytes／SHA-256
  `0701cfbdc2b7d44fdbad104dff604faee4b0543e8247624568777fe465746f9b`、production GGUF埋込み版は
  7,753 bytes／SHA-256 `d28d7df94f0fd7e8d0075a22c473333d6e7dd2bc4c36c83e8b975300a0fb94bc`として別identityに固定した。
  text-only rendererは埋込み版の既定system promptとcompact system-user／history token列へ一致する。
- frontend focused test 6件、strict clippy、公式6,866,745,504-byte GGUFからのtokenizer再構成と2 fixture encodeを確認する
  ignored実体testがPASSした。これはhost frontend証拠でありGPU PASSには数えない。
- resident weight sourceをbackend-neutral bounded upload helperへ接続した。F32 norm由来のvirtual BF16 payloadもmatrixのGGUF direct
  payloadも同一load plan digest、tensor descriptor、16 MiB以下のchunk境界でuploadできる。

## 2026-08-31: Ministral YaRN public HIP operator

- plain RoPEへfallbackしないversioned public C ABIとして、BF16 split-half YaRN prepare／execute／releaseを追加した。
  Qは`[M,32,128]`、Kは`[M,8,128]`、theta 1,000,000、factor 16、original context 16,384、
  beta fast／slow 32／1を固定し、RoPE後のQだけへposition-dependent scaleを適用する。
- host fake-runtime ABIではnull、overflow、alignment、shape、version、unsupported parameterをfail-closeし、M=1／3／17と
  最大終端262,144を含めた。host public runtimeと専用C ABI testはPASSした。
- canonical Radeon Pro V620 exact `gfx1030`向けROCm 7.14.0 target binaryで専用GPU testを再実行し、独立FP32 oracle、
  HIP-only dispatch、fallback falseをPASSした。この証拠はYaRN operatorだけを対象とし、Ministral full model、性能、
  `gfx1201`や別software tupleへ一般化しない。

## 2026-08-31: Core graph／dispatch integration

- flat Matmul出力とhead-wise state APIを混同しないため、各layerへQ／K／Vのrank-3 zero-copy `Reshape`と、
  attention出力からO projection入力へのrank-2 zero-copy `View`を追加した。105 aliasは元buffer、dtype、encoding、
  contiguous element／byte範囲を共有し、独自allocationやcopyを持たない。prefillのtied logitsは最終rowだけを非zero-offset
  `View`で射影する。graphは499 nodeとなり、T=3／17／33を含む
  focused testとstrict clippyをPASSした。
- Core owned-executionへMinistral専用YaRN submissionを追加した。Q `[T,32,128]`、K `[T,8,128]`、positions `[T]`と
  2 outputのsession、access、range、alias、contextをadapter callback前にfail-closeし、plain Rotary semanticへ変換しない。
- HIP bridgeは5 bindingを同一contextのpublic YaRN descriptorへlowerし、contiguous position mode、active-operation lifetime、
  dispatch evidence、completion timingを保持する。Rust stub runtimeにもfail-closed entrypointを追加し、`cargo test`でlinkされない
  build-only漏れを解消した。bridge host test 23件とstrict clippyがPASSした。
- generic reviewed-model registryへ`Ministral3Dense`を追加した。公式GGUF lockはalias／fingerprint／repo／revision／chatを公開する一方、
  safetensors `VerifiedCache`やsLLM-derived GGUF source fingerprintへ読み替えず、誤った検証入口を理由付きで拒否する。

## 2026-08-31: text production統合と実GPU到達点

- 236 BF16 resident weight、26層のrequest-local FP16 KV、YaRN、causal GQA、terminal-row logits／device Argmaxを
  `Ministral3ExecutionRequest`へ接続した。prefill／decodeは各遷移で1 tokenを返し、KV長、alias、dispatch target、
  completion、poisoningをfail-closeする。
- 通常CLI、direct `sllm-server`、OpenAI Completions／Chat Completions buffered／SSE、metrics、dynamic model library、
  WebUIから同じalias `ministral3-3b-instruct-2512`を選択する経路を追加した。direct起動の既定contextは、32 GiBで
  262,144-token FP16 KVを先行確保しないよう公式original contextの16,384とし、明示指定だけ262,144まで許可する。
- causal attention public ABIは従来のQ head 8／16、head dim 256／512に加えてQ head 32、head dim 128をreviewed shapeへ追加した。
  FP16 KVの暗黙score scaleはkernel実行で`1/sqrt(head_dim)`を使う。dispatch evidenceも整数分母を持てない128／512では
  `scale_denominator=0`とexact binary32 scale bitsを公開し、従来の固定16という誤記録を解消した。
- model libraryのfolder変更／rescanは直列化し、複数aliasのunload途中失敗ごとに実際のregistered setとcatalog rowを更新する。
  retry時に既にunload済みaliasを再度削除しない。新しい選択pathはrefresh成功後だけ永続化し、failure injection testで旧path保持と
  retry recoveryを確認した。
- exact V620 `gfx1030` release buildで公式6,866,745,504-byte GGUFを常駐させ、buffered／SSE生成を完走した。
  shutdown auditはHIP-only、fallback false、394 submission／394 kernel dispatch、final request-state／workspace 0である。
  exact R9700 `gfx1201`でも同じartifact、FP16 KV、1-token prompt／4-token decodeを完走し、HIP-only、fallback false、
  394／394 dispatch、cleanup 0を確認した。

## 2026-08-31: 参照生成不一致と停止判断

- 公式GGUF内tokenizer、chat templateと固定llama.cppのtoken列は一致した。公式GGUFの
  `tokenizer.ggml.scores`はI32 zero配列のためllama.cppが直接拒否するので、参照実行だけはreflink copy上で配列element typeを
  F32へ読み替えた。zero payload bytesは変更しておらず、repo artifactやsLLM入力には使っていない。
- BOSを無効化したraw `Hello` 1-token入力では、sLLM `gfx1030`／`gfx1201`がともに
  `[1307,1278,3950,1044]`（` of the day,`）、固定llama.cppが
  `[1307,1278,4304,1033]`（` of the world!`）となった。prefill tokenと最初のdecode tokenは一致し、2回目decodeの出力から
  分岐する。chatの`What is 2+2? Answer briefly.`も参照の`4`に対して反復的な誤生成となった。
- `SLLM_MATMUL_FORCE_BASELINE=1`で列は変わらず、両RDNA targetでも同じ列だった。従ってtokenizer／renderer、target固有provider、
  optimized matmul単独を原因から除外し、共有するBF16 activation／RoPE／Attention／terminal BF16 logits境界のどこで参照から
  累積差が生じるかは未解決として保持する。特にBF16 logitsがclose top-1を反転させる可能性はあるが、現時点では推定であり断定しない。
- production経路は実行可能でも数値品質の受入条件を満たさないため、Phase 60を完了／対応済みへ昇格せずactive planのまま一時停止する。
  ユーザーが戻って停止を指示したため、次architectureの自動追加もここで終了した。

[対応する計画](../../../../plans/active/2026/08/21-31/phase60-ministral3-3b-production.md)

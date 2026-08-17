# Phase 20: GGUF unification

> 状態: completed（P20-A0〜A6、2026-08-17）
> 作成日: 2026-08-17

## 目的

safetensorsと量子化sidecarを変換・開発入力へ移し、現在sLLMが実行できるBF16、FP8、NVFP4、MXFP4 modelを、
tokenizer、vocabulary、chat template、model metadata、tensor recipeとともに単一GGUFへ収容する。公開runtimeはGGUFを
正本として読み、source artifactと変換結果をmodel lockで再現可能に固定する。

## Scope

- converter、GGUF reader/runtime、standardまたはversioned sLLM extension metadata、tensor encoding、derived model lock、
  safetensors移行と互換性closeoutを扱う。
- Qwen3.5 dense BF16/MTP/vision、Gemma 4 BF16/NVFP4 mixed、Qwen3.5 MoE MXFP4 text-onlyの既存container-neutral
  descriptorをGGUFへlowerし、同じsemantic identityを維持する。
- request batching、chunked prefill、永続KV、追加model family、追加KV形式、multi-GPU、性能backlogは含めない。
- llama.cppからの直接reuseは許容するが、import前にexact source、license、local path、変更分類をprovenance recordへ追加する。
  P20-A0はsourceをinspectしただけでcodeをimportしていない。

## 固定した受入条件

1. 公開artifactはGGUF v3、little-endian、32-byte alignmentの単一fileとする。split GGUF、外部tokenizer、量子化sidecar、
   recipe sidecarを通常runtimeの必須入力にしない。
2. 標準GGUFで表現できるarchitecture、metadata、tensor typeを優先する。初期architecture名は`qwen35`、
   `qwen35moe`、`gemma4`、標準tensor typeはBF16=30、MXFP4=39、NVFP4=40とする。
3. NVFP4とMXFP4のsource value/scale planeは、数値を再量子化せず標準GGML blockへlossless repackする。logical shape、
   nibble order、block scale、outer/input scale、expert axisを変換前後で照合する。
4. pinned GGUFにはFP8 weightの標準tensor typeがない。FP8をBF16、F16、Q8_0へ変換してGGUF対応と呼ばず、A1で
   standard parserが構造を読めるversioned sLLM extensionを固定してから実装する。A0ではcustom numeric type IDを割り当てない。
5. GGUF loaderはcontainer固有rangeを、既存の`QuantizedTensorEncoding`、tensor role、scale plane、mixed recipe、MoE config、
   verified load plan、tokenizer/chat contractへlowerする。GGUF化を理由に別model semantic identityを作らない。
6. derived model lockは全source fingerprint、converter repository/commit、引数、effective config、environment、output GGUFの
   size/SHA-256、metadata digest、tensor catalog digestを含める。runtimeはGGUF本体を検証後も同じopen descriptorから読む。
7. unknown architecture/type/extension version、duplicate metadata/tensor、overflow、overlap、misalignment、truncation、recipeの
   missing/extra/ambiguous bindingをallocation前に拒否する。dtype変換やsource containerへのsilent fallbackは行わない。
8. host-only A0はsource/format decision、handoff inventory、machine-readable contract、schema/test、計画・履歴の整合で完了する。
   converter、reader、GPU、full-model evidenceはA0のPASSに使わず後続で取得する。

## 実装・検証順序

### P20-A0: source/format lockとhandoff inventory（完了）

- llama.cpp `b10453` commit `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`のGGUF header、reader/writer、constants、
  quant block、Qwen3.5/Gemma 4 conversionをfile SHA-256付きで固定した。
- GGUF v3/container、architecture、標準tensor typeとFP8 extension境界を
  [GGUF format contract](../../../../../formats/gguf.md)と
  [A0 contract manifest](../../../../../../ci/matrix/phase20-gguf-a0-v1.json)へ固定した。
- Phase 17/16F/19から渡されたdense、MTP、vision、NVFP4 mixed、MXFP4 MoE、frontend/model-lock情報をinventory化した。
- schema validationとsemantic contract testはhost-onlyで実行し、local `reference/` checkoutがないCIでもtracked manifestを
  検証できるようにした。明示的なlocal source verificationだけがcheckoutを要求する。

### P20-A1: bounded GGUF readerとextension schema

- RustでGGUF v3 header、typed metadata、tensor table、alignment/rangeをbounded/fail-closedに解析する。mmapやpayload uploadより先に
  duplicate、overflow、overlap、truncation、unknown typeを拒否する。
- tokenizer/chatとarchitecture metadataを既存frontend/configへlowerする。FP8とmixed recipeのversioned sLLM metadata、
  tensor/scale relation、canonical digestを固定し、unknown extensionを拒否する。
- synthetic tiny GGUFで0/1 tensor、非整列dimension、alignment境界前後、duplicate/overlap/truncationをhost testする。

### P20-A2: BF16 converter、writer、derived lock

- fixed Qwen3.5-4B BF16を最初のvertical sliceとし、source lockからsingle GGUFをdeterministically生成する。
- tensor payload、logical shape、tokenizer IDs、chat template、metadataをsourceと照合し、同一input/configのbyte-identical出力、
  derived lock、改竄拒否を確認する。

### P20-A3: FP8/NVFP4/MXFP4 converter

- Gemma 4 NVFP4 mixed、Qwen3.5 MoE MXFP4、FP8 weight/scaleを既存container-neutral descriptorから変換する。
- NVFP4/MXFP4はstandard blockへのlossless repackを独立decoderと照合する。FP8はA1 extensionを使い、dequantized代用品を作らない。
- MTP/vision/known-unconsumed componentとstatic KV recipeを単一container内でscope付きに保持する。

### P20-A4: runtime loader integration

- normal CLI/serverのmodel指定をGGUFへ接続し、同じverified descriptor、execution graph、provider selectionを使う。
- safetensors/sidecar importerは変換・開発用に残すが、公開runtimeの通常loadから外す。GGUF failure時に旧containerへfallbackしない。

### P20-A5: cross-format fidelityとservice evidence

- Qwen dense BF16、Gemma BF16/NVFP4 mixed、Qwen MoE MXFP4についてsource importerとGGUF loaderのdescriptor、tensor bytes、
  graph、token/logit、CLI/OpenAI lifecycleを比較する。
- canonical V620/R9700でfallbackなし、accounted memory、cleanup 0を確認する。container変更だけのための性能倍率は要求しないが、
  load time、resident/peak、TTFT/TPOTを記録する。

### P20-A6: migration、互換性、closeout

- model-lock、runtime、format、CLI、compatibility、provenance文書を同期し、公開手順からsafetensors/sidecar必須操作を除く。
- standard GGUF toolで構造/standard fieldが読めることと、sLLM extension非対応readerが破損せず明示的に非対応を示すことを確認する。
- 一回のintegration reviewと変更findingのfocused re-review後にplanをarchiveする。

## P20-A0完了証拠

- `python3 -m pytest -q ci/tests/test_phase20_gguf_a0.py`
- `python3 ci/tools/validate_markdown_links.py`
- `git diff --check`
- A0はhost-only contractであり、model、GGUF binary、raw traceをrepositoryへ追加していない。

## P20-A1〜A6完了証拠

- `ded2264035b8138da581773e42f37d11e3693fe1`にbounded reader、deterministic writer、FP8 extension、4 converter、
  derived lock、runtime lowering、GGUF-only CLI/serverを固定した。llama.cpp codeのimportはなく、A0で固定したsourceを参照した。
- final BF16/FP8/NVFP4/MXFP4 GGUFは順に9,343,583,840 / 5,779,142,624 / 9,337,229,760 /
  24,617,123,424 byte。SHA-256は`50582d6c...9ca3`、`1a9db28b...74b5`、`4e0410c6...2fb5`、
  `44022302...1fce`で、全derived lockがconverter commit `ded22640...3fe1`を保持する。
- Qwen BF16の独立2回変換はbyte-identical。standard NVFP4/MXFP4 repackは独立decoderとexact byte/valueで一致し、
  FP8はI8 carrier + versioned bindingとして元payloadを保持した。
- canonical R9700 `gfx1201`とV620 `gfx1030`でQwen BF16、Gemma NVFP4 mixed、Qwen MoE MXFP4のGGUF/source top-1が一致。
  R9700ではQwen FP8もPASSした。全caseはHIP-only、fallbackなし、cleanup 0。R9700 MoE serverのmodels/chat/shutdownもPASSした。
- host回帰はcore 173、GGUF contract 11、CLI 24、server 27 testをfailed/skipped 0でPASSした。4 final GGUFの
  `verify-model --gguf ... --derived-lock ...`もPASSし、旧公開引数はparserで拒否する。
- model、GGUF binary、raw trace、生成artifactはrepositoryへ追加していない。詳細値は対応履歴を正とする。

## 停止・再計画条件

- standard typeの実byte layoutと既存descriptorがlosslessに対応しない場合は、再量子化せずA1 format decisionへ戻る。
- FP8 extensionがstandard parserによる安全なskip/inspectionとfaithful runtime decodeを両立できない場合、custom type IDを即席で
  割り当てず、upstream標準化またはversioned carrier方式を比較する。
- semantic identityを維持できないmetadata不足が見つかった場合、loader実装を続けずA0 inventoryとderived lock契約を更新する。
- 同じwork unitの2回reject、review時間が実装時間超、1時間以上の機能進捗停止、検証/docs 30%超、見積り1.5倍超、
  acceptance変更時は追加探索を止めて同じwork unitを再計画する。

[対応する履歴](../../../../../history/2026/08/11-20/phase20-gguf-unification.md)

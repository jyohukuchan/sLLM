# Phase 40: token selection・grammar・structured generation

## 目的とauthority

Phase 40は、既存のprofile-v1生成結果を維持したまま、backend非依存のtoken selection、bounded grammar、structured
generation、確率出力、複数choice stateを一つのgeneration contractへ統合する。ユーザー指示によりPhase 37/38のMI300X
実機完了を開始gateにせず、hostと手元のV620/R9700で進める。MI300Xではgfx942 compile/selector routeまで準備するが、VMを
再確保するまで実機correctness/performance PASSを主張しない。

Authorityは`sLLM.md`、`AGENTS.md`、[main plan](../../../../main-plan.md)、[Phase 37以降roadmap](../../../../active/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md)
の順に従う。本planはroadmapのPhase 40を詳細化するtask-local planであり、Phase 41以降のscopeを前倒ししない。

完了状態（2026-08-21）: A0〜A3、B1〜B3、C1〜C2、D1〜D2とPhase Eを完了した。V620/R9700 selector contract matrix、
Qwen/Gemma sampled structured generation、最終workspace/integration reviewはPASSした。
gfx942 feature-pinned compile/routeはPASS、MI300X実機correctness/performanceはVM再確保後の別runへdeferredする。llama.cppからの
新規コードreuseは行っていないため、provenance recordの追加・変更はない。

## 固定baselineと仕様pin

- sLLM baseline: Phase 39 closeout commit `e7bcdcca09d0dff2deec3b32d80f8b75c03ee167`。
- llama.cpp semantic/reference source: local `reference/llama.cpp` commit
  `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70` (`b10453`)。既存profile-v1互換oracleは
  `f5919bf458ef190468b5c329bb293f8a54a1e69c`の既存provenance recordを維持する。
- OpenAI Chat Completions request/response schema: 2026-08-21取得のofficial
  `POST /v1/chat/completions` OpenAPIと[Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs)
  をpinする。対象fieldは`logit_bias`、`logprobs`、`top_logprobs`、`n`、`response_format`だけとし、Responses APIやtool
  protocolへ拡張しない。
- `top_logprobs`は`logprobs=true`時だけ許可し、範囲をofficial contractどおり`0..=20`とする。`logit_bias`はtoken IDごとの
  `-100..=100`を受付け、sampling前logitへ加算する。
- `response_format`は`text`、`json_object`、`json_schema`を受ける。`json_schema`は本Phaseで明示したsubsetだけを実装し、
  未対応keywordや上限超過をrequest受付時に4xxで拒否する。生成後validationだけをstructured output対応とは数えない。

## 互換性と所有境界

- 現行`SamplingParametersV1`と`ProfileSamplerV1`のgreedy、temperature、top-p、presence/frequency penalty、seedは
  `legacy-v1` adapterとして残し、追加field省略時のtoken列、tie順、RNG消費、stop semanticsを変えない。
- `temperature=0`かつ追加sampler、logit bias、grammar、logprobsが全て無効なら、既存device Argmaxを使いfull-vocabulary
  D2HとRNG消費を行わない。いずれかが有効なら`requires_logits`/selector capabilityを明示的に要求し、Argmaxでmaskを迂回しない。
- Phase 40はsampler state、grammar state、choice seed/RNG、choice stop stateとchoiceごとのgeneration/KV ownerを所有する。
  Phase 41のprefix cache、checkpoint、context shift、assistant prefill、speculation providerは変更しない。
- Phase 42はPhase 40のchoice/logprob resultを別endpointへ写像するだけとし、Phase 43のtools/function schema、tool choice、
  tool executionは本Phaseから除外する。
- llama.cppのsource expressionをcopy/adapt/portする場合は、実装前にexact blob/hashを固定し、source header、
  `THIRD_PARTY_NOTICES.md`、license、pending import commitを追加する。他engineはtechnical factsだけを参照する。

## 受入条件

1. 追加fieldを全て省略した既存requestは、固定fixtureで生成token列、finish reason、usage、stream framingがbaselineと一致する。
2. sampler chainの各stageはfixed logits、stable token-ID tie、non-aligned vocabulary、境界値、固定seedでhost oracleに一致する。
   NaN、無効なinfinity、空vocabulary、all-masked、ゼロ/非有限mass、checked arithmetic失敗をsilent fallbackせずerrorにする。
3. selected token logprobとtop-logprobsは、実際のsamplingに使ったpost-bias/post-penalty/post-grammar-mask/post-filter分布から計算し、
   token textとraw bytesを同じtoken IDへ対応させる。
4. grammarはprefixごとに受理可能tokenだけを残し、選択tokenをacceptした後もUTF-8 partial stateを保持する。無効GBNF、left recursion、
   unsupported JSON Schema、state explosion、token piece上限、全token禁止は明示errorにする。
5. `n>1`はchoiceごとにseed/RNG、sampler history、grammar、stop matcher、decoder、KV/generation ownerを分離する。非stream responseと
   SSEはchoice indexを安定して付与し、choice間のcancel/errorで別choiceの状態を共有・破損しない。
6. GPU prepared selectorはexact targetとcapabilityを記録し、fallback count 0、selected record以外の不要なfull-vocabulary D2H 0を
   evidenceに含める。CPU oracleとのtoken/logprob一致を示し、GPU unavailable、timeout、crash、zero selectionをPASS扱いしない。
7. gfx1030/gfx1201は手元実機でcorrectnessを検証する。gfx942はcompile/route testまでを本closeoutに含められるが、MI300X実機
   PASSと性能採否はdeferred evidenceとして残す。

## 固定上限

- API: `n=1..=8`、`top_logprobs=0..=20`、logit bias entry 4096、serialized grammar/schema 64 KiB。
- Grammar compile: rule 1024、rule name 128 bytes、alternative/rule 256、AST nesting 32、bounded repetition 4096、JSON enum 256、
  JSON Schema property合計1024。generic `json_object`はdepth 1、containerあたり最大4 members/items（repeat 0..3）、string/number 64、
  whitespace 16へ別boundedとする。
- Grammar runtime: stack depth 128、active state 65,536、token piece 128 bytes、token trie nodeはchecked
  `min(vocab_size * 129, 33,554,432)`。上限はrequest全体に適用し、超過時は縮退せず拒否する。
- History: frequency/presenceは既存全履歴semanticsを維持する。repeat/DRY用windowとsequence breakerは別のbounded configとし、
  request token上限を越える保持を行わない。

## Work units

### P40-A0: semantic lock・fixture・provenance

- current sampling/API/generation/HIP ABIのbaseline fixtureを追加し、既存requestのtoken列とno-readback greedyを固定する。
- ordered chain v1の順序を`raw logits -> finite validation -> logit bias -> legacy/additional penalties -> grammar/EOS mask ->
  temperature -> candidate filters -> terminal selector -> logprob metadata`へ固定する。
- llama.cppから直接reuseするunitを選ぶ場合だけ、`src/llama-sampler.cpp`、`src/llama-grammar.*`、
  `common/json-schema-to-grammar.*`と対応testのblob/hashをprovenanceへ追加する。

### P40-A1: backend-neutral sampler chain

- `SamplingParametersV1`を壊さず、ownedでclone/reset可能な`SamplerChainConfigV1`、candidate row、mask、history、
  deterministic random stream、selection resultを追加する。
- legacy adapterを最初に実装し、greedy、temperature、top-p、presence/frequency penaltyのexact compatibilityをfixtureで確認する。
- top-k、min-p、typical、ignore-EOS、repeat penalty、dynamic/adaptive temperatureを独立nodeとして追加し、disabled defaultとstage orderを
  versioned schemaへ固定する。

### P40-A2: stateful/terminal sampler

- XTCはchoice/node専用RNG stream、DRYはbounded historyとtokenized sequence breaker、Mirostat v1/v2はmutable `mu`を持つterminal
  selectorとして実装する。Mirostat同士と通常categorical terminal selectorの不正な併用は受付時に拒否する。
- vocab 1/3/7、flat logits、極小温度、`k=0/1/>vocab`、probability `0/1`、history `0/1/max`、seed `0/u64::MAX`、
  token ID境界をtestする。

### P40-A3: logit bias・probability metadata

- sparse biasをtoken IDでvalidate/sortし、duplicateの扱いをschemaで一意にする。mask済みtokenをtop-logprobsへ戻さない。
- selected tokenと上位`0..=20`件についてtoken ID、decoded token text、raw bytes、logprobを生成し、nonstream/SSEで共有できる
  transport-independent resultへ格納する。
- greedyでもlogprobs要求時は最終分布を計算し、既存Argmax-only shortcutを使わない。

### P40-B1: tokenizer byte seam・bounded token trie

- model-aware raw token piece APIをfrontendへ追加し、ByteLevel、ByteFallback、special token、unused reserved vocabulary rowを区別する。
- token IDからraw bytesへのimmutable tableとbounded trieをmodel load時に構築し、grammar request間でread-only共有する。
- 単一token decodeのreplacement characterへ依存せず、partial UTF-8をgrammar stateで継続する。

### P40-B2: GBNF compiler・runtime

- parser、AST validation、left-recursion検出、bounded automaton/stack state、UTF-8 decoder、token-trie mask、accept/reset/cloneを実装する。
- empty production、range/not-range/any、escaped byte、alternation、group、bounded repetitionをsupport表へ固定する。無制限状態増加を伴う
  constructはcompile時に拒否する。
- grammar maskはsampling chainのcandidate filter前に適用し、all-maskedをselection errorとしてrequestへ返す。

### P40-B3: JSON・JSON Schema lowering

- `json_object`用のbounded JSON grammarと、`json_schema`からGBNF/automatonへのlowererを実装する。
- subsetはobject、array、string、number、integer、boolean、null、enum、const、required、`additionalProperties:false`、
  `anyOf`、local `$defs`/`$ref`に限定する。schema key orderをoutput orderへ反映する。
- `allOf`、`not`、conditional/dependent keyword、remote `$ref`、regex/pattern、format、numeric/string/array size constraint、recursive
  `$ref`は初期subsetから除外し、無視せず4xxで拒否する。後からsupportを増やす場合はfixtureと上限を同時追加する。

### P40-C1: generation choice state

- single-choice loopを`GenerationChoiceV1`へ抽出し、choice index、derived seed、sampler、grammar、stop decoder、usage、finish reason、
  backend request ownerをchoice単位にする。
- `n=1`は既存loopへのzero-semantic-delta adapterとし、`n=2..=8`はchoiceを独立requestとしてschedulerへ投入する。
  seed派生はversioned counter/hashで固定し、choice 0の既存seed列を維持する。
- 途中cancel、queue full、一choice error、異なるfinish token、stop跨ぎ、UTF-8跨ぎをtestする。

### P40-C2: OpenAI Chat Completions adapter

- strict request schemaへ`logit_bias`、`logprobs`、`top_logprobs`、`response_format`、sampler extension、`n`を追加し、既存の
  unsupported-field表を更新する。
- nonstream choicesとstream chunkのindex/logprobs/finish reasonをofficial schemaへ写像する。streamではchoiceごとのdelta順を保ち、
  terminal chunkと`[DONE]`を一度だけ送る。
- `json_object`ではpromptにJSON指示がない場合を受付時に拒否し、schema/grammar compile errorをgeneration開始前に返す。

### P40-D1: prepared selector contract・HIP ABI

- backend-neutral `PreparedTokenSelectorV1`へimmutable bounded inputと固定size `SelectionOutputV1`を定義する。CPU referenceとHIP
  implementationを別capabilityとし、GPU requestのsilent CPU fallbackを禁止する。
- 既存Argmax ABIを変更せず、`version=1`、`struct_size`、reserved-zero validationを持つadditive HIP sampler ABI、bindings、bridge、
  CMake/build.rs trackingを追加する。
- first targetは`M=1` BF16 terminal row、sparse bias、history aggregate、valid-token bitset、temperature/filter、counter RNG、selected
  recordとする。決定的reductionを使い、順序非決定atomicへ依存しない。

### P40-D2: model adapter・selected-only D2H

- Qwen/Gemma terminal projectionからprepared selectorを呼び、transaction completion後にselected recordだけをreadbackする。
- legacy greedyは既存Argmaxをcontrolとして残し、GPU sampler有効時のfull logits readback countを0へ固定する。
- grammar compile/token trie traversalはRustに残し、deviceへはbounded valid bitsetだけを渡す。all-zero/invalid maskはlaunch前後で
  fail closedにする。

### P40-E: verification・integration review・closeout

- Host: core/frontend/serverのunit・contract・integration、全workspace `cargo test --workspace --all-targets`、fmt、clippy
  `-D warnings`、markdown/link consistencyを実行する。
- Compile: HIP off + gfx1030/gfx1201/gfx942 feature-pinned build、fake C ABI negative/accounting、unknown target/zero capabilityを確認する。
- GPU: V620/R9700でvocabulary `1,3,17,255,256,257,248320`、odd history/bias/mask、NaN/Inf/all-mask、temp 0/positive、
  fixed seed反復、CPU oracle token/logprob、exact kernel/target、fallback 0、full-vocab D2H 0を確認する。
- Model: sampled generation、grammar、JSON subset、logprobs、`n=1/2/8`をQwen/Gemmaの利用可能targetで検証し、性能は同一artifact、
  同一prompt/token budget、warmup、独立process反復で比較する。
- 一回のintegration reviewでcorrectness/security blockerを解消し、planをarchive、history/API/architecture/main-plan/provenanceを同期し、
  Phase全体を一つの最小commitへ整理してpushする。

## 実行順と停止条件

`A0 -> A1 -> A2/A3 -> B1 -> B2 -> B3 -> C1 -> C2 -> D1 -> D2 -> E`を基本順とする。A2/A3、grammar
fixture、API fixture、HIP fake ABIは依存が分かれる時だけ並列化する。以下の場合は同じwork unitの追加実装を止めてreplanする。

- compatibility fixtureが二回連続で拒否され、旧token列を維持できない。
- grammar verification/docsがwork unitの30%を超え、support subsetの縮小で閉じられる。
- GPU selectorのfull-vocabulary D2H、CPU fallback、非決定RNGを除去できない。
- V620/R9700のcorrectnessが二回連続でCPU oracleに一致しない。
- Phase 41以降のKV/session/tool ownershipを変更しないと進められない。

MI300X実機が無いこと自体はhost/手元GPU work unitのblockerにしない。gfx942実機だけが未完なら、exact未検証targetとして明記し、
Phase 37/38またはMI300X再検証runへ証跡を引き渡す。

## Closeout checklist

- [x] legacy-v1 token列とArgmax no-readbackを維持
- [x] ordered sampler chainと全追加samplerを実装
- [x] post-mask logprobsとOpenAI wireを実装
- [x] bounded GBNF/token trie/UTF-8 stateを実装
- [x] 明示JSON Schema subsetとunsupported rejectionを実装
- [x] independent `n=1..=8` choice stateを実装
- [x] HIP prepared selectorとselected-only D2Hを実装
- [x] V620 `gfx1030` selector correctnessを記録
- [x] R9700 `gfx1201` selector correctnessを記録
- [x] gfx942 compile-only statusを記録（MI300X real correctness/performanceはdeferred）
- [x] provenance、API/architecture、main plan、historyを同期（新規llama.cpp reuseなし）
- [x] integration reviewを完了
- [x] commit、pushを完了（このarchive planを含む公開commitで充足）

## Verification record

| lane | 現在の証跡 | status |
| --- | --- | --- |
| Host contracts | `cargo test --workspace --all-targets`、`cargo clippy --workspace --all-targets -- -D warnings`、fmt、Python API fixture、markdown link、`git diff --check` | PASS |
| HIP host/ABI | additive TokenSelect C ABI、Rust bridge、selected-record contract、negative validation、host CTest 4/4 | PASS |
| V620 `gfx1030` | UUID `GPU-76a08c...`、vocabulary `1,3,17,255,256,257,248320`×counter `0,1`、odd mask/additive、固定seed反復、CPU token exact/logprob tolerance `.005`、all-masked/nonfinite、fallback 0、selected record D2H 16 bytes/full-vocabulary D2H 0 | PASS（selector contract scope） |
| R9700 `gfx1201` | UUID `GPU-a8e9...`、V620と同一matrix、CPU token exact/logprob tolerance `.005`、fallback 0、selected record D2H 16 bytes/full-vocabulary D2H 0 | PASS（selector contract scope） |
| gfx942 / MI300X | feature-pinned gfx942 compile/route | compile PASS、real deferred |
| Qwen/Gemma model | V620 `gfx1030`でQwen BF16とGemma mixed NVFP4のtemperature 1.0、fixed seed、selected logprobs、JSON Schema maskを実行。Qwenはstructured `n=2`とchoice `n=8`、Gemmaはvalid JSON completionを確認し、全request HIP-only/fallback 0、shutdown cleanup 0 | PASS |
| Integration / release | reviewで発見したcategorical累積順とshared grammar rule returnの2 correctness blockerを修正し、focused re-reviewをPASS | PASS |

V620/R9700 selector matrixのtracked summaryは
[phase40-token-selector-gpu-summary-v1.json](../../../../../../ci/matrix/phase40-token-selector-gpu-summary-v1.json)を正とする。

[history](../../../../../history/2026/08/21-31/phase40-token-selection-grammar-structured-generation.md) /
[main plan](../../../../main-plan.md)

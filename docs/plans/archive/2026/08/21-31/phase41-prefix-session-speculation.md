# Phase 41: prefix/KV・session state・speculation

## 目的とauthority

Phase 41は、Phase 40のtoken selectionとchoice stateを維持したまま、cross-request prefix reuse、context shift、
session checkpoint、assistant prefill、external draft/ngram speculationを一つのidentity-safeなstate contractへ統合する。
Phase 37/38のMI300X性能laneを開始gateにせず、host、gfx1030/gfx1201、gfx942 compile routeで進める。MI300X実機PASSは
VM再確保後へdeferredし、本Phaseの完了条件へ含めない。

Authorityは`sLLM.md`、`AGENTS.md`、[main plan](../../../../main-plan.md)、
[Phase 37以降roadmap](../../../../active/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md)、本planの順とする。本PhaseはLMCache/RadixAttention、
dynamic model lifecycle、Responses/Completions wire、LoRA、multi-model routingを前倒ししない。

完了状態（2026-08-22）: A0〜A2、B1〜B2、C1〜C2、Dを完了した。Qwen/Gemmaのprefix fork、全state-plane
checkpoint、context shift、assistant prefill、model-neutral speculationをproduction contractへ接続し、V620/gfx1030と
R9700/gfx1201の実GPU state matrixをPASSした。gfx942はwave64 feature-pinned compile-only PASSで、MI300X実機は
VM再確保後へdeferredする。production checkpointはstateless prompt境界に限定し、mid-generation resumeをclaimしない。

## 固定baselineと所有境界

- baselineはPhase 40 closeout commit `f27479bc87cebc1df402759ddcfa85a76e8af505`。
- prefix/session identityはmodel-lock fingerprint、derived artifact/recipe、adapter identity、renderer/template digest、exact token列、
  KV encoding/layout、model target semantics、context policy versionを含む。path、alias、cache directoryだけをidentityにしない。
- Phase 41はprefix entry、state image、context position policy、assistant prefix、draft proposal/verification/accountingを所有する。
  Phase 39のtransport replay、Phase 40のsampler/RNG/grammar/choice state、Phase 42以降のwire adapterはそれぞれ既存ownerを維持する。
- prefix ownerは公開済みstateだけを保持する。in-flight transition、workspace、completion、queue、prepared plan、token selector outputを
  request間で共有しない。cache hit後のrequest mutationは必ずCOWまたは独立stateで行う。
- VMM KVは完全pageをread-only共有し、部分tail pageは最初のappend前にcopy-on-writeする。linear-attention/GDN stateは
  active slotを含むdevice-side cloneとする。contiguous-resident KVはpage共有をclaimせず、同一device上のbounded copyでforkする。
- checkpointは全対応KV encodingのvalue/scale/outer-scale plane、linear/GDN state、token history、conversation、position stateを
  versioned raw imageとして保存する。FP16へのdequantize/requantize、別encoding fallback、再prefillをrestore成功として扱わない。
- llama.cpp sourceをcopy/adapt/portする場合だけ、実装前にexact blob/hash、source header、notice、license、pending import commitを追加する。
  technical behaviorだけを参照する場合はno-copyを記録する。

## 固定上限とsecurity

- prefix cacheは起動時明示opt-in。entry `1..=256`、総logical token `1..=1,048,576`、総resident byte上限を必須とし、
  LRUはlease中entryをevictしない。default disabledで既存memory behaviorを維持する。
- checkpoint schemaはlittle-endian `sllm-session-checkpoint-v1`、header 4 KiB以下、section `1..=4096`、section/total lengthは
  checked arithmetic、単一checkpoint最大64 GiB、conversation最大16 MiB、token history最大1,048,576 tokenとする。
- checkpoint directoryは0700、fileは0600。symlink、non-regular file、hard-link count不一致、owner不一致、world/group writable、
  path traversalを拒否する。同一directoryの新規temporary fileへwrite/fsyncし、rename後directory fsyncする。
- header、section table、各section payload、全file digestを検証後にだけstateをpublishする。truncated/extra/duplicate/overlap、未知必須section、
  version/identity/encoding/layout/target mismatch、checksum不一致、quota超過をfail closedにする。
- conversation、token IDs、KV bytes、checkpoint pathをmetrics、normal log、props/slotsへ出さない。API credentialや環境変数は保存しない。
- ngram indexはrequest-local、最大history 1,048,576、ngram `1..=16`、draft width `1..=8`。external draft widthも`1..=8`、
  target verifyは1 block `2..=9` rowとし、proposal/accept counterはchecked u64とする。

## 受入条件

1. prefix keyの一要素でも異なる場合はmissまたは明示rejectになり、異model/adapter/template/KV/target間のsilent reuseがない。
2. exact/partial/longest hit、non-aligned token/page tail、LRU boundary、lease中eviction、concurrent reader、cancel/errorを検証する。
   reused requestのtarget token、visible output、stop、sampler/grammar stateは同じ入力をfresh prefillした結果と一致する。
3. VMM hitは共有physical page数とCOW page数をauditし、cache ownerのbytesをrequestへ重複accountしない。contiguous pathはcopy byte数を
   明示し、page-shareとして報告しない。fork後のappend/rewind/cancelがcache ownerや別readerを変更しない。
4. context shiftは保持prefix/recent範囲、absolute/logical position、RoPE/mRoPE、attention maskをversioned policyへ固定する。
   `63/64/65`、`127/128/129`、capacity境界、keep 0/1/max、shift反復、stop/UTF-8境界をtestし、未対応model/policyを拒否する。
5. checkpoint round tripはFP16、dynamic/static FP8、NVFP4のKV plane、linear/GDN active state、token history、conversation、identityを
   byte/checksum exactで復元する。atomic replace、quota、corruption、truncation、wrong model/adapter/template/KV/target/versionを検証する。
6. assistant prefillはrenderer/tokenizer後の共通generation inputとして型付けし、prefill bytes/tokenをvisible outputへ再公開しない。
   empty/nonempty、Unicode/byte fallback、stop prefix、grammar、`n=1/2/8`でchoice stateを分離し、既存request省略時のtoken列を維持する。
7. MTP、external draft、ngramは同じmodel-neutral propose/verify/accept accountingを使う。draftは公開target stateを更新せず、reject時は
   targetのaccepted prefixとreplacementだけをpublishする。disabled時と、全reject/partial/all-accept時のtarget token列、sampler RNG、
   grammar/stop、usageを通常逐次生成へ一致させる。
8. GPU state copy/share/import/exportを変更したtargetではexact HIP dispatch、numerical/state oracle、fallback 0、cleanup 0を確認する。
   gfx1030/gfx1201のaffected matrixを実機で行い、gfx942はfeature-pinned compile/routeまでを本Phaseに含める。

## Work units

### P41-A0: identity・state image・fixture lock

- `PrefixStateIdentityV1`、`ContextPositionPolicyV1`、`SessionStateHeaderV1`をbackend-neutral coreへ追加する。
- Phase 40 legacy generation fixture、Qwen/Gemma fresh-prefill output、MTP target-only token列をbaselineとして固定する。
- model/template/adapter/token/KV/target digestのcanonical byte encodingとrejection matrixをtestする。

### P41-A1: prefix index・lease・eviction

- exact token digestだけでなくtoken trie/radix相当のbounded longest-prefix lookupを実装する。entryはimmutable publication後だけvisibleにする。
- lease、reader count、LRU generation、logical/resident accounting、admission/evictionをcheckedにし、active readerをevictしない。
- `PrefixCacheBackendV1`をopaque state fork/publish/removeとして定義し、frontend/serverがdevice pointerやpage tableを所有しないようにする。

### P41-A2: opaque KV/state forkとCOW

- additive C ABIとRust bindingsへquiescent state fork、KV page-share/COW metadata、linear-state cloneを追加する。既存append/view/release ABIを変えない。
- VMM page ownerをreference countedにし、child VAへread-only mapする。append範囲のshared pageだけprivate allocationへD2D copyしてwriteableにする。
- contiguous-resident pathは同device D2D cloneを行う。encodingごとのvalue/scale/outer planeとpublished length/generationを一つのtransactionでforkする。
- Qwen/Gemma request factoryへprefix forkを接続し、workspace/queue/prepared cacheはfresh request ownerへ残す。

### P41-B1: context shift

- generation admissionに`disabled`と`keep-prefix-recent-v1`を追加し、容量到達前だけshiftを開始する。
- retained token列から新しいopaque request stateをtransactionalに構築し、成功後だけownerを交換する。位置/attention/RoPE policyをmodel adapterで検証する。
- shift中cancel/errorはold published ownerを保持し、中間stateを破棄する。sampler/grammar/stop historyの保持範囲を明示する。

### P41-B2: checkpoint export/import・atomic storage

- quiescent KV/linear stateのbounded chunk export/import ABIを追加し、全planeとactive slotをencoding-native bytesで扱う。
- canonical header/section table、SHA-256、atomic writer、strict reader、quota managerをRust coreへ実装する。
- Qwen/Gemma requestからconversation/token/state imageをsaveし、同じresident identityへrestoreするfactoryを追加する。
- CLI/server startup optionは明示指定時だけcheckpointをloadする。不存在を新規会話へsilent fallbackせず、指定checkpoint errorをstartup failureにする。

### P41-C1: assistant prefill

- `GenerationInputV1`を互換維持したまま、rendered promptとassistant prefixを区別する共通prepared inputへlowerする。
- Qwen templateはgeneration markerとの境界、Gemma raw-textは明示suffix境界を固定し、Phase 42/43 adapterが同じ型を再利用できるようにする。
- usage、stop matcher、grammar、decoderはassistant prefixをprompt stateとして初期化し、visible completionへ含めない。

### P41-C2: model-neutral speculation provider

- bounded `DraftProviderV1`、proposal block、target verification、publication decision、accountingをcore/frontendへ追加する。
- 既存Qwen MTPをこのprovider adapterへ移し、MTP固有hidden stateとtarget block kernelはmodel adapter内へ残す。
- ngram providerはcommitted token historyの最長suffix matchからdeterministic proposalを作る。external draft providerは独立executor/state/RNGを持ち、
  target model identityとtokenizer vocabulary compatibilityを開始時に検証する。
- target samplerを唯一のvisible token/RNG ownerにし、rejected draftやdraft RNGをusage/visible outputへ混ぜない。

### P41-D: production integration・verification・closeout

- Qwen/Gemma production owner、CLI/server config、audit/metricsへprefix hit/miss/share/COW/copy、checkpoint、shift、draft counterをbounded labelなし数値で接続する。
- host unit/contract/API、native host C ABI、HIP off、gfx1030/gfx1201/gfx942 compile、V620/R9700 state matrixを実行する。
- Qwen/Gemmaでfresh対hit/restore/shift、ngram、利用可能なexternal draft/MTPを比較し、fallback、cleanup、process残留を確認する。
- 一回のintegration reviewでcorrectness/security blockerを解消し、planをarchive、history/main-plan/runtime/model-lock/API/compatibilityを同期する。
  Phase全体を一つの最小commitへ整理してpushする。

## 実行順と停止条件

`A0 -> A1 -> A2 -> B1/B2 -> C1 -> C2 -> D`を基本順とする。host identity/storage、assistant fixture、ngram fixtureは
所有ファイルが分かれる場合だけ並列化する。次の場合は同じwork unitの追加実装を止めてreplanする。

- page ownershipまたはcheckpoint restoreが異identity/異encodingへsilent reuseする。
- state export/importがFP16変換、full prompt replay、CPU fallbackでしか成立しない。
- prefix hit、context shift、speculationが逐次target token/RNG semanticsを二回連続で維持できない。
- verification/docsが30%を超え、support上限またはwire surface縮小でPhaseの本来のstate機能を保てる。
- Phase 42以降のendpoint、Phase 45のdynamic lifecycle、LMCache/RadixAttentionを実装しないと進められない。

MI300X実機不在は本Phaseのhost/RDNA/compile workをblockしない。gfx942実機だけが未完ならexact未検証としてPhase 37/38または
次回MI300X sessionへ証跡を引き渡す。

## Closeout checklist

- [x] identity/key/rejection fixtureを固定
- [x] bounded longest-prefix cache、lease、LRUを実装
- [x] VMM read-only share/COWとcontiguous D2D forkを実装
- [x] context shift policyを実装
- [x] 全KV encoding・linear/GDN checkpoint round tripを実装
- [x] assistant prefillを共通generation inputへ実装
- [x] MTP/external/ngram共通speculation contractを実装
- [x] host/native/API testをPASS
- [x] V620/R9700 affected GPU matrixとgfx942 compile routeを記録
- [x] integration review、docs/history/provenance同期を完了
- [x] commit、pushを完了

[main plan](../../../../main-plan.md) /
[roadmap](../../../../active/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md)

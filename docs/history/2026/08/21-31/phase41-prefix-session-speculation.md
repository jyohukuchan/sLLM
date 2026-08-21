# Phase 41 prefix・session state・speculation

## 目的と到達点

Phase 41は、Phase 40のtoken selection/choice semanticsを維持したまま、cross-request prefix reuse、context shift、
session checkpoint、assistant prefill、model-neutral speculationを一つのidentity-safe state contractへ統合した。
2026-08-22にhost/API/native、V620 `gfx1030`、R9700 `gfx1201`、gfx942 feature-pinned compile routeをPASSし、
Phase 41を完了した。MI300X real runはVM削除後のためdeferredであり、gfx942実機PASSを主張しない。

llama.cppは機能比較と意味論の参照に用いたが、Phase 41ではsourceを直接copy/adapt/portしていない。新規provenance
blob、notice、license、pending import commitはない。

## 実装内容

- model-lock、derived artifact/plan、adapter、renderer/tokenizer、exact tokens、KV encoding/layout、target、context policyを
  canonical identityへ含めるbounded longest-prefix cacheを追加した。entry/reader/LRU/counterはcheckedで、lease中entryを
  evictionせず、公開済みprompt stateだけを保持する。
- Qwen/Gemmaのopaque KV/linear stateへadditive C ABIのfork、bounded chunk export/import、same-device D2D、VMM page share/COWを
  追加した。QwenはKVとGDN/linear全plane、Gemmaはfull attention 8層と実topologyのsliding attention 40層を扱う。
  post-COW queryでrequest accountingを更新し、shared physical bytesをcache quotaへ一度だけ計上する。
- `keep-prefix-recent-v1`はlogical/absolute positionと保持範囲をversioned policyへ固定した。Qwen/Gemma production executorは
  容量到達前にretained tokensをexplicit RoPE/attention positionでfresh ownerへprefillし、成功後だけowner/historyを交換する。
  反復shift、63/64/65、127/128/129、capacity境界を固定した。Qwen GDNはcompact logical sequenceから再計算する。
- little-endian `sllm-session-checkpoint-v1`、SHA-256、全section digest、64 GiB/file上限、0700 directory、0600 file、
  owner/mode/hardlink/symlink/no-follow検査、quota、temporary write/fsync/rename/directory fsyncを実装した。Qwen/Gemmaの
  encoding-native全KV plane、linear/GDN state、token history、conversation、position/generation stateをcross-session fresh ownerへ
  transactionalに復元する。
- production checkpointはstateless prompt checkpointに限定した。起動時にstrict loadを一度行い、requestのfull token列が
  checkpoint token historyをprefixとして持ち、suffixがnon-emptyの場合だけ継続する。saveはfresh prompt prefill後かつ最初の
  visible delta前にatomic replaceする。load/save同時指定、prefix/context/draft併用、Qwen MoE/FP8/multimodalはfail closedである。
  mid-generation resume、暗黙のbackend-global会話、client間state共有は実装・claimしない。
- sampler/RNGとgrammarのversioned bounded snapshot/restoreをcoreへ追加し、seed、token、grammar payloadをDebug/logへ出さない。
  assistant prefillはrenderer/tokenizer後のprepared inputへlowerし、decoder/stopをprimeするがvisible completionへ再公開しない。
- MTP、external、ngramを同じbounded propose/verify/accept/accounting contractへ統合した。Qwen productionは明示`MtpAuto`または
  request-local ngramを実行し、`Disabled`は従来の逐次target-onlyを使う。external draftは独立executorが未provisionedなら
  startupで拒否し、configuration-only identityを実行可能providerとして扱わない。

## Production制約

- prefix cacheはexact-greedy text requestだけで有効化する。sampling/grammarはfresh path、multimodalおよびprefix+MTPは
  fail closedである。
- context shiftはQwen dense BF16 text + FP16 KVとGemma textに限定する。Qwen quantized/MoE/multimodal、prefix/draft、
  device-selector samplingとの併用はGPU work前に拒否する。
- checkpointはprompt boundaryのsave/continuationであり、Phase 43以降のwire session IDやmid-generation transport resumeではない。

## Verification status

| lane | 記録 | 判定 |
| --- | --- | --- |
| Host | workspace all-target tests、clippy `-D warnings`、fmt、diff check | PASS |
| Core contracts | prefix/cache/session/context、Qwen/Gemma state image、sampler/grammar snapshot、assistant/speculation | PASS |
| Native host/ABI | public runtime C ABI、fork/COW、all-plane import/export、negative validation、CTest 4/4 | PASS |
| V620 `gfx1030` | FP16 VMM fork/COW 63/64/65/127/128/129、FP8 dynamic/static、NVFP4、linear 5 planes、target-only、cleanup 0 | PASS |
| R9700 `gfx1201` | V620と同一state matrix、target-only、fallbackなし、cleanup 0 | PASS |
| MI300X `gfx942:sramecc+:xnack-` | ROCm 7.14/LLVM 23、wave64 feature-pinned compile/link | compile-only PASS、real run deferred |

実GPU runnerはsource/child K/V byte-exact、child append後のsource不変、encoding別2/4/6 plane、linear active slot/scratchを
数値/state oracleへ照合した。V620/R9700の実行後GPU process、cleanup failure、uncorrectable ECCは0だった。詳細は
[tracked GPU summary](../../../../../ci/matrix/phase41-state-gpu-summary-v1.json)を正とする。

## Reviewで解消した主なfindings

- prefix reader/counter overflowでreader/auditがpartial mutationする経路をpreflight/checked更新へ修正した。
- shared VMM bytesをGemma cache quotaへ計上せず、post-COW bytesもsession accountingへ反映しない問題を修正した。
- context shift後にfull historyを残して二回目shiftが失敗する問題、selector prefillがcontext初期化を迂回する問題、
  capacity exact boundaryのoff-by-oneを修正した。
- checkpoint設定がproductionで無視される経路、Qwen MoEでsilent ignoreする経路、default disabledでもfull context capacityを
  確保する回帰を修正した。
- speculative draft widthとcounter arithmetic、checkpoint position inversion、prefix reader overflowをfail closedへ修正した。

## 引き渡し

- MI300X real state/context/checkpoint runはVM再確保後にexact gfx942 tupleで追加し、compile-only evidenceを実機PASSへ昇格しない。
- true mid-generation checkpoint resume、wire session ownership、external draft executor provisioningは後続API/lifecycle phaseで扱う。
- LMCache/RadixAttention互換、dynamic model lifecycle、Responses/Completions endpointはPhase 41へ前倒ししていない。

[archive plan](../../../../plans/archive/2026/08/21-31/phase41-prefix-session-speculation.md) /
[main plan](../../../../plans/main-plan.md) /
[runtime architecture](../../../../architecture/runtime.md) /
[model lock](../../../../models/model-lock.md) /
[GPU compatibility](../../../../compatibility/gpu.md) /
[AMD GPU compatibility](../../../../compatibility/amd-gpu.md)

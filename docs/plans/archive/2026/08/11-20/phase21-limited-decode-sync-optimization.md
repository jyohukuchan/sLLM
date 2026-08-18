# Phase 21: limited decode segment synchronization optimization

> 状態: completed（candidate rejected、production defaultはprofiledを維持）
> 作成日: 2026-08-17
> 完了日: 2026-08-18

## 目的

単一request、batch 1の通常text decodeで、同一HIP stream上のsemantic opごとに作成・record・query・destroyしている
completion/timing eventを、既存のmodel-neutral execution segment境界へ集約する。semantic op、kernel、provider、tensor layout、
transactional state publicationを変更せず、host orchestrationとHIP runtime event処理だけを減らす。

本Phaseは性能backlog全体を扱わない。通常実行のper-op timing無効化とsegment terminal completionへの集約という一つの
限定candidateを、fresh baseline、構造count、correctness、case固有noise envelopeで評価する。改善しないcandidateはdefaultへ
採用せず、理由と測定結果を履歴へ残してPhaseを閉じる。

## 開始時点の事実

- public HIP numeric operation completionは、completion eventとtiming start eventの2 eventを常時保持し、成功query時に
  `hipEventElapsedTime`を計算する。通常generationでもこのcontractを通る。
- model-neutral `ExecutionSegment`は同一queueのcompletion ownerをboundaryまで保持するが、flush時にはretained ownerを順番に
  個別queryし、各dispatch metadataをauditへ集約する。
- Phase 9/13でprepared plan、same-stream segment、terminal boundary、owner lifetime、transactional state publishは実装済みである。
  Phase 21はこの骨格を置換しない。
- 直近のQwen3.5-4B BF16 GGUF固定laneでは、R9700/V620のmedian TPOTが26.689/29.685 msだった。ただしPhase 21開始時には
  current source、同一artifact、同一toolchainでbaselineを取り直し、履歴値をcandidateの性能証拠として再利用しない。

## Scope

### 対象

- 通常production executionとprofile/evidence executionを区別する内部timing policy。
- 通常実行でのper-op timing start event、elapsed-time取得、per-op completion queryの除去。
- 同一queue・同一streamの非空segmentにつき、terminal completion eventを最大1個record/queryする完了契約。
- terminal成功後にretained ownerをHIP queryなしでfinalizeし、既存dispatch metadataをauditへ集約するmodel-neutral path。
- submission途中の同期error、terminal failure、pending、timeout、cancel、drop時のowner lifetime、quarantine、state非公開。
- Qwen3.5 dense BF16のdual-GPU end-to-end測定と、Gemma 4/Qwen3.5 MoE adapterが共通segment contractを使うhost回帰。

### 非対象

- token IDとpositionのH2D統合、device-side position生成、Argmax readback transferの融合。
- full-attention layerごとのKV append publication境界、KV/GDN/MTP state algorithmの変更。
- HIP Graph/command-list、event/completion pool、registry lock削減、multi-stream実行。
- Matmul/GEMV、attention、MoE、sampling、fusion、量子化kernelの変更。
- request/continuous batching、chunked prefill、prefix cache、永続化、multi-GPU。
- DeepSeek V4、TurboQuant、追加model family、追加model/KV形式。
- public CLI/APIの性能flag、既存GGUF/model-lock形式、README、release packaging。

## 固定する実行契約

1. semantic opの入力、出力、数値順序、provider選択、kernel dispatch数を変更しない。Phase 21のcandidateはhost/runtime
   orchestrationだけを変更する。
2. 既存public C ABIのstandalone completion、query/wait/read/timing semanticsを破壊しない。production集約が追加ABIを必要とする場合は
   versioned additive contractまたは内部execution entryを使い、既存symbolの意味を変更しない。
3. 通常production modeではper-op timingを収集しない。profile/evidence modeだけが既存と同等のper-op HIP event timingを明示的に
   有効化し、host clockによる代用値を返さない。
4. segment terminal eventは、対象ownerと同じcontext、queue、streamにrecordする。別streamのowner、readback、cancel check、
   transactional publishを一つのsegmentへ暗黙結合しない。
5. terminal event成功は、同一stream上でそれ以前にenqueueされたoperationの完了証拠としてのみ使う。各ownerはterminal成功まで
   input/output/state/workspace/queue/plan lifetimeを保持する。
6. dispatch metadataはsubmission時に固定し、terminal成功後にownerごとのHIP queryなしで既存auditへ加算する。submission count、
   kernel dispatch count、fallback、segment/boundary、provider identityを欠落または二重計上しない。
7. terminal failure、timeout、drop、cancel、segment marker作成/record失敗では、未完了output、KV、linear stateを公開しない。
   safetyを確認できないownerは既存のquarantine/poison contractへ渡し、成功扱いで逐次pathへfallbackしない。
8. timing policyは内部で決まり、通常ユーザーのCLI/API操作やmodelごとのopt-inを増やさない。

## 固定した受入条件

### Correctness・lifetime

1. fake-HIP/host testで、通常modeの非空segmentがper-op timing start eventを作らず、terminal completion eventを最大1個だけ
   create/record/query/destroyすることをexact countで確認する。空segmentはeventを作らない。
2. profile/evidence modeではper-op timingが正のHIP elapsed timeを返し、通常modeのcompletion timing要求は明示的な
   unsupportedまたは契約済みの非収集状態となる。silentなhost-clock値を返さない。
3. 1/2/17 owner、segment境界前後、異種semantic/causal/linear ownerを含むhost caseで、順序、dispatch audit、owner解放を照合する。
4. submission途中error、terminal query pending/fatal、marker create/record failure、timeout、cancel、owner dropをfault injectionし、
   double release、use-after-free、event leak、未完了state publishがない。
5. public standalone completion ABI、C/Rust layout、既存completion query/wait/read/timing、numeric operation host testsを回帰する。
6. Qwen3.5-4B BF16 GGUFの固定greedy generationをcanonical V620 `gfx1030`とR9700 `gfx1201`で実行し、baselineとtoken、
   stop、submission/kernel/fallback、segment/boundary、resident/peak、cleanupを照合する。CPU fallback、zero selection、timeoutはPASSにしない。
7. Gemma 4とQwen3.5 MoEについては、共通segment adapter、audit、terminal boundaryをhost testで回帰する。Phase 21だけを理由に
   大型full-model GPU matrixを追加しない。

### 性能candidateの採用

8. A0でQwen3.5-4B BF16 GGUF、greedy、単一request、batch 1、固定prompt/output、ROCm/toolchain、exact target、clock/health条件を
   manifestへ固定し、baselineのevent create/record/query/destroy、completion query、submission、kernel、CPU orchestration wall、
   TTFT、TPOT、E2E、resident/peakを取得する。
9. final比較は各targetでwarmup 3 + measured 10以上とし、baseline/candidate順をcounterbalanceする。median、MAD、p10/p90、
   run-order driftを記録し、単一最良runを採用根拠にしない。
10. candidateは受入条件1の構造削減を満たし、少なくとも一つのprimary wall metricでcase固有noise envelopeを越える改善を示し、
    他targetのprimary metricに説明不能なnoise超過退行がない場合だけ通常modeへ採用する。全target共通の固定率は置かない。
11. candidateが改善しない場合、既存defaultを維持してcandidateを棄却する。限定candidateを実装・測定し、採否と残差を記録すれば
    Phase 21は完了できるが、棄却結果を「高速化済み」と表記しない。

### Integration・文書

12. affected Rust/native host test、exact `gfx1030`/`gfx1201` compile、focused dual-GPU generation、format/clippy/diff/link check、
    integration review 1回を行う。finding修正時はそのfindingだけをfocused再確認する。
13. 採用時はruntime architecture、CI/evidence schema、main plan、historyを同期する。棄却時はsource candidateをdefaultへ残さず、
    baselineと棄却理由をmain plan/historyへ同期する。完了時に本planをarchiveへ移す。

## 実装・検証順序

### P21-A0: fresh baselineとcontract freeze

- current production pathを変更せず、fake-HIP event countと`ExecutionSegment::flush` owner query countを観測できるbounded counterを追加する。
- Qwen3.5-4B BF16 GGUFの固定artifact/model lock、prompt/output、V620/R9700、ROCm/toolchain、clock/health、測定回数をmanifestへ固定する。
- event/completion count、CPU orchestration wall、TTFT/TPOT/E2E、submission/kernel、segment/boundary、memory/cleanupのfresh baselineを取得する。
- public standalone completionとproduction segment completionの互換境界、normal/profile timing policy、failure state machineをtest fixtureで固定する。

### P21-A1: explicit timing policy

- production内部executionへnormal/profileの明示policyを渡す。通常modeはper-op timing start eventとelapsed-time計算を無効化する。
- public standalone ABIと既存profile/evidence pathはper-op timingを維持する。timingが無効なownerから値を読もうとした場合は明示的に拒否する。
- op familyごとの重複runtime実装は共通helperへ集約するが、numeric kernel、descriptor、provider registryは変更しない。
- fake-HIPでtiming enabled/disabled、event create/record failure、release、accountingを確認する。

### P21-A2: segment terminal completion

- 同一queueのretained owner列を閉じるterminal marker/completionを追加し、非空segmentの末尾にeventを1個だけrecordする。
- terminal成功後は各ownerをHIP queryせずにfinalizeし、保持resourceを解放してsubmission時dispatch metadataをauditへ加算する。
- marker作成前のsubmission error、marker create/record failure、pending/fatal、timeout/drop/cancelを既存safety/quarantineへ接続する。
- readback、transactional publish、error/cancel boundaryは既存`ExecutionBoundaryKind`のまま維持し、境界を跨ぐ集約は行わない。

### P21-A3: model-neutral integrationとfocused correctness

- common `ExecutionSegment`へterminal completion pathを接続し、Qwen/Gemma/MoE adapter側の個別最適化を追加しない。
- synthetic owner数1/2/17、異種owner、空segment、複数segment、全fault injection、audit exact-countをhostで検証する。
- public ABI、all-target Rust、native fake-HIP、exact target compileを実行する。
- Qwen3.5-4B BF16固定generationをV620/R9700で確認し、token/state/audit/memory/cleanup差分があれば性能測定へ進まない。

### P21-A4: performance decisionとcloseout

- A0と同一tupleでbaseline/candidateをcounterbalancedに測定し、構造count、CPU wall、TTFT、TPOT、E2E、memoryを比較する。
- case固有noise envelopeに基づいて採用または棄却する。採用しないcandidate、未実装backlog、測定不能なclaimを完了結果へ混ぜない。
- integration review 1回とfindingのfocused re-reviewを行い、runtime/main plan/history/evidence schemaを同期する。
- planをarchiveへ移し、次のPhaseは自動で割り当てない。

## 計測lane

| lane | 用途 |
| --- | --- |
| D0 | fake-HIP event/query exact count、synthetic segment、fault injection。GPU性能claimに使わない |
| D1 | 変更対象GPU、Qwen3.5-4B fixed short decode、warmup 1 + measured 3。correctnessと改善方向だけを見る |
| D2 | canonical V620/R9700、同一GGUF/lock、warmup 3 + measured 10以上、counterbalanced baseline/candidate。最終採否に使う |

## Rollback・停止・再計画

- normal/profile timingを安全に分離できない、public standalone completion semanticsを破る、またはowner lifetimeをterminal event一つで
  証明できない場合は、ABIの意味を変更せずcandidateを棄却する。
- token/state/audit mismatch、silent fallback、event/resource leak、cleanup不良はperformanceに関係なくblockerとする。
- structural countを削減してもwall metricがnoise内または退行する場合はdefaultへ採用せず、event pool、graph replay、H2D統合へscopeを
  広げない。それらは未割当backlogに残す。
- 同じwork unitの2回reject、review時間が実装時間超、1時間以上の機能進捗停止、検証/docs 30%超、見積り1.5倍超、
  acceptance変更時は追加探索・検証を止めて同じwork unitを再計画する。

## 完了結果

- 通常opのeventless completion、同一queueのuntimed fence、fence成功後のowner finalizeをadditive C/Rust ABIとして実装した。
  standalone completionとprofile/evidence pathの既定は従来どおりPROFILEDで、per-op timingを維持する。
- fake-HIPでは17 ownerがper-op eventを作らず、segment全体で1 eventだけを作ることをexact countで確認した。
  fence成功前finalize、fence record failure、active submission中のmode変更、release/accountingもfail-closedに確認した。
- primary laneはQwen3.5-4B BF16 GGUF、prompt `Hello world`、greedy、3-token output、
  canonical V620/R9700単独可視化、各3 warmup + 10 measured、baseline/candidate交互順に固定した。
  全runでtoken `[0,271,760]`、HIP-only、fallbackなし、submission/kernel、segment/boundary、cleanup 0が一致した。
- final counterbalanced E2E中央値はV620でbaseline 4.14636秒、candidate 4.15222秒（+0.14%）、
  R9700でbaseline 4.92133秒、candidate 4.93015秒（+0.18%）だった。MADはそれぞれ
  baseline/candidate 29.41/40.24 ms、43.91/33.45 msで、差はnoise内かつ改善ではない。
- 構造削減は成立したが固定した採用条件を満たさないためcandidateを棄却した。Qwen/Gemma productionは
  `ExecutionSegment::profiled`を選び、通常defaultのper-op completion/timingを維持する。deferred primitiveは
  fault-testedな実験基盤として残すが、高速化済みとは扱わない。
- Gemma 4のreview中に、一時的にstate publicationをaudit boundaryへ加算してboundary countを6から150へ変える差分を検出した。
  同期を維持したまま未追跡の既存publication境界として処理し、最終GPU smokeでbaselineと同じ6へ戻した。
- Rust affected test、native host 3/3、H3 contract 65 test、JSON/schema、clippy（既存3 lintを明示allow）、
  exact gfx1030/gfx1201 release compile、両GPU Qwen smoke、V620 Gemma smokeをPASSした。

[対応する履歴](../../../../../history/2026/08/11-20/phase21-limited-decode-sync-optimization.md)

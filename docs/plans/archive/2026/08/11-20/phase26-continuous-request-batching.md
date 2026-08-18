# Phase 26: continuous request batching

> 状態: 完了（candidate棄却、host plannerのみ保持、production未採用）
> 作成日: 2026-08-18

## 目的

Phase 23の`P23-O3`を実装し、現在の「FIFOから一件を取り出し、generation全体を一つのworkerとbackend mutexが保持する」
serviceを、複数requestのwaiting/running setから毎step実行対象を選ぶcontinuous request batchingへ置き換える。

同一model/compatible execution classのdecode tokenを一つのGPU batchへまとめ、並行requestのaggregate throughputを高める。
各requestのtoken順、sampling RNG、stop/usage、KV/GDN/MTP state、cancellation、stream出力、error、cleanupは独立に維持する。
単なる複数worker化やmutex範囲縮小をrequest batchingと呼ばず、GPU dispatchの`B>1`と実測aggregate改善を必須のmechanism proofとする。

## 開始根拠

- `sLLM.md`は単一requestと複数requestのリクエストバッチ処理の両方で主要推論engineより高速であることを目標にしている。
- Phase 23のconcurrency=2ではV620の完了時刻が0.471/0.937 s、R9700が0.325/0.651 sで、ほぼ完全に直列化していた。
  HTTP/SSE residualは約0.5〜0.6 msであり、transportではなく単一whole-generation workerが主要因だった。
- current `SchedulerV1`は一件ずつ`spawn_blocking`し、`ChatGenerationBackendV1::generate`がrequest全体を同期実行する。
  production backendもresident modelとmutable request stateを同じmutex内に置き、generation終了まで保持する。
- Phase 25はprojection familyを`M=B`へ拡張可能なcontractで整備する。Phase 26はその成功に依存せず、Phase 25の採用または
  rollback後のstable sourceから開始し、batch shapeでproviderを再選択する。
- R9700は`M>1` hipBLAS GEMMが強く、request batchingでdecodeを`M=1`から`M=B`へ変えることはPhase 24で観測した
  arithmetic-intensity/provider差を活用できる可能性がある。

## 用語

- 本Phaseでは複数requestを動的に束ねる機能を「リクエストバッチ処理」または`continuous request batching`と呼ぶ。
  単に「バッチ処理」と書かない。
- `B`: 一回のGPU decode stepへ参加するrequest row数。
- waiting set: admission済みだがprefillまたは実行slotを待つrequest。
- running set: device stateを所有し、prefill/decode/finishのいずれかへ進めるrequest。
- compatibility class: 同一model resident、dtype/encoding、execution kind、sampling/logits requirement等から、同じGPU graphへ
  安全に束ねられるrequest集合。

## 権限とproposal status

- この文書はユーザーの詳細計画作成指示に基づく。Phase番号とPhase 25後の順序は割り当てるが、production実装、public設定追加、
  GPU再実行、採用thresholdのhard gate化は、ユーザーがPhase 26開始または本planを明示承認した時点で行う。
- 下記throughput/latency thresholdはnonblocking AI proposalである。originはPhase 23 `P23-O3`、scopeはQwen3.5-4B BF16
  OpenAI-compatible service、costはscheduler/execution ownership再構成とdual-GPU concurrency evidence、expiryはP26-A0完了時とする。
- request間state alias、wrong token、cross-request RNG/output、cancellation漏れ、unbounded queue/memory、fallback、cleanup failureは
  correctness/security blockerであり、throughputで相殺しない。

## Primary scope

- model: Qwen3.5-4B dense BF16 GGUF、text-only、通常greedy generation。
- target: canonical V620 exact `gfx1030`、R9700 exact `gfx1201`。
- transport: OpenAI-compatible non-streamとSSE。direct host harnessはscheduler semanticsとGPU mechanismのcontrolに使う。
- concurrency: `C=1,2,3,4,7,8`。production初期上限はP26-A0のVRAM/throughput/fairness結果で固定する。
- execution:
  - waiting/running setとrequest state machine。
  - requestごとのtokenizer/sampler/stop/output/cancellation state。
  - immutable resident modelとmutable per-request execution ownerの分離。
  - compatible requestのbatched decode、row-to-request mapping、per-row position/state binding。
  - whole-prefillをdecode round間へboundedに挿入するprefill/decode interleave。
  - bounded output ringとnetwork writer分離によるrequest単位backpressure。

## Compatibility lanes

| lane | Phase 26での扱い |
| --- | --- |
| Qwen dense BF16 greedy text | primary、`B>1`必須 |
| Qwen dense BF16 sampled text | per-request RNG/logitsを分離して対応 |
| Qwen MTP | initial singleton class可。通常requestを止めず、既存exact semanticsを維持 |
| Qwen multimodal | vision prefillはsingleton、text decodeは安全にclass一致する場合だけbatch化 |
| Qwen MoE / FP8 / NVFP4 / MXFP4 | correctness control。provider未対応ならsingleton class |
| Gemma 4 | scheduler/lifecycle host contractを共有。GPU batchingはsecondary gate後 |

singleton classはGPU/CPU fallbackではなく、既存production pathを一request rowで実行する互換laneである。primary Qwen dense BF16が
`B>1`にならない実装をPhase 26成功とはしない。singleton laneがactive batch全体を同期的に停止しないscheduleを定義する。

## 非対象

- chunked prefill、prefix/Radix cache、LMCache、KV永続化、speculative algorithm変更。
- multi-GPU、tensor/expert/pipeline parallel、RCCL、Infinity Fabric、RDMA、複数modelを一batchへ混ぜる実装。
- attention algorithm、new quantization、TurboQuant、DeepSeek V4、MoE grouped GEMM、GPU sampling自体の新規最適化。
- model load、GGUF hash/upload pipeline、vision encoder batching。
- 無制限queue、要求ごとの専用GPU stream、thread-per-request GPU submit、単なる同時host worker化。
- long promptを細分化するchunked prefill。Phase 26ではwhole-prefill間の順序とdecode優先度を制御するが、一つのprefill内部は分割しない。

## Architecture contract

### Request state machine

requestは少なくとも次の状態を持つ。遷移はschedulerだけが行い、terminal状態から再実行しない。

`Queued -> Admitted -> PrefillReady -> DecodeReady -> Backpressured -> DecodeReady -> Finished`

どのnonterminal状態からもrequest-localに`Cancelled`または`Failed`へ遷移できる。GPU submission後のcancelはcompletionまでresourceを
保持し、結果を公開せずstateを安全に破棄する。別requestのbatch rowは継続できる設計を優先する。

### Ownership分離

- resident model、immutable weight、model lock、kernel/prepared templateはmodel backendが共有する。
- KV、GDN/linear state、workspace slice、position、token history、sampler RNG、stop matcher、output ring、auditはrequest ownerが持つ。
- current backend mutexをgeneration全体へ保持しない。mutex/lockはwaiting/running metadataの短い更新か、request ownerの排他的遷移に限定する。
- row mapは`batch row -> request owner ID -> token/position/state slice`をcheckedに保持し、completion後も同じmappingでoutputを配る。
  request IDのsortやcompactionでRNG/output順序を変えない。

### GPU execution

- batch descriptorは`B` rowのtoken/position、rowごとのKV/GDN state binding、outputを表現する。異なるposition/committed lengthを
  単一scalarへ潰さない。
- full attentionはrowごとのlogical KV lengthとstorage ownerを分離する。必要なら既存optional block-table abstractionを使うが、
  Phase 26のためだけにprefix sharing/COWを追加しない。
- linear/GDN state更新はstaged outputからrequestごとにcommitし、row-local validation failureやcancelが別rowのpublished stateへ
  波及しない。batch-wide backend failureでは参加ownerをfail closedにし、未参加requestのstateを変更しない。
- batched Argmax/logitsのrow順をrow mapへexactに対応させる。sampled requestのRNG消費はrequest単独実行と同じ回数・順序にする。
- prepared/cache identityはmodel、compatibility class、`B`、row layout、state/binding generationを含める。異なるrequest ownerの
  raw pointer identityを永続cache keyへ漏らさず、templateとdynamic bindingsを分離する。

### Scheduling・fairness

- decode-ready requestは一roundにつき原則一token進めるround-robinをbaseline policyとする。
- 新規prefillはdecode round間だけに挿入し、連続prefill数またはtoken budgetをboundedにする。chunked prefillがないため、
  単一long-prefill中のpreemptionは保証せず、固定mixed caseでhead-of-line delayを明示する。
- compatibility classごとにbatchを作るが、少数classを無期限starveさせないage/round boundを持つ。
- output ringが満杯のrequestはGPU batchから一時除外し、別requestを停止させない。上限超過、disconnect、timeoutでは
  そのrequestだけをcancelする。
- shutdownは新規admission停止、queued failure通知、active cancellation、in-flight drain、request state解放、resident shutdownの順とする。

## 提案する受入基準

### Host・scheduler correctness

1. deterministic fake clock/executorで`C=1,2,3,4,7,8`、異なるprompt/output長、stop時刻、sampling classを実行し、
   admission、row mapping、round-robin、finish順、usageをoracleと一致させる。
2. queue full、event ring full、slow SSE consumer、disconnect、timeout、shutdown、backend error、panic/join failureでbounded memoryと
   request-local cancellationを確認する。一requestのfailureで無関係なrequestをcancelしない。
3. same seed/profileの各requestは単独実行と同じgenerated token、visible output、stop reason、usage、RNG消費順を持つ。
4. duplicate/unknown request ID、row欠落/重複、stale generation、state slice overlap、position/KV length不一致をdispatch前に拒否する。
5. singleton compatibility laneとbatched laneの切替でpublic OpenAI profile、SSE event順、reasoning/content分離、error schemaを変えない。

### GPU・model correctness

6. tiny batched GPU oracleは`B=1,2,3,4,7,8,15,16,17`と異なるrow position/stateを両GPUで実行し、独立referenceへ照合する。
7. primary full-modelは同じrequest集合をserial baselineとcontinuous candidateで実行し、requestごとのprompt/completion tokens、
   output、stop、usage、KV/GDN committed length、sampling stateをexact一致させる。
8. cancellation caseはbatch submit前、in-flight、completion後publication前を含み、cancelled rowを公開せず、残りrowが正しく継続する。
9. dispatch auditはexact HIP target、`B>1` kernel/provider、fallbackなし、row count、submission、cleanup terminal-zeroを記録する。

### Performance・capacity

10. production測定はprofilerなし、同一model/request set、3 warmup + 10 measured set以上、serial/candidate counterbalanced順で行う。
11. primary homogeneous caseは28 input / 128 outputの同一requestを`C=1,2,4,8`。mixed caseは17/32、255/64、256/128、
    257/32を同時投入し、arrival skew `0/5/17 ms`も別caseで測る。
12. metricはaggregate completion tokens/s、request/s、makespan、per-request TTFT/TPOT/E2E p50/p95/max、GPU batch occupancy、
    queue wait、prefill stall、VRAM high-waterとする。最初のfinishだけをthroughput改善として扱わない。
13. `C=1`の固定patternでstableな性能悪化を残さず、`C=2`のaggregate completion tokens/sをserial baseline比30%以上改善することを
    primary adoption proposalとする。両GPUで非悪化を満たし、少なくとも一targetで30%以上、もう一targetで20%以上を要求する。
14. `C=4/8`はthroughputが`C=2`より低下しない範囲でdefault上限候補にする。VRAM不足、tail latency急増、provider crossover後退が
    見える最小`C`の一つ前をbounded defaultとし、無理に8へ固定しない。
15. fairnessは同一class/arrival windowで、最も遅いrequestのdecode service回数が最も速いunfinished requestから2 round超遅れない。
    mixed prefillによる非preemptible stallは別に計測し、chunked prefill未実装を隠さない。
16. model-resident bytesはbaseline不変。request-state/workspaceはactive request数と明示的に比例し、queue中requestへGPU stateを
    先行確保しない。admissionはcontext容量とdevice-memory budgetをcheckedに満たす。

### Evidence・closeout

17. source/tree、binary、runner、oracle、ROCm、target、GPU identity、model/derived lock、request-set digest、arrival schedule、
    scheduler configをevidenceへ固定する。
18. raw model、binary、生成全文、full logits、rocprof trace、request本文を追跡せず、bounded aggregate、digest、schema、runner、
    plan/historyをcommit対象にする。
19. affected server/frontend/core/HIP checks、dual-GPU correctness/performance、1回のintegration review、findingのfocused re-review、
    main plan/runtime/history/provenance同期を完了する。
20. throughput基準未達でも、正しいstepwise ownershipが将来機能の独立価値を持つ場合は自動採用しない。production default、
    retained infrastructure、reverted candidateを分類してユーザーへ判断を返し、negative resultでPhase 26を完了できる。

## 計測matrix

| case | request set | 目的 | 採否への使用 |
| --- | --- | --- | --- |
| H0 | fake `C=1,2,3,4,7,8` | lifecycle、row map、fairness | correctness |
| H1 | cancel/error/backpressure/shutdown | isolation、bounded resource | correctness |
| G0 | synthetic `B=1,2,3,4,7,8,15,16,17` | batched GPU numerical/state oracle | correctness |
| Q1 | 28/128 × `C=1` | single-request regression | hard performance |
| Q2 | 28/128 × `C=2` | primary aggregate throughput | hard performance |
| Q4 | 28/128 × `C=4` | scaling、VRAM、tail latency | default上限 |
| Q8 | 28/128 × `C=8` | saturation boundary | default上限 |
| M0 | 17/32 + 255/64 + 256/128 + 257/32 | mixed length/fairness | hard regression |
| A0 | same requests、arrival `0/5/17 ms` | continuous admission | mechanism/performance |
| S0 | greedy + 3 sampled profiles | per-request RNG/logits isolation | correctness |
| C0 | one in-flight cancellation | surviving rows継続 | correctness |
| B0 | slow SSE + fast non-stream | backpressure isolation | correctness/performance |
| X0 | singleton MTP/multimodal/MoE control | compatibility保持 | correctness |

## 作業順序

### P26-A0: baseline・policy・threshold freeze

- Phase 25 final source identityをbaselineにし、成功candidateを前提にしない。
- current Q1/Q2/Q4、scheduler queue wait、backend mutex hold、GPU idle gap、VRAM、SSE backpressureを両GPUで再取得する。
- compatibility class、max active/queued、memory admission、round-robin、prefill insertion、backpressure、timeout、shutdown policyを固定する。
- correctness tolerance、request-set/arrival digest、performance threshold、secondary lanesを実装前manifestへ固定する。

### P26-A1: stepwise frontend・scheduler host contract

- whole-generation `generate`だけに依存しないrequest state objectを追加し、prepare/admit/prefill/select/accept/finish/cancelをstep化する。
- fake executorでwaiting/running set、row map、per-request sampler/stop/output、fairness、error/cancel/shutdownを実装・検証する。
- current public API/SSE adapterは保持し、transportからscheduler stateを直接操作させない。

### P26-A2: resident model・request owner分離

- production backend mutexからimmutable resident/session/templateとmutable request ownerを分離する。
- requestごとにKV/GDN/workspace/audit/lifecycleを所有し、admission前とcleanup後のmemory accountingを追加する。
- singleton executionでQ1と全existing semantic testを通し、ownership変更だけによる性能・correctness差を閉じる。

### P26-A3: batched decode execution

- typed row map、per-row token/position/state binding、batched projection/attention/GDN/terminal outputを実装する。
- Phase 25 providerを`B>1`で再選択し、B1-specific decisionを無条件に引き継がない。
- G0を両GPUで通し、in-flight cancellation、partial failure、state publication、fallback、cleanupを確認する。

### P26-A4: continuous scheduling・prefill interleave

- decode-ready rowsをcompatibility classごとに束ね、一round一tokenのbaseline policyで実行する。
- waiting prefillをdecode round間へboundedに挿入し、singleton class、sampled row、backpressured rowをstarveさせない。
- output ring/network writerを分離し、slow consumerがGPU batchと別requestを停止しないことをH1/B0で確認する。

### P26-A5: dual-GPU full-model採否

- Q1/Q2/Q4/Q8/M0/A0/S0/C0/B0/X0をserial/candidate counterbalanced順で実行する。
- correctness、aggregate throughput、single-request regression、fairness、tail latency、VRAM、mechanism proofをまとめて採否する。
- target差はまずmax batch/provider selectionで吸収し、scheduler/model pathの複製はcorrectnessまたは再現する重大な性能問題が
  共通policyで解消できない場合だけ検討する。

### P26-A6: integration・closeout

- server/frontend/core/HIP tests、schema、OpenAI compatibility、runtime docs、provenance、one integration reviewを完了する。
- bounded aggregateとdefault max-active/max-batchの根拠を記録し、main plan/historyを同期してplanをarchiveする。
- chunked prefill、GPU sampling、MoE/Gemma batchingは結果に応じて別の未割当follow-upへ戻し、自動開始しない。

## Rollback・停止・再計画

- cross-request token/state/RNG/output混同、unbounded memory、cancel propagation、fallback、cleanup failureが一つでもあればproductionへ採用しない。
- host同時実行だけでGPU row countが常に1、またはQ2 aggregate throughputが改善しない場合はcontinuous request batching成功と表記しない。
- single-request悪化をconcurrency throughputで無条件に相殺しない。共通schedulerで解消できなければcandidate/defaultを分離して判断する。
- chunked prefill、GPU sampling、prefix cache、multi-GPUを追加してQ2を救済しない。
- Phase 25 candidateがPhase 26の`B>1`で悪化する場合は、Phase 26 provider selectionを再計測し、Phase 25 historyを書き換えない。
- 同じwork unitが2回reject、review時間が実装時間超、functional progressが1時間停止、verification/docsが30%超、
  見積り1.5倍超、acceptance変更時は追加実装・検証を止め、同じwork unitを再計画する。

[Phase 23 bounded summary](../../../../../../ci/matrix/phase23-performance-discovery-summary-v1.json)
[Phase 25 archive](../../../../archive/2026/08/11-20/phase25-batch-compatible-projection-family-optimization.md)
[Phase 26 bounded summary](../../../../../../ci/matrix/phase26-continuous-request-batching-summary-v1.json)
[対応する履歴](../../../../../history/2026/08/11-20/phase26-continuous-request-batching.md)

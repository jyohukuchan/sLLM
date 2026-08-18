# Phase 23: cross-engine differential performance discovery

> 状態: complete（探索・順位付け完了、production最適化は未実装）
> 作成日: 2026-08-18

## 目的

Phase 21のcompletion event集約とPhase 22のshape-aware BF16 matvecは、局所的な構造削減またはoperator改善を
確認できた一方、production full-model wall timeの有意な改善へ転化しなかった。Phase 23は次の最適化を先に選ばず、
現行sLLMのcritical pathを細粒度に計測し、既存推論engineとの差分から、これまで見落としていた最適化余地を抽出する
探索専用Phaseとする。

成果は、比較可能性を監査した測定結果、wall timeの帰属、engine間の実装・実行方式の差、既知backlogとの照合、
Amdahl上限を含むranked opportunity inventoryである。kernel、runtime、scheduler、frontend、model loadのproduction実装は
変更せず、Phase 24以降へ渡す最大3件の候補と、その反証条件を決める。明確な大候補がなければ、その否定結果自体を
Phase 23の有効な完了結果とする。

## 開始時点の事実

- Phase 5のQwen3.5-4B BF16 baselineでは、exact-token llama.cpp wrapperに対するsLLM TTFTがV620で
  49.4〜278.5倍、R9700で31.4〜742.1倍長かった。差は255〜1024 token prefillで拡大したが、当時のsLLM内訳は
  request、prefill、decode、render/tokenize程度であり、差を特定のhost処理、transfer、kernel、waitへ帰属できない。
- Phase 21は17 completion ownerを1 fenceへ集約したが、full-model中央値はV620/R9700で0.14%/0.18%遅くnoise内だった。
- Phase 22はV620のMLP gate/up operatorを約32%短縮したが、最終full-modelは0.52%遅かった。operator単独の比率、
  同時に悪化した区間、host/GPU overlap、計測されていないresidualを一つのcritical pathとして説明する必要がある。
- 既存性能backlogにはdispatch、H2D/D2H、fusion、quantized path、MoE、MTP、sampling、service、request state、
  model load、long context等の仮説がある。ただし、候補間を同じ尺度で比較したfresh evidenceと、既存engineとの差分に
  基づく網羅性確認はまだない。
- Phase Xでは、llama.cpp HIPの大幅な性能差がGDN kernelではなくFlash Attention build coverageに由来した。
  source/build option、実際に選択されたprovider、fallbackを同時に監査しなければ、profile上の上位kernelだけでは
  根因を見落とすことを示している。

## Scope

### 対象

- current sLLMのcold start、warm single-request、prefill、decode、sampling、frontend、OpenAI-compatible API、
  request admission/queue、cleanupを含むend-to-end critical path。
- primary workloadはQwen3.5-4B BF16 GGUF、greedy、単一request、canonical V620 exact `gfx1030`と
  R9700 exact `gfx1201`とする。
- primary比較peerは同一model source revisionから生成したBF16 artifactを使う固定llama.cppとする。
- serving engine controlは、vLLMまたはSGLangのうち、A0で同一model semantics、dtype、GPU-only、入力/output条件を
  fail-closedに満たせるものを少なくとも一つ選ぶ。両方が成立する場合は両方を使う。成立しないengineは実行比較から外し、
  非比較理由とarchitecture/source上の技術差だけを記録する。
- secondary workloadは、primaryで見つけた支配区間がmodel共通かを判定するため、Gemma 4 mixed NVFP4 dense pathと
  Qwen3.5-35B-A3B MXFP4 MoE pathの代表caseをR9700で取得する。V620はmodelとworkspaceが安全に収まるbounded caseだけを使う。
- direct engineとHTTP non-stream/SSEを分離し、transportを含む差と含まない差を混同しない。
- host span、HIP API/runtime trace、memory copy、GPU kernel、同期・idle、VRAM/clock/temperature/powerのbounded summary。
- 既存engineの公開documentation、固定source、実行manifestから、provider selection、graph reuse、memory planning、
  batching/scheduling、sampling、tokenization、model loadに関する技術差を抽出する。
- 既存backlogの各項目を`confirmed`、`new`、`lower-priority`、`disproved`、`premise-changed`へ分類する。

### 非対象

- kernel、provider、runtime、scheduler、frontend、API、model loaderのproduction変更。
- candidate実装、autotuning、request batching、chunked prefill、prefix cache、TurboQuant、DeepSeek V4、multi-GPUの実装。
- performance thresholdのrelease gate化、全model・全shape・全engineの総当たり、長時間load/soak test。
- 条件が異なるweight format、quantization、context、sampling、batch、GPU offloadを同じ速度比として扱うこと。
- CPU fallback、partial GPU offload、timeout、crash、zero sample、別GPU targetをGPU performance evidenceとして扱うこと。
- vLLM、SGLang、その他の非llama engineからのcode copy、adapt、port。
- model、binary、raw profiler trace、生成全文、大きなsample列のGit管理。

## 比較可能性契約

各engine/runは、次を一つのcomparison manifestへ固定する。

1. engine repository、完全commitまたはrelease、build flags、compiler/runtime/library、binary hash。
2. model source revision、artifact hash、weight/activation/KV dtype、tensor layout変換、context上限、offload範囲。
3. exact prompt token IDs、requested/actual output tokens、stop token、greedyまたはsampling設定、seed、chat template。
4. batch、concurrency、parallel slot、prefill chunk、prompt/prefix cache、MTP/speculation、Flash Attention、graph/capture、
   tuning/cacheのon/offとwarm state。
5. exact GPU UUID/BDF/target、visible device、clock/power profile、ROCm、kernel、他process、実行前後health。
6. metric境界、clock source、warmup/measured回数、run順、timeout、異常sampleの扱い。

比較は次の三層に分ける。

| comparison class | 条件 | 許可する主張 |
| --- | --- | --- |
| E0 semantic-exact | 同一token列、model semantics、dtype、GPU、batch、cache、出力条件 | E2E、TTFT、TPOT、prefill/decode throughputの比率 |
| E1 system-equivalent | source modelと意味は同じだがartifact layoutまたはengine固有最適化が異なる | system全体の差。kernel単体の優劣へ読み替えない |
| E2 diagnostic-only | dtype、quantization、offload、機能、metric境界のいずれかが一致しない | 方向性と技術差だけ。速度比と勝敗を出さない |

engine固有のtokenizer/renderを含むAPI比較とは別に、可能なengineではexact token IDsを直接与えるlaneを持つ。
両engineで同じmetric境界を作れない場合は共通の外側wall clockを正とし、engine内部metricは内訳としてだけ扱う。

## Workload matrix

### Primary broad scan

| lane | input / output | concurrency | cold/warm | 目的 |
| --- | --- | ---: | --- | --- |
| C0 | model loadのみ | 0 | cold | open/read/hash、parse、repack、allocation、H2D、prepareを分解 |
| S1 | 1 / 1 | 1 | warm | request固定費、first-token下限 |
| S2 | 17 / 17 | 1 | warm | 非整列の短い対話、Phase 5継承 |
| P1 | 255 / 32 | 1 | warm | 256境界直前 |
| P2 | 256 / 32 | 1 | warm | 256境界 |
| P3 | 257 / 32 | 1 | warm | 256境界直後 |
| P4 | 2,047 / 32 | 1 | warm | 2K境界直前のprefill支配case |
| P5 | 2,049 / 32 | 1 | warm | 2K境界直後、chunk/provider切替検出 |
| D1 | 17 / 128 | 1 | warm | decode支配、ITL分布とtoken間gap |
| Q1 | 17 / 32 | 2 | warm | queue、同時要求、serialization検出 |
| Q2 | 17 / 32 | 4 | warm | scheduling/batchingのsystem差 |

- まずC0/S1/S2/P2/D1/Q1のbroad scanを行い、支配区間またはengine差が変化する境界だけP1/P3/P4/P5/Q2へ進める。
- full broad scanは各process内で3 warmup + 10 measuredを基本とし、cold loadはprocess cache条件を明示して独立5回以上とする。
- baselineとpeerはrun順を交互化する。GPUを跨ぐ絶対値比較ではなく、各GPU内のengine差と同一engineのtarget差を分ける。
- secondary modelはS2/P2/D1のうちprimaryで支配要因が変わった最小caseだけを使い、探索を全matrixへ拡大しない。

### API・sampling controls

- direct exact-token、CLI render/tokenize、HTTP non-stream、HTTP SSEを同じprompt/outputで測り、差分を
  render、tokenize、JSON、queue、generation、stream writerへ帰属する。
- primary final比較はgreedyとする。samplingのcostを見る独立controlではfixed seed、temperature/top-p、penalty、
  vocabulary sizeを揃え、token列の一致をperformance比較条件にはしない。
- prefix/prompt cache、MTP、continuous batching等はprimary exact laneで無効化する。engineのproduction advantageを
  観測する別laneでは有効状態を明記し、exact laneと比率を混ぜない。

## 計測モデル

### End-to-end span

requestごとにmonotonic request ID/span IDを付け、少なくとも次の境界を記録する。

1. socket accept / CLI start、parse、validation、chat render、tokenize。
2. admission、queue wait、worker acquire、model/backend lock acquire。
3. request state作成、graph/template lookup、prepared plan lookup/prepare、allocation。
4. prefill H2D、host enqueue、GPU kernels、completion wait、first logits/sample/token publication。
5. decode各stepのtoken/position準備、H2D、op submit、KV/state publication、argmax/sampling、D2H、decode、stop。
6. SSE/non-stream encode、channel/backpressure wait、socket write、request cleanup。
7. cold laneのfile open/read/hash、GGUF metadata/tensor table、mmap/read、CPU transform/repack、device allocation、
   H2D、prepared plan作成、model-ready publication。

instrumentation off/onでtoken、stop、dispatch、allocation/cleanupが一致することをhost/focused GPU controlで確認する。
span timestamp自体のoverheadを測り、instrumented wallをproduction E2E値へ流用しない。

### GPU・runtime分解

- rocprofiler/rocprofv3等の固定toolでHIP API、kernel dispatch、memory copy、marker相関を別runとして取得する。
- kernelはsemantic roleとshapeへ結び、matvec/GEMM、attention/GDN、normalization、elementwise/fusion、KV、quantize/dequantize、
  routing/MoE、argmax/sampling、otherへ分類する。
- count、total/median/p90 duration、bytes、grid/workgroup、provider/kernel ID、launch間host gap、stream idleを集計する。
- GPU counter、occupancy、bandwidth、cache、wave stallは上位3区間だけの独立laneで取得し、counter収集によるserializationを
  normal wallへ混ぜない。
- H2D/D2Hはpayload size、回数、同期境界、staging/pinned/pageableを記録する。4-byte転送も回数とcritical-path waitで評価する。

### Critical-path accounting

単純なduration合計ではなく、overlapを考慮したinterval unionと依存関係からcritical pathを作る。

`E2E = frontend + queue/lock + host-active + host-wait + transfer-critical + GPU-critical + response/cleanup + residual`

- overlapしたhost、copy、GPU intervalを二重計上しない。
- 各primary caseでE2Eの95%以上を上記categoryへ帰属することを目標にする。95%未満でも測定を成功値から除外せず、
  residualの大きさ、位置、推定原因、追加計測costを明示して候補のconfidenceを下げる。
- tokenごとのITLをGPU-active、host gap、transfer/waitへ分解し、medianだけでなくp10/p90、MAD、周期的spikeを保持する。
- Phase 21/22の局所差について、改善区間、相殺区間、unaccounted区間を再現できる範囲で説明する。

## Engine差分の抽出規則

### 実行差分

- caseごとにsLLMとpeerのTTFT/TPOT/E2E差を、cold load、frontend、queue、prefill GPU、decode GPU、host gap、
  transfer/sync、sampling/responseへwaterfall分解する。
- peerが速い区間だけでなく、sLLMが同等または速い区間も残し、改善不要な領域を明確にする。
- source/build option、実provider/kernel、graph/capture、workspace、memory pool、thread/stream、batch/cacheの差をmanifestと
  traceの両方から照合する。設定差だけで説明できる場合はcode最適化候補にしない。

### Reference inspectionとprovenance

- llama.cppは固定commitとbounded file/functionを記録して技術差を確認する。直接reuse候補が生じた場合は、Phase 23では
  importせず、reuse mode、license/provenance cost、対象箇所を候補ledgerへ記録する。
- vLLM、SGLang、その他のengineは公開documentationと固定sourceから技術的事実だけを抽出し、inspection noteを
  implementation案から分離する。source expression、control flow、testをcopy、adapt、portしない。
- paper、標準、vendor documentationに同じ技術根拠がある場合はそちらをimplementation basisの第一候補にする。

### Opportunity ledger

各候補は次を必須fieldとする。

- observationと測定case、sLLM wall share、peerとの差、関連trace/span。
- suspected mechanismと反証可能な仮説。単に「peerが速い」では候補にしない。
- `new` / `existing-backlog` / `premise-changed` / `disproved`の由来分類。
- model/GPU/dtype/workload coverage、共通性優先順位、correctness/semantic boundary。
- Amdahl上限、現実的期待改善、confidence、実装cost、検証cost、risk、provenance/reuse区分。
- 最小のfalsification microcase、full-model採否lane、rollback境界、候補のexpiry条件。

rankは少なくとも`critical-path share × credible removable fraction × coverage × confidence`を分子に、
実装・検証costとriskを分母にした共通尺度で比較する。算式の数値だけで自動決定せず、複数model/GPUへの共通性と
product要件を併記する。Phase 24候補は最大3件とし、一つ目は原則として最も共通性が高いものを選ぶ。
full-modelで5%以上の現実的改善余地は、Phase 21/22のnoise内結果を受けたPhase 23計画時のAI提案による
nonblocking優先度heuristicとする。scopeはPhase 24 shortlist、costは小さいが共通性の高い改善を下位へ置く可能性、
expiryはPhase 23 closeoutとし、5%未満の候補の記録・提示・ユーザー選択を妨げない。

## 作業順序

### P23-A0: measurement contractとpeer固定

- current source/build/model/GPU identity、既存Phase 5/9/21/22 evidence、benchmark metric境界を棚卸しする。
- llama.cppとserving controlの固定revision、build option、model artifact、GPU supportを確認し、E0/E1/E2へ分類する。
- workload matrix、exact token IDs、cache、warmup/sample、counterbalance、health、timeout、raw artifact保存先をmanifestへ固定する。
- phase開始時の既存backlog snapshotを作り、計測前に候補順位を付けない。

### P23-A1: coarse matched cross-engine scan

- primary broad scanをsLLM/llama.cpp/serving controlで実行し、cold/warm、direct/API、prefill/decode/concurrencyの
  差が拡大するcaseを特定する。
- E0/E1/E2を監査し、比較不能なrowをratioから除外する。
- deep profile対象を最大3 critical-path regionへ絞る。

### P23-A2: sLLM fine-grained critical-path instrumentation

- additive host spansとrequest/kernel correlationを実装し、instrumentation off/onのsemantic一致とoverheadを検証する。
- selected caseでHIP API/kernel/copy traceを取得し、critical-path accountingとresidualを作る。
- cold load、warm prefill、decode、API/serviceのうちA1で支配的だった区間を優先する。

### P23-A3: peer differential analysis

- peerの外側wallと利用可能な内部metric/profileを同じcategoryへ正規化し、sLLMとの差をwaterfall化する。
- 固定source/documentationからprovider selection、graph reuse、memory/scheduler/frontend方式の差をtechnical noteへ抽出する。
- differenceがbuild/config、algorithm、kernel、launch/sync、memory、scheduler、frontendのどこに由来するかを反証可能な仮説にする。

### P23-A4: cross-model confirmationとopportunity ranking

- 上位regionだけをGemma 4 dense low-bitとQwen MoEでbounded測定し、共通・dtype固有・architecture固有へ分類する。
- 既存backlogをfresh evidenceと照合し、新規候補、優先度低下、否定、前提変更による復活を記録する。
- Amdahl上限、期待改善、cost/risk/provenance、falsification testを埋め、最大3件のPhase 24候補を順位付けする。

### P23-A5: repeatability、review、closeout

- 上位候補の根拠caseをfresh processで再取得し、run-order drift、health、fallback、trace accountingを確認する。
- affected host/tool checksと1回のintegration reviewを行い、findingだけをfocused re-reviewする。
- bounded summary、comparison manifest、opportunity ledger、negative findings、main plan/historyを同期し、本planをarchiveする。
- Phase 24は自動開始せず、候補と推奨順をユーザーへ提示する。

## 計測lane

| lane | 用途 | performance claim |
| --- | --- | --- |
| X0 | schema、fake clock/span、manifest、aggregator、fallback/cleanup | 不可 |
| X1 | production設定の外側wall、direct/API、3 warmup + 10 measured | E0/E1 classに応じて可 |
| X2 | additive host spansとrequest/kernel correlation | 区間比率のみ。production E2Eへ流用しない |
| X3 | HIP API/kernel/copy trace | device/runtime帰属のみ。instrumented wall比較不可 |
| X4 | 上位区間のGPU counters | 機構の確認のみ。normal wall比較不可 |
| X5 | secondary model bounded confirmation | 共通性分類のみ。全model一般化不可 |

## 完了条件

1. primary Qwen3.5-4B BF16についてV620/R9700のcold、warm prefill、decode、direct/APIを取得し、exact identity、
   health、fallbackなし、sample、metric境界を監査する。
2. llama.cppをprimary peerとしてE0またはE1のmatched comparisonを両GPUで作る。serving controlは少なくとも一つを
   実行比較するか、exact条件を満たせない理由を再現可能に記録してE2 technical comparisonへ限定する。
3. sLLMのE2E critical pathをfrontend、queue/lock、host-active/wait、transfer、GPU semantic category、sampling/response、
   residualへ分け、overlapとobserver effectを明示する。
4. prefill、decode、cold load、serviceのうち少なくとも三領域で、engine差を設定・provider・algorithm・runtime・host処理の
   いずれかへ帰属する。帰属できない差はunknownとして残し、推測でcandidateへ昇格しない。
5. Gemma 4またはQwen MoEの少なくとも一つで上位候補をbounded再確認し、共通候補とmodel/dtype固有候補を分離する。
6. opportunity ledgerが全既存backlog familyを照合し、新規/既知/否定/前提変更、Amdahl上限、coverage、confidence、cost、risk、
   provenance、反証testを持つ。
7. Phase 24へ渡す候補を最大3件に絞る。大候補がない場合はその否定結果と、追加計測で判断が変わる条件を示す。
8. production sourceとdefault、public API、GGUF/model lock、対応model/GPU範囲を変更しない。Phase 23完了を高速化済みと表記しない。
9. raw trace/model/binaryを追跡せず、bounded aggregate、manifest、schema、digest、technical noteだけをGit管理する。
10. integration review、plan/history/main plan/provenance consistencyを完了し、Phase 23をarchiveする。

## 予定deliverable

- cross-engine comparison manifestとE0/E1/E2判定表。
- host span / HIP trace correlation schema、runner、bounded aggregator。
- case別critical-path waterfall、kernel/transfer/host-gap summary、observer-effect report。
- engine technical-difference note。llama.cppと非llama engineのprovenance境界を分離する。
- ranked opportunity ledgerと、既存backlogのreclassification表。
- Phase 24候補最大3件のone-page brief。scope、期待改善、falsification、cost、risk、rollbackを含む。

## 完了結果

- canonical V620/R9700でQwen3.5-4B BF16のwarm direct、256-token prefill、128-token decode、HTTP non-stream/SSE、
  concurrency=2、fresh-process cold loadを取得し、HIP-only、fallbackなし、cleanup 0を確認した。
- 固定llama.cpp peerとの256-token prefillはE1 system-equivalentとし、sLLMはV620/R9700で6.44x/6.60x長かった。
  fresh decodeはtoken列と出力長が異なるためE2に限定し、速度比を作らなかった。
- 最大の新規候補は、generationが最終行だけを使うのにprefillで`[M,vocab]`全行のLM head/argmaxを実行する処理である。
  256-token caseのproduction E2E Amdahl上限はV620 13.06%、R9700 37.92%だった。
- decode matrix family、continuous batching、cold loader pipelineを既存backlogから再確認した。HTTP/SSE framingは約0.5〜0.6 msで
  lower-priorityとした。Gemma 4 R9700 controlではdevice timeの83.67%がmatvecで、matrix familyのmodel横断性を確認した。
- Phase 24 shortlistを`P23-O1` prefill last-row projection、`P23-O2` projection-family fusion/shared load/plan replay、
  `P23-O3` continuous batchingの順に固定した。Phase 23では候補を実装していない。
- bounded result、全候補ledger、backlog再分類、digestは
  [集計JSON](../../../../../../ci/matrix/phase23-performance-discovery-summary-v1.json)、engine差は
  [technical note](../../../../../references/phase23-inference-engine-performance-differential.md)を正とする。raw trace/model/binaryは追跡しない。

## Rollback・停止・再計画

- instrumentationがtoken、stop、dispatch、state publication、cleanupを変えた場合はprofileを採用せず、additive span境界を縮小する。
- profiler overheadでcritical pathを分類できない場合はraw trace量を増やし続けず、外側wall、host spans、kernel microcontrolを
  別processへ分ける。
- peerがCPU fallback、partial offload、異なるdtype/cache/batchを使った場合はE0/E1から外し、比較可能なengineまたはE2へ切り替える。
- broad scan後はdeep profileを最大3 regionに限定する。同じunknown原因が2回の追加計測でも解けない場合はunknownとして残す。
- review時間が実装時間超、functional progressが1時間停止、verification/docsがwork unitの30%超、見積り1.5倍超、
  metric/acceptance変更時は同じwork unitを止めて再計画する。
- Phase 23中に有望candidateが見つかっても実装へ進まない。計測契約とranked inventoryを閉じてからユーザーへ次Phaseを提案する。

[対応する履歴](../../../../../history/2026/08/11-20/phase23-cross-engine-differential-performance-discovery.md)

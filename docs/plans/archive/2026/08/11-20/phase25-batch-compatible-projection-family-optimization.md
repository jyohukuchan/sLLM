# Phase 25: batch-compatible projection-family optimization

> 状態: 完了（A0 negative discovery、production candidateなし）
> 作成日: 2026-08-18

## 目的

Phase 23の`P23-O2`を、Phase 24後のcurrent sourceで再計測してから限定実装する。対象は単独のBF16 `M=1`
matvec variantではなく、同じactivationを消費するprojection family、producer/consumer境界、prepared planを一つの
critical-path unitとして扱う最適化である。

Phase 26のリクエストバッチ処理でdecode row数が`M=1`から`M=B`へ変わっても再利用できるdescriptor、layout、provider
selectionを優先する。Phase 25は複数requestのscheduler、per-sequence state、KV block tableを実装しないが、synthetic
`B=1,2,3,4,7,8,16,17`でcandidateの適用範囲とprovider crossoverを固定し、`M=1`だけへ閉じたABIを追加しない。

## 開始根拠

- Phase 23ではQwen3.5-4B BF16の通常decodeがV620 32.43 tok/s、R9700 36.99 tok/sで、projection familyを
  fusion/shared load/plan replayとして扱う`P23-O2`の期待E2E改善を8〜20%とした。
- Phase 22のsingle-shape wave32x8 candidateはV620 gate/up operatorを約32%短縮したが、down、R9700、最終wallで相殺され、
  full-modelは0.52%悪化した。従って一shapeの局所勝利を採用根拠にしない。
- Phase 24はterminal rowsを削減したが、R9700では`M>1` hipBLAS GEMMから`M=1` decode reductionへのprovider遷移により
  P2 E2E改善が0.49%に留まった。FLOP削減だけでなくweight traffic、arithmetic intensity、provider crossoverを測る必要がある。
- 現行runtimeはsemantic descriptor、prepared cache、target/shape別kernel registryを持つため、公開APIを変えずに
  multi-output familyまたはcoarse planを内部実装できる。

## 権限とproposal status

- この文書はユーザーの詳細計画作成指示に基づく。Phase番号と順序は割り当てるが、production sourceの変更、GPU再実行、
  採用thresholdのhard gate化は、ユーザーがPhase 25開始または本planを明示承認した時点で行う。
- 下記thresholdは現時点ではnonblocking AI proposalである。originはPhase 23 `P23-O2`とPhase 22/24のnegative finding、
  scopeはQwen3.5-4B BF16 projection family、costはdual-GPU profile/oracle/full-model比較、expiryはP25-A0のfresh profile完了時とする。
- correctness/security境界は性能proposalとは別である。wrong output、state corruption、fallback、cross-request alias、範囲外view、
  cleanup failureを性能で相殺しない。

## Primary scope

- model: Qwen3.5-4B dense BF16、GGUF、通常text generation。
- target: canonical V620 exact `gfx1030`、R9700 exact `gfx1201`。
- workload: normal greedy decodeを主対象とし、sampled decodeとnormal prefillを回帰controlにする。
- family候補:
  - Q/K/V projectionの共通activation inputとdispatch/plan境界。
  - MLP gate/up projection、SiLU/multiply、down projectionのproducer/consumer境界。
  - final RMSNorm、wide-vocabulary projection、Argmaxのdecode terminal family。
  - reusable prepared descriptor/bindingと、複数opをまとめるcoarse plan/command-list。
- provider selection keyは少なくともexact target、physical dtype/encoding、projection role、`M/K/N`、layout、alignmentを含める。
- graph/model adapterは共通semantic pathを維持する。exact target固有kernel/providerは、共通pathの下で再現可能な性能差を
  解消する場合だけregistry dispatchとして分け、Qwen execution flowをtarget別に複製しない。

## Secondary scope

- Qwen3.5 MoEのBF16 shared expert、Gemma 4のBF16 projectionは、P25-A0で同じsemantic familyとproviderを共有し、
  candidate regionがdevice timeの10%以上を占める場合だけhost contractまたはbounded GPU controlへ含める。
- `B>1`はsynthetic operator/layout/capability evidenceに限定する。複数requestのKV/GDN/stateを実際にまとめるproduction executionは
  Phase 26が所有する。
- llama.cppの直接reuseを選ぶ場合は、実装前にexact revision/path/hash/licenseをprovenance recordへ追加する。
  vLLM/SGLang等はfacts-only/no-copyとし、inspection noteと実装を分離する。

## 非対象

- scheduler、waiting/running queue、continuous request batching、chunked prefill、request fairness、HTTP transport変更。
- KV block table、paged KV、prefix cache、cross-request state sharing、multi-stream scheduling。
- attention/GDN algorithm、MTP speculation algorithm、MoE grouped GEMM、GPU sampling。
- weight/KV quantization、TurboQuant、DeepSeek V4、新model format、GGUF再pack。
- targetごとに別graphを維持する実装、runtime全体のJIT/autotuning DB、requestごとのHIP Graph instantiate。
- Phase 22 candidateの無計測復元、Phase 21 event集約だけの再提案、局所kernel改善によるfull-model claim。

## Candidateの選定規則

P25-A0でcurrent sourceをprofileし、次の式と境界で最初のwork unitを一つだけ選ぶ。

`priority = production critical-path share × credible removable fraction × model/shape coverage ÷ implementation risk`

1. 同一activationを読む2個以上のprojectionが連続し、family全体でdecode device timeの15%以上ならmulti-output familyを第一候補とする。
2. semantic prepare/binding/submitがfamily wallの10%以上ならcoarse prepared plan/replayを第一候補とする。
3. wide-vocab `M=1`がdecode E2Eの5%以上で、`gfx1201`の`M=1/M>1` provider差が再現する場合はterminal providerを候補にする。
4. 上記を満たさなければ実装を開始せず、profile結果でPhase 25を否定完了または再計画する。

一つ目のcandidateがoperatorでは改善してもfull-modelで棄却された場合、同じwork unitへ別kernel、fusion、plan replayを足して
救済しない。P25-A0で独立順位を持つ次候補へ進むには、最初のwork unitを閉じてscopeと残り見積りを再確認する。

## Semantic・layout contract

- family opは個々のprojectionと等価なordered outputsを返す。Q/K/V、gate/up等のoutput ID、shape、stride、dtype、encoding、
  scale、bias有無を明示し、暗黙のtensor順序に依存しない。
- fusion前後でlogical accumulation dtype、activation function、scale適用、rounding/output dtypeを固定する。
  数値順序を変更するcandidateはbaseline bit-exactを前提にせず、事前固定したf64/FP32 oracle toleranceとtoken-level oracleで判定する。
- input/output alias、overlap、zero shape、overflow、非contiguous/unaligned view、unsupported encodingはprepare前またはprepare時に
  fail closedとする。fallbackによる別backend/CPU実行は認めない。
- `M`はbatch sizeを意味し得るが、Phase 25のproduction request ownerは一つのままにする。descriptorへrequest ID、KV pointer、
  sampling stateを混在させない。
- prepared identityはfamily kind、全tensor view/buffer identity、access mode、`M/K/N`、role、target、binding generationを含める。
  異なる`M`、target、weight、output orderを同じcache entryとして再利用しない。
- candidateが新しいcoarse command-listを使う場合も、state publication、terminal readback、cancellation、error boundaryを跨いで
  completionを隠さない。

## 提案する受入基準

### Correctness

1. host descriptor/layout testは`M=1,2,3,4,7,8,16,17`と、各候補の実`K/N`、`K/N`境界前後、unaligned tiny shapeを含む。
2. tiny GPU oracleを両targetで実行し、各family outputを独立f64またはFP32 referenceへ照合する。非有限分類、top-1、
   output order、padding非破壊を検証し、CPU fallback、timeout、crash、zero selectionをPASSにしない。
3. Qwen full-model greedyはprompt/completion token IDs、stop、usage、visible output、KV/GDN length、dispatch audit、cleanupを
   baseline/candidateで一致させる。
4. samplingは固定seedのtemperature/top-p/penalty 3 profileでtoken列とstop/usageを一致させ、必要なlast logitsを固定toleranceで比較する。
5. prefill 17/255/256/257、normal decode、MTP target-only、明示all-logitsの既存contractを壊さない。
6. provider不適合、prepared-cache collision、partial submission、cancellationではoutput/stateを公開せずrequest ownerをfail closedにする。

### Performance・resource

7. production採否はprofilerなし、同一binary/model/request、3 warmup + 10 measured以上、baseline/candidateをcounterbalanced順で取得する。
8. primary D0は28 input / 128 output。補助として17/2、255/2、256/2、257/2、decode 32/256を固定し、TTFT、TPOT、
   decode token/s、E2E、request/workspace high-waterを記録する。
9. 固定した全target/patternでstableなE2Eまたはprimary span悪化を残さず、任意のtarget/patternでfull-model E2EまたはTPOTを
   5%以上改善した場合にproduction candidateを採用する。局所operatorだけの5%は採用条件を満たさない。
10. 0〜2%の負差はbaseline/candidateを再度挟み、最終bracketで非悪化を確認する。一回でも2%超悪化したcaseは原因を解消するか棄却する。
11. mechanism proofはfamily dispatch数、kernel/provider identity、device duration、weight/activation bytes、workspace、prepared hit率を示す。
    profiler wallをproduction改善値へ使わない。
12. synthetic `B=1,2,3,4,7,8,16,17`でprovider crossoverとthroughput/rowを記録する。`B>1` microcaseはPhase 26の
    production throughput claimではなく、batch-compatible contractの証拠に限定する。
13. model-resident bytesを増やさず、request/workspace high-waterをbaseline比1%超増やさない。必要な有限workspaceは明示し、
    targetごとの無制限algorithm cacheを作らない。

### Evidence・closeout

14. source/tree、binary、runner、oracle、ROCm、target、GPU identity、model/derived lock、prompt、sampling、healthをdigest-boundにする。
15. raw model、binary、full logits、generated全文、rocprof traceを追跡せず、schema、bounded aggregate、digest、runner、plan/historyだけをcommit対象にする。
16. affected host/build/GPU tests、1回のintegration review、findingだけのfocused re-review、main plan/history/provenance同期を行う。
17. candidateが基準未達ならproduction defaultへ残さず、negative resultと次に判断を変える条件を記録してPhase 25を完了できる。

## 計測matrix

| case | shape/workload | 目的 | 採否への使用 |
| --- | --- | --- | --- |
| H0 | family host shapes `M=1,2,3,4,7,8,16,17` | descriptor、layout、cache identity | correctness |
| G0 | tiny distinctive tensors、両GPU | family numerical oracle | correctness |
| O0 | actual QKV shape、`M=1..17` | multi-output/shared-load crossover | mechanism |
| O1 | actual gate/up/down shape、`M=1..17` | MLP family crossover | mechanism |
| O2 | `K=2560,N=248320`、`M=1..17` | wide-vocab provider | mechanism |
| D0 | Qwen 28 / 128 greedy | primary decode wall/TPOT | hard performance |
| D1 | Qwen 32 / 256 greedy | longer steady decode | hard regression |
| P0 | Qwen 17 / 2 | short prefill regression | hard regression |
| P1/P2/P3 | Qwen 255/256/257 / 2 | Phase24 boundary regression | hard regression |
| S0 | 3 fixed sampling profiles | logits/sampling保持 | correctness |
| M0 | MTP target-only control | state/terminal contract保持 | correctness |

## 作業順序

### P25-A0: current profileとcandidate freeze

- Phase 24 final binary/source/model lockをbaseline identityとして固定する。
- D0/D1/P0/P1/P2/P3のproduction wallと、QKV、MLP、terminal familyのdevice/host spanを両GPUで再取得する。
- `M=1..17` microprofileでprovider、roofline proxy、dispatch数、prepared overheadを分離し、candidateを一つ選ぶ。
- candidate、tolerance、case、採用threshold、Gemma/MoE secondary gateを実装前にmanifestへ固定する。

### P25-A1: batch-compatible host contract

- typed multi-output familyまたはcoarse plan descriptor、checked layout、provider key、prepared identityをhost testで先に固定する。
- `M=1`を特別扱いして公開ABIへ埋め込まず、`M>1`と非整列境界を同じcontractで表現する。
- failure、alias、overflow、cache collision、boundary crossingをfake backend/stubで検証する。

### P25-A2: bounded candidate implementation

- A0で選んだ一familyだけを実装する。baseline providerは比較・rollback用にregistryへ保持する。
- graph adapterはtarget-neutralなfamily opをlowerし、exact target差はkernel registryのprovider selectionへ閉じ込める。
- fusion/planへ含めないstateful boundaryは既存submission順を維持する。

### P25-A3: dual-GPU numerical・mechanism proof

- G0とO0/O1/O2のうち該当familyを両GPUで実行し、数値、provider identity、fallbackなし、cleanupを確認する。
- actual shapeでdispatch削減、shared activation、prepared reuse、device durationのどれが成立したかを分類する。
- operatorで改善しない、または一targetで再現する悪化がある場合はfull-model採否へ進まずcandidateを修正または棄却する。

### P25-A4: full-model採否

- D0/D1/P0/P1/P2/P3とS0/M0をbaseline/candidate交互順で実行する。
- 全pattern非悪化かつ任意pattern full-model 5%以上のproposal criteriaで採否し、target splitは共通pathに再現する問題が残る場合だけ検討する。
- 採用後もPhase 26用`B>1`結果をperformance claimへ混ぜず、batch-compatible capabilityとして記録する。

### P25-A5: integration・closeout

- affected checks、schema、Markdown/provenance、one integration reviewを完了する。
- main plan/historyを結果へ同期してplanをarchiveする。Phase 26はPhase 25の採否にかかわらず、stable final source identityから開始できる。

## Rollback・停止・再計画

- wrong output、token/state差、fallback、prepared collision、cleanup failureはcandidateを採用しない。
- operator改善がfull-modelへ転化しない場合は、Phase 22と同様にcandidateを除去してnegative resultを残す。
- `M=1`改善が`M>1`を悪化させる場合、共通semantic pathを維持したprovider selectionで解消できなければcandidateを棄却する。
- 最初のcandidateへ量子化、attention、request batching、複数fusion familyを追加して救済しない。
- 同じwork unitが2回reject、review時間が実装時間超、functional progressが1時間停止、verification/docsが30%超、
  見積り1.5倍超、acceptance変更時は追加実装・検証を止め、同じwork unitを再計画する。

[Phase 23 bounded summary](../../../../../../ci/matrix/phase23-performance-discovery-summary-v1.json)
[Phase 24 bounded summary](../../../../../../ci/matrix/phase24-terminal-row-summary-v1.json)
[Phase 25 bounded summary](../../../../../../ci/matrix/phase25-projection-family-summary-v1.json)
[対応する履歴](../../../../../history/2026/08/11-20/phase25-batch-compatible-projection-family-optimization.md)

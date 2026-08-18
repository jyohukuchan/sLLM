# Phase 27: exact decode projection weight-stream/provider optimization

> 状態: 完了（negative discovery、production変更なし）
> 作成日: 2026-08-18
> 完了日: 2026-08-18

## 目的

Phase 23で残ったQwen3.5-4B BF16のllama.cppに対するhistorical exact-token decode差を、Phase 24後のcurrent sourceと
固定llama.cppでfreshに再現する。通常decode `M=1`のQ/K/V、MLP gate/up/down、linear/GDN、terminal LM headについて、
必須weight streamと演算を同じ仕事量でどれだけ効率よく処理できているかをtarget/shape/provider別に分解する。

Phase 25が否定したのはprojection-family間のshared activation、launch除去、plan replayだけであり、projection kernel/provider自体が
llama.cppと同等であることは証明していない。Phase 27はこの未確認部分を扱い、fresh matched gapへ最大寄与するwork unitを一つだけ
実装・採否する。gapを再現できない、または5% full-model改善へ届くcandidateがない場合も、比較結果と反証条件を残してnegative
completionできる。

## 開始根拠

- Phase 23のfresh sLLM long decodeはV620 32.43 tok/s、R9700 36.99 tok/sだった。fresh llama.cpp laneはtoken列と出力長が
  異なりE2 diagnostic-onlyだったため、current sourceの勝敗ratioは確定していない。
- Phase 23が参照したhistorical exact 17/17 laneではsLLM throughputは固定llama.cppのV620 72.4%、R9700 71.2%だった。
  約28〜29%のthroughput residualが現在も存在するかをfresh E0で検証する必要がある。
- Phase 25のfresh profileではdecode projection device shareがV620 86.48%、R9700 79.23%だった。一方、gate/upで共有できる
  activationはweight trafficの0.00543%、launch完全除去の楽観上限もTPOTの0.94%/2.60%で、P23-O2方式はcandidateなしとなった。
- Phase 22はV620 gate/up operatorを約32%短縮したが、down、R9700、full-model wallで相殺された。単一shapeの局所改善を
  provider全体やproduction改善へ一般化せず、全shapeのgap contributionからwork unitを選び直す必要がある。
- `sLLM.md`とmain planは、最適化済み単一requestを同一条件のllama.cppより高速にすることを一つの基準としている。

## 権限とproposal status

- 本文書はユーザーのPhase 27詳細計画作成指示に基づく。Phase番号、scope、作業順を割り当てるが、production source変更、
  GPU再実行、llama.cpp再build、performance thresholdのhard gate化はPhase 27開始の明示指示後に行う。
- 「全固定target/patternで悪化なし、任意target/patternでfull-model 5%以上改善」はユーザーがPhase 24で決定し、Phase 27にも
  明示的に引き継いだ採用規則として扱う。correctness、fallback、cleanup、resource boundaryは性能で相殺しない。
- model-resident/request workspace 1%とmodel-ready 5%はAIが提案するnonblocking resource guardである。originはinternal repackの
  duplicate weight/startup regression risk、scopeはlayout/repack candidate、costはR0計測、expiryはP27-A2 candidate freezeとする。
  超過時は自動棄却せず、性能利益とresource costを分けてユーザー判断へ返す。
- trusted-solo-developmentを維持する。llama.cppの直接reuseは許可されるが、実装前にexact revision/path/blob hash、reuse mode、
  license、変更点をprovenanceへ記録する。vLLM/SGLang等はfacts-only/no-copyとする。

## Primary scope

- model: Qwen3.5-4B dense BF16 GGUF、revisionとGGUF digestを固定する。
- target:
  - V620 exact `gfx1030`、UUID/BDFをevidenceへ固定する。
  - R9700 exact `gfx1201`、UUID/BDFをevidenceへ固定する。
- execution: warm、single request、batch 1、通常text greedy decode、MTP off、同一KV/cache policy。
- primary comparison: current sLLMと固定llama.cppのdirect exact-token replay。API、tokenizer、sampling差をprimary wallへ混ぜない。
- projection roles:
  - full attention Q/K/V/O。
  - linear/GDN attentionのpacked projectionとrecurrent output projection。
  - MLP gate/up/down。
  - final RMSNorm後のwide-vocabulary LM headとterminal reduction。
- candidate boundary:
  - exact target、`M/K/N`、role、layout、alignmentに基づくprovider selection。
  - weight vector load/coalescing、wave/workgroup、tile/reduction、prefetch/cache利用。
  - logical tensorを変えない内部weight layoutまたは起動時repack。ただしduplicate resident weightを恒久保持しない。
  - accumulation/output dtype、rounding、state publicationを維持するbounded epilogueまたはprovider変更。

## Secondary scope

- Qwen3.5-4Bのnormal sampled decodeをcorrectness controlに使う。性能採否はgreedy primaryで行う。
- Gemma 4 BF16/mixed-low-bitやQwen MoEは、同じBF16 providerとexact shapeを共有する場合だけhost/provider contractのcompile controlへ
  含める。Phase 27の性能claimを別dtype/modelへ一般化しない。
- llama.cppのweight layout/provider source inspectionは、fresh E0 gapへの因果仮説を作るbounded referenceとして使う。
  source上の類似だけでcandidateを採用しない。

## 非対象

- projection-family shared activation/fusion、launch-plan replayだけによるP23-O2の再実施。
- Phase 22 wave32x8 candidateの無計測復元、単一shapeのoperator勝利だけによる採用。
- prefill、Phase 24 terminal-row path、chunked prefill、continuous request batching、multi-sequence KV/GDN ABI。
- attention/GDN algorithm変更、MTP algorithm、GPU sampling、prefix/KV cache、multi-GPU。
- TurboQuant、DeepSeek V4、FP8/NVFP4/MXFP4、新model format、GGUF公開format変更。
- runtime event集約、HIP Graph、H2D/D2H統合、cold loader。計測でこれらがgapの主因ならPhase 27を拡張せず別候補へ戻す。
- targetごとのQwen graph/model adapter複製。必要なtarget差は共通semantic path下のprovider registryへ閉じ込める。

## Fresh E0比較契約

### Identity

- sLLM source/tree、release binary、HIP kernel/provider、ROCm、compiler、build flagsをdigest-boundにする。
- llama.cppはPhase 23で固定したcommit `f5919bf458ef190468b5c329bb293f8a54a1e69c`を最初のpeerとし、source、binary、
  CMake flags、HIP target、Flash Attention等の実build coverageを再確認する。比較不能なbuildなら速度ratioを作らない。
- model revision、logical tensor dtype/shape、tokenizer revision、RoPE/context設定、KV dtype、GPU offload、batch/cache条件を一致させる。
  engine内部artifact layout差は記録し、kernel単体差とsystem差を混同しない。
- CPU fallback、partial GPU offload、GTT spill、timeout、crash、zero sample、別targetはPASSにしない。

### Exact-token replay

- primary laneはtokenizer/render/samplerを外し、同じprompt token IDsと固定continuation token IDsを両engineへ一stepずつ入力する。
  各stepは同じposition、logical KV length、token historyを持ち、teacher-forced replayのpost-prefill decode wallを測る。
- logits取得境界を揃え、stepごとのtop-1、finite分類、selected logitsまたはbounded logit sliceを固定toleranceで比較する。
  数値差で次入力tokenが分岐しても、固定continuationを使うperformance replayは同じ仕事列を維持する。
- secondary generation laneは同じgreedy token列が得られるcaseだけE0 full-generation ratioを許可する。token列、stop、出力長が
  異なるcaseはE2へ降格し、勝敗ratioを作らない。

### Timing

- production wall laneはprofilerなし、同一GPU/model/token sequence、3 warmup + 10 measured以上、engine順をcounterbalancedにする。
- TTFT/prefillをprimary decode wallから除き、first decode input readyから各step completionまでのTPOT、decode tok/s、p50/p90/max、
  MADを記録する。固定token数を完走したmakespanを正とし、最速stepだけを採用値にしない。
- CLI/APIはsecondary system controlとし、direct replayとの差からfrontend/transport residualを分類する。
- profiler/counter laneは別process/runで取得し、observer effectを含むinstrumented wallをproduction改善値へ使わない。

## Critical-path・weight-stream accounting

各projection role/shapeについて、少なくとも次をbaseline manifestへ記録する。

- dispatch count、kernel/provider identity、`M/K/N`、grid/workgroup、wave、input/output/weight dtypeとlayout。
- logical mandatory weight bytes、activation/output bytes、FLOPs、kernel duration、family/全decode device share。
- `mandatory weight bytes / kernel duration`によるeffective weight-stream rate。これはDRAM counterではなくroofline proxyと明記する。
- top gap familyではhardware counterから実DRAM/L2 traffic、cache hit、VALU/VMEM instruction、occupancy、wave stallを取得する。
- kernel間host gap、HIP API、H2D/D2H、completion wait、GPU idleを別categoryにし、projection kernelへ誤帰属しない。
- llama.cpp側も対応role/shape/providerを可能な範囲で同じ分類へ写像し、engine固有fusionで一対一対応しない区間はfamily wallで比較する。

`gap contribution = sLLM decode wall share × matched family slowdown × credible fixable fraction`

単純なprojection shareではなく上式で候補を順位付けする。最低95%のwall accountingを目標にするが、未帰属分が残る場合は
大きさ、位置、追加計測costを明示し、confidenceを下げる。

## Candidate freeze規則

P27-A2終了時に、次の優先順で最初のwork unitを一つだけ固定する。

1. 同じlogical shape/familyでllama.cppまたは既存sLLM providerより再現可能に遅く、provider selectionだけで5% full-model上限へ
   届く場合はshape/target-aware provider dispatchを選ぶ。
2. mandatory bytesは同じでもeffective weight-stream rateが低く、counterがuncoalesced load、vector幅、wave、occupancy、cacheの
   いずれかを示す場合は、最大gap familyのkernel mappingを選ぶ。
3. weight layout/repack差が支配的なら、GGUF論理formatを変えず、existing resident weightを置換でき、startup/VRAM条件を満たす
   bounded internal layoutを選ぶ。
4. memory trafficではなくreduction/convert/epilogue instructionが支配的なら、accumulationとoutput contractを維持する範囲で
   compute pathを選ぶ。
5. fresh E0 gapが5%未満、projection外が主因、または最大候補のAmdahl上限が5%未満なら実装せずnegative completionする。

同じcandidateがoperatorでは改善してもfull-modelで棄却された場合、別wave、fusion、layout、target splitを追加して救済しない。
独立順位を持つ次候補へ進むにはwork unitを閉じ、残り見積りとacceptanceを再確認する。

## 受入基準

### Correctness

1. host descriptor/provider testはactual Q/K/V/O、gate/up/down、LM-head shapeと、`M=1,2,3`、`K/N`境界前後、zero、overflow、
   unaligned/noncontiguous viewを含む。Phase 27のproduction採用は`M=1`だが、公開contractをunsafeなM1固定にしない。
2. tiny GPU oracleを両targetで実行し、baseline/candidateをf64またはFP32 referenceへ照合する。finite分類、top-1、padding非破壊、
   output order、unsupported layout fail-closedを確認する。
3. exact-token replayで各stepのtoken/position、KV/GDN committed length、bounded logits、provider audit、cleanupを一致させる。
4. Qwen full-model greedyはprompt/completion token IDs、visible output、stop、usageをbaseline/candidateで一致させる。
5. fixed seedのsampling 3 profile、short prefill、255/256/257 terminal-row境界、MTP target-only、明示all-logitsを回帰controlにする。
6. provider不適合、partial submission、cancellation、timeout、query failureではoutput/stateを公開せず、CPU/backend fallbackしない。

### Performance・resource

7. cross-engine gapはE0-D0/D1/D2、production採否はF0/F1/F2のprofilerなしTPOT、decode tok/s、post-TTFT E2Eを正とする。
   operator/counter値またはteacher-forced replayだけではproduction採用しない。
8. 固定した全target/patternでstableな悪化を残さず、任意target/patternでfull-model decode TPOTまたはpost-TTFT E2Eを5%以上
   改善したcandidateだけをproductionへ採用する。
9. 0〜2%の負差はbaseline/candidateを再度挟み、最終bracketで非悪化を確認する。一回でも2%超悪化したcaseは原因を解消するか
   candidateを棄却する。
10. historical exact throughput ratio 0.724/0.712は参考値に限定し、fresh E0だけでcurrent llama.cpp gap、gap closure、勝敗を主張する。
11. nonblocking resource proposalとしてmodel-resident/request workspace +1%、fresh model-ready +5%をguardにする。internal repackは
    元weightを解放して置換する。guard超過はcorrectness blockerにせず、利益とresource costを別判断としてユーザーへ返す。
12. shared semantic/graph pathを維持する。gfx1030/gfx1201差は共通provider registryのselectionで吸収し、共通candidateが重大な
    再現悪化を解消できない場合だけtarget-specific providerを許容する。

### Evidence・closeout

13. source/tree、binary、runner、oracle、ROCm、GPU UUID/BDF、model/derived lock、token sequence、llama.cpp identity、build flags、
    counter toolをdigest-boundにする。
14. raw model、binary、full logits、生成全文、raw trace/counter dumpを追跡しない。bounded aggregate、digest、schema、runner、
    plan/history/provenanceだけをcommit対象にする。
15. affected host/build/GPU/full-model test、one integration review、changed findingだけのfocused re-review、main plan/history/runtime/
    provenance同期を行う。
16. candidateなしまたは基準未達ならproduction defaultへ残さず、fresh exact gap、最大寄与区間、反証された仮説、次に判断を
    変える条件を記録してPhase 27を完了できる。

## 計測matrix

| case | workload | 目的 | 採否への使用 |
| --- | --- | --- | --- |
| H0 | actual shape + boundary host cases | descriptor/provider/layout | correctness |
| G0 | tiny distinctive projection tensors、両GPU | numerical/provider oracle | correctness |
| E0-D0 | 17 prompt + 128 fixed continuation | historical exact laneのfresh replacement | hard performance |
| E0-D1 | 28 prompt + 128 fixed continuation | Phase 23 long-decode continuity | hard performance |
| E0-D2 | 255 prompt + 128 fixed continuation | larger KV control、projection share確認 | hard performance |
| F0 | sLLM normal greedy 17 / 128 | production short-context candidate採否 | hard performance |
| F1 | sLLM normal greedy 28 / 128 | production Phase 23 continuity | hard performance |
| F2 | sLLM normal greedy 255 / 128 | production larger-KV regression | hard performance |
| O0 | Q/K/V/O actual shapes | attention projection gap | mechanism |
| O1 | gate/up/down actual shapes | MLP weight-stream gap | mechanism |
| O2 | `K=2560,N=248320` LM head | wide-vocab reduction/provider gap | mechanism |
| C0 | top one or two gap families + counters | DRAM/L2/VMEM/occupancy/stall | mechanism |
| S0 | 3 fixed sampling profiles | logits/RNG/stop regression | correctness |
| P0 | prefill 17/255/256/257 + output 2 | Phase 24 regression | correctness/performance |
| M0 | MTP target-only control | state/terminal regression | correctness |
| R0 | model-ready and resident/workspace high-water | repack/resource regression | resource |

## 作業順序

### P27-A0: identity・runner・acceptance freeze

- Phase 24採用、Phase 25 negative、Phase 26 production未接続のcurrent sourceをbaseline identityとして固定する。
- fixed llama.cpp source/build/model/targetを再確認し、exact-token replay runnerとtoken-sequence digestを作る。
- E0-D0/D1/D2、F0/F1/F2、tolerance、timing boundary、counter tool、採用規則、resource proposalをbaseline測定前manifestへ固定する。
- local Qwen serviceがV620 pairを占有している場合は停止し、single-V620 fallbackを使わずcanonical GPU identityを確認する。

### P27-A1: fresh dual-GPU E0 baseline

- sLLM/llama.cppを同じtarget向けにfresh buildし、E0-D0/D1/D2をcounterbalanced順で両GPU取得する。
- token/state/logit oracle、HIP-only、fallbackなし、GTT spillなし、cleanup terminal-zeroを確認する。
- historical 0.724/0.712 gapが再現、縮小、消失、逆転のどれかを分類する。fresh条件が一致しなければE1/E2へ降格して
  current勝敗を主張しない。

### P27-A2: role/shape別critical-path differential

- production wallとは別のprofileでO0/O1/O2をrole/shape/providerへ分解し、両engineのfamily wallを対応付ける。
- 最大gap familyだけC0 counterを取得し、mandatory bytes、effective stream rate、実traffic、cache、VMEM、occupancy、stallを照合する。
- llama.cppの対応provider/layout/build pathをbounded inspectionし、差をprovider selection、kernel mapping、layout、compute、host gap、
  engine固有fusion、比較不能へ分類する。
- gap contributionとAmdahl上限からcandidateを一つ固定する。5%へ届かなければA3以降へ進まずnegative closeoutする。

### P27-A3: host contract・bounded numerical oracle

- candidateのprovider key、layout、alignment、workspace、prepared identity、failure contractをhost testで先に固定する。
- G0をbaseline providerとcandidate providerで両GPU実行し、非整列値と境界両側を含む数値oracleを通す。
- llama.cpp direct reuseを選ぶ場合は実装前にprovenance recordを追加する。独立実装ならreference inspection noteとsourceを分離する。

### P27-A4: one-candidate implementation

- A2で選んだ一family・一原因だけを実装する。baseline providerは比較・rollback用に維持する。
- Qwen graphは共通semantic descriptorをlowerし、target/shape差はprovider registryで選択する。
- accumulation、rounding、output layout、completion、transactional state publication、cancellation境界を変更しない。

### P27-A5: dual-GPU mechanism・full-model採否

- O0/O1/O2の該当familyでprovider identity、kernel duration、effective stream rate、fallbackなしを確認する。
- E0-D0/D1/D2でgap closureを確認し、F0/F1/F2、S0/P0/M0/R0をsLLM baseline/candidate counterbalanced順で実行する。
- 全pattern非悪化かつ任意pattern 5%以上の規則で採否する。target splitは共通registry selectionで重大な問題を解消できない場合だけ
  検討し、model graphを複製しない。

### P27-A6: integration・closeout

- affected checks、schema、Markdown/provenance、one integration review、findingだけのfocused re-reviewを完了する。
- bounded summaryにfresh exact engine ratio、critical-path accounting、candidate mechanism、採否、resourceを記録する。
- main plan/runtime/historyを結果へ同期し、planをarchiveする。次phaseを自動開始しない。

## 完了結果

- current sLLMの28-token prompt / 128-token greedy decodeはV620 32.38 tok/s、R9700 37.00 tok/sだった。固定llama.cpp
  `f5919bf`の128-token decode-onlyは48.94/53.96 tok/sでgapは残った。ただしGGUF bytes、token列、timing boundaryが一致しないため
  比較はE1 system-equivalentに限定し、engine勝敗またはE0 exact claimには使わない。
- fresh runtime profileではmandatory projection 8,409,579,520 bytes/tokenに対し、V620のprojection device timeは
  sLLM 17.71 ms、llama.cpp 18.99 msでsLLMが6.76%短かった。R9700は17.85/15.86 msでsLLMが12.53%長く、peer相当まで
  短縮できた場合のproduction TPOT上限は約7.36%だった。
- projectionを除くcoarse residualはsLLM/llama.cppがV620 5.38/1.41 ms、R9700 5.26/1.49 msだった。ただしsLLM側は
  prefill projectionだけを除いており、prefillのGDN/attention/normとR9700のMTP内部workが残る。llama.cppともstep境界が
  一致しないためdecode-only比率にはせず、projection外device候補を再計測する探索根拠に限定する。
- Phase 22の同系provider candidateはV620 operatorを短縮してもR9700を約13%悪化させ、V620 full-modelも0.52%悪化した。
  Phase 25で棄却済みのgate/up fusionを再実施せず、共通経路で全target非悪化かつ任意pattern 5%改善へ届くprojection-only
  candidateはないと判定した。
- P27-A3〜A5は`NO_COMMON_PROJECTION_CANDIDATE`で閉じ、production kernel/default、target split、resource使用量を変更していない。
  bounded結果は[Phase 27 summary](../../../../../../ci/matrix/phase27-weight-stream-summary-v1.json)を正とする。

## Rollback・停止・再計画

- wrong token/logit/state、fallback、GTT spill、prepared collision、cleanup failure、unbounded workspaceはcandidateを採用しない。
- fresh E0が成立しない場合はE1/E2結果を速度勝敗へ変換せず、比較runnerまたはartifact条件を再計画する。
- projection外のhost gap、transfer、attention/GDN、build coverageがgapの主因なら、Phase 27へ別最適化を混ぜず原因を別候補として返す。
- operator改善がfull-modelへ転化しない場合はPhase 22同様にcandidateを棄却する。複数kernel/fusion/layoutを足して救済しない。
- target差はまずprovider selectionで扱う。gfx1030/gfx1201のgraph path分岐はcorrectnessまたは重大な再現性能問題なしに追加しない。
- 同じwork unitが2回reject、review時間が実装時間超、functional progressが1時間停止、verification/docsが30%超、
  見積り1.5倍超、acceptance変更時は追加実装・検証を止め、同じwork unitを再計画する。

[Phase 23 bounded summary](../../../../../../ci/matrix/phase23-performance-discovery-summary-v1.json)
[Phase 25 bounded summary](../../../../../../ci/matrix/phase25-projection-family-summary-v1.json)
[Phase 27対応履歴](../../../../../history/2026/08/11-20/phase27-exact-decode-projection-weight-stream-provider-optimization.md)

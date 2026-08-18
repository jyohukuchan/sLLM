# Phase 28: decode projection外device処理の限定最適化

> 状態: 完了（明示例外採用）
> 作成日: 2026-08-18

## 目的

Qwen3.5-4B dense BF16の通常decodeについて、projection以外のGPU device処理をcommitted output token単位で正確に分離し、
canonical V620 `gfx1030`とR9700 `gfx1201`で最大寄与を持つ共通またはstable adoption scope固有familyを限定最適化する。

Phase 27で算出したV620 5.379 ms、R9700 5.257 msというprojection除外残差は、全kernel aggregateからprefillの
projectionだけを引いた値である。prefill中のGDN、attention、normalization等と、R9700のMTP内部stepが残っているため、
純粋なdecode projection外device timeではない。llama.cpp側ともstep境界が一致せず、3.80倍/3.54倍という比率は採用根拠に
できない。Phase 28はこの値を探索仮説へ降格し、最初に境界を修正してから短縮対象を決める。

## ユーザー決定と採用規則

- Phase 28の実装対象はprojection外device処理の短縮に限定する。
- candidateの採用単位を`adoption scope S`とする。`S`は実行前に評価できるstable dispatch keyで同じproviderへrouteされる入力集合である。
- `S`の代表full-model caseの少なくとも一つがTPOTまたはpost-TTFT E2Eを5%以上改善し、`S`内の全validation caseが非悪化なら、
  candidateを`S`へ採用する。`S`外はbaseline providerを維持し、provider identityとselection overhead込みの非悪化を確認する。
- `S`が固定matrix全体ならshared adoption、真部分集合ならscoped adoptionとする。
- stable keyはexact target、dtype、semantic op、shape/layout/alignment、target-only/MTP mode、mechanism上意味のあるcontext境界等で構成する。
  benchmark case名、prompt内容、実測結果、個別token列をkeyにしない。数値境界は`B-1/B/B+1`とscope内の複数代表値を検証する。
- Qwen graph、semantic op、state contractはgfx1030/gfx1201で共通にする。adoption scope差は共通registry下のprovider selectionまたは
  kernel mappingへ閉じ込め、model graphを複製しない。管理性のためshared pathを優先する。
- correctness、fallback、state publication、cleanup、resource failureは性能で相殺しない。

## 用語と計測境界

- `committed decode step`: 通常target modelが一つのoutput tokenを確定し、対応するKV/GDN stateをcommitした区間。
  MTPのdraft、verify、replay、内部Argmaxは別substepとして数え、output token数へ混ぜない。
- `adoption scope S`: candidate providerを選ぶstable keyと、そのkeyに一致するproduction入力集合。final run前にkey、代表case、
  境界、baselineへrouteするcomplementをmanifestへ固定する。
- `projection`: Q/K/V/O、linear/GDN input/output、MLP gate/up/down、LM headを実行するsLLM matmul kernelまたは
  hipBLAS/hipBLASLt kernel。Phase 28では計測するが変更しない。
- `projection外device`: committed decode step内でGPU上に実行された次の処理。
  - linear/GDN recurrent gated normとcausal convolution。
  - attention preprocessのheadwise normalization、RoPE、gate extraction。
  - causal attention、KV append/encoding。
  - RMSNorm。
  - residual add、SiLU、sigmoid、copy等のelementwise処理。
  - embedding、greedy Argmax、decode workspaceのdevice clear/copy。
- HIP APIのhost duration、kernel間idle、H2D/D2H、scheduler/frontendは`wall residual`として別集計する。原因の誤帰属を防ぐため
  観測するが、Phase 28では最適化しない。

## Primary scope

- model: fixed Qwen3.5-4B dense BF16 GGUFとderived lock。
- target:
  - canonical V620 exact `gfx1030`、UUID/BDF固定。
  - canonical R9700 exact `gfx1201`、UUID/BDF固定。
- workload: warm single request、batch 1、greedy、通常target-only decode、FP16 KV、同一prompt/completion token列。
- primary performance comparison: 同じsLLM source identityのbaseline/candidate。llama.cppはfamilyの存在と上限を確認する
  E1 diagnosticに限定し、採否やengine勝敗へ使わない。
- candidate boundary: A2で寄与が最大と確認したprojection外familyと、その入力・出力・stateを変えないkernel/provider実装。
- current R9700 productionのMTP経路はsecondary regression controlとし、target-only primaryへMTP内部stepを混ぜない。

## 初期candidate inventory

順位はfresh A1/A2 profileで確定し、次の順を事前決定とはしない。

1. linear/GDN recurrent gated norm
   - recurrent stateのcopy、decay、previous projection、update、current projectionのpass数。
   - Q/K L2 normとoutput RMSNormのthread-0 serial reduction。
   - value-head、dimension、key-dimensionのmapping、vector load、coalescing、wave reduction。
   - FP32 recurrent state、BF16 RNE stage、double-buffer transactional publicationは維持する。
2. attention preprocess
   - 現行head当たり1 threadで行うhead-dim 256のnorm、BF16 round、RoPE、gate copyをwave/workgroup並列化できるか。
   - Q/K raw scale、NeoX/MRoPE position component、Q gate layoutを維持する。
3. causal attention・KV
   - short-contextでのscore/value reduction、GQA head mapping、KV append/encodingのdevice share。
   - algorithm、KV layout、encodingを変更せず、現行online-softmax contract内のmappingだけを候補にする。
4. RMSNorm・elementwise・Argmax・device clear/copy
   - 個別寄与が小さい場合はlaunch数だけで順位を上げず、合計wall contributionと除去可能割合で判断する。
   - projectionとのfusion、sampling algorithm変更、host transfer統合は行わない。

## Fresh baselineとaccounting contract

### Identity

- source/tree、release binary、HIP kernel sources、ROCm 7.14.0、LLVM 23、exact target、code object、GPU UUID/BDF、model lock、
  GGUF digest、runnerをSHA-256へ結ぶ。
- public HIP runtime、all-dispatch HIP、fallbackなし、GTT spillなし、cleanup terminal-zeroを必須とする。
- local Qwen serviceがV620 pairを占有する場合は停止する。single-V620 Qwen構成へfallbackしない。

### Step boundary

- evidence/profile modeだけに`request_id`、`committed_output_step`、`execution_phase`、`model_component`、`op_family`、
  `is_projection`、MTP substep種別を持つbounded dispatch recordを追加する。
- prefill terminal Argmaxを時刻推定の境界として流用しない。execution transactionのprefill/decode開始、terminal completion、state commitを
  正規境界とし、各kernel dispatchをその区間へ写像する。
- production defaultにper-op timing overheadを追加しない。profiler/evidence laneとprofilerなしwall laneを別processで取得する。
- 各committed output stepについて、projection、各projection外family、device copy/fill、unclassified device、host/API、idleを集計する。
  MTP controlではdraft/verify/replayを別列にし、target-only primaryと混ぜない。

### Accounting

- familyごとにcalls/output token、device ns/output token、p50/p90/max、全device share、production TPOT shareを記録する。
- kernel durationだけでなく、grid/workgroup/wave、VGPR/SGPR/LDS/scratch、input/output/state bytes、FLOPsまたは主要instruction量を記録する。
- top familyはrocprof counterでDRAM/L2 traffic、VALU/VMEM、occupancy、wave stallを取得する。counter runのwall値は採否に使わない。
- `fixable contribution = production TPOT share × measured family reduction`でcandidateのfull-model期待値を算出する。
- projection外device合計と各familyの和が一致しない部分は`unclassified`として残し、projectionやhostへ推測で付け替えない。

## Candidate freeze

P28-A2終了時に次の条件で最小のcoherent work unitを一つ固定する。

1. 少なくとも一つのstable scopeで同じsemantic familyが最大または上位寄与を持つ。両target共通の寄与ならshared scopeを優先する。
2. actual-shape baselineとbounded prototypeから、`S`の代表full-model caseで5%以上へ届くcredible reductionがある。
3. candidateを受ける`S`とbaselineへ残すcomplementをstable keyで実行前に分離できる。
4. `S`をtarget、mode、shape/context境界で定義する場合、そのkeyがmechanismに対応し、境界両側と複数代表値を列挙できる。
5. numerical stage、state layout/publication、KV encoding、completion boundaryを変更しない。

単一familyで5%へ届かない場合、隣接するprojection外stageを同じdataflow上でまとめることが必要だとA2が示したときだけ、まとめた
work unitのscope、独立寄与、correctness boundaryを実装前に固定する。測定後に無関係なkernelを追加して5%へ救済しない。
どのstable adoption scopeでもprojection外device時間のcredible短縮上限が5% full-modelへ届かない場合は、production実装を開始せず
negative completionする。

## 受入基準

### Correctness

1. selected familyのhost descriptor/provider testはactual Qwen shapeに加え、`M=1,2,3`、非整列値、zero、overflow、境界両側を含む。
2. canonical両GPUのtiny oracleをFP64またはFP32 referenceへ照合し、finite分類、BF16 RNE stage、padding非破壊、output order、
   unsupported layoutのfail-closedを確認する。
3. GDNを変更する場合はprevious/next convolution state、FP32 recurrent matrix、expected/committed length、double-buffer swapを
   stepごとに照合する。partial submission、timeout、cancel、query failureではstateを公開しない。
4. attention/KVを変更する場合は17、255、256、257および選択した長context境界でtoken/position、causal range、KV content/lengthを照合する。
5. Qwen full-modelのtarget-only greedyはbaseline/candidateでprompt/completion token IDs、bounded logits、visible output、stop、usageを一致させる。
6. R9700 current MTP、short prefill、explicit all-logits、fixed-seed sampling 3 profileをregression controlにする。
7. CPU/backend fallback、別target code object、GTT spill、timeout、crash、zero test selectionはPASSにしない。

### Performance

8. profilerなしで3 warmup + 10 measured以上を取得し、baseline/candidate順をtargetごとにcounterbalanceする。
9. 固定full-model matrixは両targetの17/128、28/128、255/128 target-only greedyと、R9700 current production MTP controlとする。
10. `S`の代表caseの少なくとも一つがdecode TPOTまたはpost-TTFT E2Eを5%以上改善し、`S`内の全validation caseでstableな悪化が
    ない場合にcandidateを`S`へ採用する。`S`外はbaselineへrouteする。
11. `S`内caseの0〜2%負差はbaseline/candidateを挟み直し、最終bracketで非悪化を確認する。stableな悪化が残ればscope keyを
    mechanismに基づいて再定義してfinal matrixを全再実行するか、そのscopeのcandidateを棄却する。
12. `S`外ではbaseline provider identity、dispatch count、selection overhead込みの性能driftがないことを確認する。
13. operator、counter、device-timeだけの改善では採用しない。family短縮がfull-model wallへ転化しない場合は棄却する。
14. projection kernel/provider/timeをcontrolとして記録し、Phase 28 candidateがprojectionへ仕事を移しただけの場合は採用しない。

### Resource・architecture

15. model-resident weight、GGUF、public API、semantic graph順序を変更しない。workspace/register/LDS増加はactual valueを記録する。
16. gfx1030/gfx1201やrequest scopeの差は、同じsemantic descriptorを受けるstable provider selectionまたはkernel mappingに限定する。
17. AI提案のnonblocking guardとしてrequest workspace +1%、model-ready +5%を記録する。originはtemporary bufferやextra prepareの
    regression risk、scopeはPhase 28 candidate、costは既存R0再計測、expiryはA2 candidate freezeとする。超過だけで自動棄却しない。

### Evidence・closeout

18. raw trace、binary、model、full logits、生成全文を追跡しない。schema-validなbounded aggregate、digest、runner、test、plan/historyだけを残す。
19. affected host/build/GPU/full-model checks、one integration review、changed findingだけのfocused re-reviewを行う。
20. 採用時はruntime/compatibility/provenance/main planを同期する。candidateなし・基準未達時はproduction defaultへ残さず、
    corrected projection外device breakdownと再検討条件を記録してnegative completionできる。

## 計測matrix

| case | workload | 目的 | 採否への使用 |
| --- | --- | --- | --- |
| H0 | selected provider actual shape + boundary | descriptor/layout/state contract | correctness |
| G0 | tiny distinctive tensor、両GPU | numerical/state oracle | correctness |
| B0 | prompt 28 + 16 committed target-only steps | per-step family accounting | mechanism |
| B1 | prompt 255 + 16 committed target-only steps | context-dependent attention control | mechanism |
| C0 | top family actual shapes + counters | traffic/occupancy/stall attribution | mechanism |
| D0 | adoption key、scope内代表値、`B-1/B/B+1`、scope外control | routing coverage/leakage | hard adoption |
| F0 | prompt 17 + output 128、両GPU | short-context full-model | hard performance |
| F1 | prompt 28 + output 128、両GPU | Phase 27 continuity | hard performance |
| F2 | prompt 255 + output 128、両GPU | larger-KV full-model | hard performance |
| M0 | R9700 current production MTP | production regression | hard non-regression |
| S0 | fixed sampling 3 profile | logits/RNG/stop regression | correctness |
| P0 | prefill 17/255/256/257 | Phase 24 regression | correctness |
| R0 | model-ready、resident/workspace high-water | resource accounting | resource |

## 作業順序

### P28-A0: Phase 27 residual correction・acceptance freeze

- Phase 27 summary/history/main planの3.80倍/3.54倍claimを、prefill/MTP/timing boundaryが混在したcoarse residualへ訂正する。
- current source、model、toolchain、GPU、adoption rule、target-only primary、MTP control、scope manifest fieldsをbaseline manifestへ固定する。
- evidence-only dispatch recordとcommitted-step boundaryのhost contract/testを先に作る。

### P28-A1: fresh exact-step dual-GPU baseline

- exact target向けにfresh buildし、B0/B1をtarget-onlyで取得する。
- projectionとprojection外をstepごとに分け、GDN、attention preprocess、causal attention/KV、RMSNorm、elementwise、Argmax、copy/fill、
  unclassifiedへ100% accountingする。
- R9700 MTPは別runでdraft/verify/replayを分類し、target-only baselineへ混ぜない。
- Phase 27 coarse residualとの差をprefill contamination、MTP内部work、token divisor、unclassifiedへ説明する。

### P28-A2: top family attribution・candidate freeze

- 両targetのdevice ns/output token、TPOT share、call count、resource、counterを比較し、最大fixable contributionを持つfamilyを選ぶ。
- actual-shape bounded prototypeまたは既存provider controlで保守的なfamily reductionを測り、full-model 5%へ届くか算出する。
- coherent work unitを一つ固定し、変更kernel、provisional adoption scope `S`、stable key、scope内代表case、境界、baseline complement、
  保持する数値/state境界、期待削減、rollback pointを記録する。
- credible 5%候補がなければA3以降へ進まずnegative closeoutする。

### P28-A3: host contract・tiny GPU oracle

- provider key、shape、alignment、workspace、state generation、failure contractとscope membershipをhost testへ固定する。
- G0をbaseline/candidate両providerで実行し、非整列値と境界両側を含むnumerical/state oracleを両GPUでPASSさせる。
- 外部codeをreuseする場合は実装前にprovenanceへexact source/path/hash/reuse modeを記録する。

### P28-A4: bounded implementation

- A2で固定したprojection外work unitだけを実装する。baseline providerは比較・rollback用に維持する。
- common semantic descriptorとtransactional state publicationを保ち、target差はprovider registry/kernel mappingへ閉じ込める。
- projection、host scheduler、MTP algorithm、GGUF/KV formatを変更しない。

### P28-A5: mechanism・correctness verification

- H0/G0/D0、Qwen target-only、M0/S0/P0、failure/cancel/cleanupを実行する。
- B0/B1/C0でselected familyのdevice time、calls、resource、counter、projection control、unclassifiedを再取得する。
- mechanismが成立しない、別familyへ時間が移る、state/roundingを変える場合はcandidateを棄却する。

### P28-A6: full-model adoption

- F0/F1/F2/M0をprofilerなし・counterbalancedで両GPU取得する。
- final run前にadoption scope `S`をfreezeする。`S`の代表caseが5%以上改善し、`S`内全caseが非悪化ならcandidateを`S`へ採用する。
  `S`が固定matrix全体ならshared、真部分集合ならscoped adoptionとする。
- D0でscopeの境界両側と複数代表値を再測定し、benchmark固有分岐やcandidate leakageがないことを確認する。
- `S`外はbaseline providerへrouteし、provider identity、dispatch count、selection overhead込みで非悪化を確認する。
- 採用しないscopeのcandidate defaultを除去し、baseline production pathを維持する。

### P28-A7: integration・closeout

- bounded summary/schema/test、affected validation、integration reviewを完了する。
- main plan、history、runtime、compatibility、provenanceを実結果へ同期し、planをarchiveする。
- 次Phaseを自動開始しない。

## Rollback・停止・再計画

- wrong token/logit/state、fallback、GTT spill、cleanup failure、unbounded allocationはcandidateを採用しない。
- corrected projection外device上限が5% full-modelへ届かない場合は実装せずnegative completionする。
- 最大差がhost/API/idle、projection、prefill、MTP algorithmに移った場合はPhase 28へ混ぜず別候補として返す。
- 同じwork unitが2回reject、review時間が実装時間超、functional progressが1時間停止、verification/docsが30%超、
  見積り1.5倍超、acceptance変更時は追加作業を止め、同じwork unitを再計画する。

[Phase 27 bounded summary](../../../../../../ci/matrix/phase27-weight-stream-summary-v1.json)
[Phase 28 bounded summary](../../../../../../ci/matrix/phase28-nonprojection-summary-v1.json)
[Phase 28対応履歴](../../../../../history/2026/08/11-20/phase28-decode-nonprojection-device-optimization.md)

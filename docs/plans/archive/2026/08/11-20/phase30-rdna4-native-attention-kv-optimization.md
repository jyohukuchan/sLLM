# Phase 30: RDNA4 native attention/KV hardware-path最適化

> 状態: 完了（gfx1201 native FP8 read/wave providerを限定採用）
> 作成日: 2026-08-19

## 目的

exact RDNA4 `gfx1201`のcausal/full attentionとKV appendについて、現行のgfx1030/gfx1201共通scalar/vector baselineを維持したまま、
RDNA4が持つnative FP8 conversion、packed dot、FP16/BF16/FP8 matrix instructionを実際に使うtarget-scoped providerを設計・比較する。

Phase 30は「FP8 KVを使っている」というstorage上の事実と、「RDNA4のnative FP8演算を使っている」というexecution上の事実を分離する。
候補のcode objectに期待するISAが存在し、actual dispatchがそのsymbolへ到達した場合だけnative pathと呼ぶ。

主な対象は次の三work unitである。

1. FP8 KV append/readのsoftware codecをRDNA4 native conversionへ置き換えられるか確認する。
2. decode `M=1`ではmatrix core使用自体を目的化せず、packed dotとkey tile処理でper-token直列loopとblock同期を減らす。
3. prefill `M>1`ではQK/PVをtile化し、FP16/BF16/FP8 WMMA/SWMMACまたは同等の検証済みlibrary providerを比較する。

gfx1030は共通semantic contractとbaseline providerを維持するcontrol targetである。gfx1201専用最適化のためにmodel graph、KV format、
frontend、service APIを複製しない。

## 発見済みbaseline facts

- generic causal attentionはFP16用ID 2とpacked-KV用ID 3を報告するが、実体は一つの`causal_attention_kernel`へ
  `encoding`を渡す共通launchである。logical/device symbol文字列は別でも、compiled kernel bodyは別れていない。
- Q/outputはBF16、FP16 KVはFP16であり、QKは各要素をFP32へ変換してscalar積算する。FP8 KVもE4M3FN byteをsoftwareでFP32へ展開して
  scaleを掛け、同じ積算へ入る。
- head dimension 256では256-thread blockの各threadがkeyごとに概ね一要素を処理し、dot reduction、online softmax、V updateの間に
  約11回のblock同期を行う。context 16,384では一block当たり約18万回の同期となる。
- exact gfx1201 code objectには`v_wmma_*`、`v_swmmac_*`、`v_dot2_*`、`v_dot4_*`、native FP8 convertがなく、
  `v_mul_f32`、`v_add_f32`、integer bit manipulation、`v_ldexp_f32`で実行される。
- GFX12 ISAにはOCP FP8相互変換、FP8 packed dot、FP16/BF16 packed dot、FP16/BF16/FP8 WMMA/SWMMACが存在する。
  Phase 10/15Oのmodel weight FP8はgfx1201 hipBLASLt native providerを既に使用しており、toolchain全体のnative FP8非対応が原因ではない。
- Phase 16はFP8/NVFP4 KV format、memory、quality、direct packed consumptionを完成させるbaseline providerとして計画され、
  RDNA4専用attention provider、FlashAttention、WMMAは実装範囲に含めていなかった。
- 2026-08-19の16,384-token isolated measurementでは、V620 FP8 KV attentionはFP16比約6.3%遅く、R9700はFP16側のDPM二峰性により
  単一比率を確定できなかった。これは両targetともsoftware FP8 pathの結果であり、RDNA4 native FP8性能として採否へ使わない。

これらはA0でcurrent source/buildから再確認する。過去のtemporary benchmarkや`/tmp` code objectをfinal identityとして再利用しない。

## Phase固有の境界

### 対象

- semantic op:
  - opaque KV stateへのBF16 K/V append。
  - Qwen系generic causal/full attention。
  - 同じalgorithmを安全に共有できる場合のGemma BF16/sliding attention。
- encoding:
  - FP16 KV baseline。
  - dynamic/static OCP E4M3FN FP8 KV。
- target:
  - primary: R9700 exact `gfx1201`、UUID `GPU-a8e9ddefa2d60f55`。
  - control: V620 exact `gfx1030`、UUID `GPU-76a08c022586fed6`。
  - common ABI/headerを変更した場合だけexact `gfx942` compile/link controlを行い、実機PASSや性能へ一般化しない。
- request mode:
  - decode `M=1`。
  - prefill `M>1`。
  - KV append chunk `1/32/256` token。
- representative head shape:
  - Qwen3.5: `q_heads=16`、`kv_heads=4`、`head_dim=256`。
  - Gemmaはactual reviewed shapeとsliding/full境界を別scopeとして扱う。

### 非対象

- gfx1030へnative FP8またはmatrix providerを偽装すること。
- NVFP4、MXFP4、TurboQuant、新しいKV format。
- model weight FP8/FP4 provider、projection/GDN、MTP algorithm、continuous batchingの再設計。
- Paged Attention、RadixAttention、prefix sharing、multi-GPU。
- request全体のFP16/BF16 KV mirror、全context事前dequant、host round-trip。
- matrix instructionを含むだけで採用すること。使用率、device時間、full-model転化を伴わないISA出現は成功条件ではない。

## Providerとadoption scope

provider selectionは共通registry内のstable keyだけで行う。

- exact target。
- KV encodingとstatic/dynamic scale contract。
- request mode `decode/prefill`。
- Q/K/V dtype、head shape、GQA ratio、layout/alignment。
- mechanism上必要なcontextまたはM bucket。

prompt、token値、測定後の勝敗、個別model出力をrouting keyにしない。context/M境界を使う場合は、final run前に境界`B`を固定し、
`B-1/B/B+1`とscope内複数代表値を検証する。

候補は次のscopeを独立採否できる。

| scope | target | encoding | mode | 主指標 |
| --- | --- | --- | --- | --- |
| S1 | gfx1201 | FP8 dynamic/static | append | TTFT/prefill wall、append device ns/token |
| S2 | gfx1201 | FP16 | decode M=1 | post-TTFT TPOT、attention family ns/token |
| S3 | gfx1201 | FP8 | decode M=1 | post-TTFT TPOT、attention family ns/token |
| S4 | gfx1201 | FP16/BF16 | prefill M>1 | TTFT、prefill attention device time |
| S5 | gfx1201 | FP8 | prefill M>1 | TTFT、prefill attention device time |

S4/S5はdtypeと数値機構が異なるため、同時合格を要求しない。Gemmaを統合する場合もQwenと別adoption scopeにし、
Qwenで速いことをGemma pathの採用根拠にしない。

## Candidate inventory

### C1: gfx1201 native FP8 codec

- appendのBF16/F32→OCP E4M3FNをnative packed conversionで行い、dynamic row scale reductionとK/V atomic publicationを維持する。
- attention loadのE4M3FN→FP32をnative scalar/packed conversionへ置き換え、token/head scaleを同じstageで適用する。
- 全256 code、NaN、Inf、signed zero、subnormal、saturation、RNE/tieを現行`kv-fp8-v1` contractとbit-exact比較する。
- hardware instructionのnonfinite/roundingがcontractと一致しない場合、補正を含むcandidateを一つだけ比較する。bit-exactにできない場合はN2へ分類し、
  codec高速化だけをN0として扱わない。
- code objectに`v_cvt_f32_fp8`/`v_cvt_pk_f32_fp8`と`v_cvt_pk_fp8_f32`等の期待命令があり、
  gfx1030 artifactに混入しないことを確認する。

### C2: decode packed-dot / wave-tiled online attention

- matrix coreへ直行する前に、FP16/BF16 `v_dot2`、FP8 `v_dot4`、native conversion後のpacked FMAをactual shapeで比較する。
- keyを1個ずつ全blockで処理するbaselineから、複数keyをwave/LDS tileへ読み、QK score、max/sum、V accumulationをまとめる。
- 256-thread LDS treeをkeyごとに繰り返さず、wave shuffle、固定tree、tile単位online softmaxでblock同期回数を削減する。
- QはBF16、FP16 KVはFP16であるため、FP16 packed dot候補ではQのBF16→FP16変換範囲と数値分類を明示する。
- FP8 packed dot候補ではQをFP8へ量子化する場合と、Kをnative decodeしてBF16/FP16 dotへ入れる場合を混同しない。
  前者は新しいQ quantizationでありN2候補、後者はstorage帯域削減とnative decodeを使うFP16/BF16演算候補である。
- decode M=1で16x16 matrix tileを15/16空費する候補は、batch/head groupingで実利用率を説明できない限りC3へ持ち込まない。

### C3: prefill matrix attention

- QKとPVを16x16以上へtile化し、rocWMMA/CK Tile、直接WMMA/SWMMAC、llama.cppから再利用可能なHIP FlashAttention候補を比較する。
- plain scalar loopをcompilerが自動matrix化すると仮定せず、actual code objectとprofiler counterでmatrix instruction到達を証明する。
- FP16/BF16候補は両input dtypeを揃え、FP32 accumulatorとonline softmaxの精度stageを記録する。
- FP8候補はQ、K、softmax probability、VのどこをFP8/BF16/FP16にするか、scale axis/lifetime、QK/PV accumulatorを明示する。
- causal mask、GQA、RoPE済みK、sliding/full、terminal row、odd M、head tailをtile paddingで壊さない。
- full score/probability matrixをcontext二乗でresidentに保持しない。FlashAttention型bounded tileまたは同等のmemory上限を要求する。

### Variant上限

- C1はpure nativeと必要なら補正版の最大2候補。
- C2はbaseline、packed-dot、wave-tiledの最大3候補。C1 winnerだけと組み合わせ、codec×attentionの全直積を作らない。
- C3はlibrary/reuse候補とcustom候補の最大2つ。correctnessまたはactual ISAが成立しない候補を性能matrixへ残さない。
- 各stageで5% full-model改善へ届くAmdahl上限がない場合、後段production統合へ進まずnegative resultを記録する。

## 数値・出力変更規則

`docs/compatibility/numerical-output-changes.md`を正本とする。

- N0候補:
  - bit-exact native FP8 codec。
  - arithmetic order、dtype、rounding stageを維持するload/store置換。
- N1候補:
  - real-number equationとdtype/rounding stageを維持し、逐次和から固定balanced treeへ変えて解析的誤差boundを非増加にするもの。
  - softmax max/sum/V accumulationの全stageについて非増加を説明できる場合だけN1とする。一部のdotだけを改善して全体をN1としない。
- N2候補:
  - BF16 QをFP16/FP8へ変換する。
  - softmax probabilityをFP32からFP16/BF16/FP8へ下げる。
  - exp近似、scale recipe、accumulator、丸めstageを変更して既存tolerance内の小さな誤差増加が見込まれる。
  - 人間承認なしにproduction採用しない。
- N3候補:
  - race、非決定atomic、未説明のtoken差、非有界なscale/overflow、provider repeat不一致。
  - 採用せずreplanする。

出力へ影響しうる候補は、最初のtoken/logit分岐、scope、provider/source identity、rollback、性能・resource結果を台帳へ記録する。
N1のために全model FP64検証を新しい定常gateにしない。N2/N3の分類解消に必要な場合だけbounded high-precision controlを追加する。

## Fresh baseline・測定contract

- final candidateと同じsource、release options、ROCm 7.14、exact target、model lock、GGUF、runnerをdigestへ結ぶ。
- local Qwen subagent serviceを停止し、V620 pairをsLLM測定と競合させない。single-V620 Qwenへfallbackしない。
- GPUはUUIDで一台だけ可視化し、一度に一GPUだけ測定する。foreign process、VRAM/GTT、temperature、clock、power、ECC、loader rootを
  pre/during/postで記録する。
- R9700 FP16 baselineで観測した14.8〜26.7 msのDPM二峰性を放置しない。operator比較は同一session、同一immutable KV stateで
  baseline/candidateをinterleaveし、HIP event device timeとwallを分離する。
- full-modelは3 independent process以上、`B-A-A-B`または`A-B-B-A`でcounterbalanceする。bracket baseline p50が2%を超えてdriftしたrunは
  採否へ使わず、一度だけhealth確認後に取り直す。
- profiler laneとprofilerなしwall laneを分離し、rocprof observer effect下のwall値を採否へ使わない。
- clock/performance levelは変更せず、full-model warmupでproduction相当のloadへ到達させる。観測したDPM modeを後付けrouting keyにしない。

### Operator matrix

| case | shape | 指標 | 用途 |
| --- | --- | --- | --- |
| O0 | FP8 codec 全256 code + special | bit pattern、rounding、ISA | correctness/native proof |
| O1 | append chunk 1/32/256 | device ns/token、GB/s、publication | C1 mechanism |
| O2 | decode context 1/1024/4096/9999/10000/10001/16384/32768 | attention ns/layer/token、sync、traffic | C2 mechanism/boundary |
| O3 | prefill M 2/3/7/15/16/17/32/64/256 | device time/query、matrix utilization | C3 mechanism/boundary |
| O4 | head dim 255/256/257、GQA 2/4/8/16 | tail、padding、provider routing | correctness |
| O5 | static/dynamic FP8、FP16 | value/scale bytes、VMM commit | resource |

long-context O2はsynthetic committed KVでattention format差を直接測る。10k tokenの二次計算量prefillをO2へ混ぜない。
実prefill/TTFTはfull-model F2で別に測る。

### Full-model matrix

| case | workload | 主指標 | 採否への使用 |
| --- | --- | --- | --- |
| F0 | Qwen3.5-4B、prompt 17/255、output 128 | short-context TPOT/TTFT、token/logit | scope control |
| F1 | prompt 9999/10000/10001/16384、output 32/128 | long-context TPOT、post-TTFT E2E | decode hard performance |
| F2 | input M 1024/4096/10000+、generation 1 | TTFT、prefill attention、peak VRAM | prefill/append hard performance |
| F3 | FP16 KV対FP8 KV、同じweight provider/model lock | KV-only attribution | hard attribution |
| F4 | Gemma fixed/sliding境界 | model-specific semantics/性能 | Gemma scopeのみ |
| F5 | service non-stream/SSE、cancel/recovery | visible output、usage、cleanup | integration correctness |

F3ではweight dtype/providerを比較途中で変えない。FP8 weightのnative matmul改善をFP8 KV改善として数えない。

## 受入基準

### Correctness・native execution

1. C1は全FP8 code、NaN/Inf、signed zero、subnormal、saturation、tie、static/dynamic scaleを現行contractと照合する。
2. attentionはscalar oracleでM `1/2/3/7/15/16/17/37`、KV `1/3/255/256/257/1023/1024/1025`を全出力照合する。
   10k以上は全出力をCPUで再計算せず、固定head/dimension sample、finite、causal/GQA mapping、baseline differentialを組み合わせる。
3. expected/committed length、K/V atomic publication、generation、cancel、timeout、partial failure、release/cleanupを維持する。
4. no full-cache mirror、no CPU/別encoding fallback、no wrong-target artifactをactual dispatchとmemory auditで確認する。
5. candidate code objectへ期待するnative conversion/dot/matrix命令が存在し、actual kernel symbolへdispatchする。
   logical metadata文字列だけでnative PASSにしない。
6. gfx1030ではbaseline providerへrouteし、gfx1201専用命令を含むcode objectをloadしない。
7. same-provider repeatを再現可能にし、token/logit差をN0〜N3へ分類する。
8. CPU fallback、timeout、crash、zero test selection、profiler failureをGPU PASSにしない。

### Performance adoption

9. operator candidateは対象family device p50で5%以上短縮し、full-modelで5%へ届くAmdahl上限がある場合だけproduction候補へ進む。
10. production採用は一般規則を維持し、事前freezeしたadoption scope内の全validation patternで主full-model指標にstableな悪化がなく、
    少なくとも一つの代表patternで5%以上改善することを要求する。
11. decode scopeはpost-TTFT TPOT、prefill/append scopeはTTFTを主full-model指標とする。device timeは機構説明でありwall条件を置き換えない。
12. 一部scopeだけ合格した場合はgfx1201/encoding/mode/bucketへ限定採用し、complementはbaselineへrouteする。
13. FP8がFP16より速いことを各providerの採用条件にしない。FP16 candidateとFP8 candidateはそれぞれ同encoding baselineに対する改善で判定する。
14. memory削減、ISA出現、matrix utilization、kernel単体の大幅改善だけではproduction採用しない。

### Resource・architecture

15. value/scale plane、logical/physical committed bytes、peak VRAM、scratch、VGPR/SGPR/LDS、occupancy、matrix utilization、sync countを記録する。
16. tile scratchはrequest/model shapeから上限計算でき、decode stepごとのunbounded allocation、host readback、device-wide synchronizeを追加しない。
17. semantic graph、public API、GGUF、model lock、KV encoding versionを変更しない。encoding変更が必要ならPhase 30を止め別formatとして再計画する。
18. provider差は共通registryへ閉じ込め、gfx1201専用algorithmをQwen/Gemma graphへ直接分岐として埋め込まない。

### Evidence・closeout

19. raw trace、code object、model、raw KV、full logits、生成全文を追跡しない。bounded aggregate、digest、schema/test、plan/historyだけを残す。
20. affected host/build/GPU/full-model checks、one integration review、changed findingだけのfocused re-reviewを行う。
21. 採用時はruntime、GPU/software compatibility、provenance、numerical-output ledger、main planを同期する。
22. どのscopeも5%条件へ届かない場合はtemporary providerをproduction sourceから除去し、software baseline維持のnegative completionとする。

## 作業順序

### P30-A0: acceptance・current ISA baseline freeze

- current sourceからexact gfx1030/gfx1201 artifactをclean buildし、single physical kernel、encoding branch、instruction inventoryを再取得する。
- provider scope、数値分類、operator/full-model metric、10k+ context matrix、routing boundaryをsummary manifestへfreezeする。
- Phase 16/23/28/29 evidenceと今回のfresh baselineを混ぜず、source identityと測定目的を分ける。

### P30-A1: interference-controlled long-context baseline

- O1/O2/O3とF0/F1のbaselineを取得し、append、QK/reduction、softmax、PV、host/runtimeを分解する。
- same-session interleaveでR9700 DPM二峰性を監査し、再現可能なpaired baselineを確立する。
- attention/KV familyのfixable shareとfull-model 5%に必要な短縮率をcontext/mode別に計算する。

### P30-A2: C1 native FP8 codec prototype

- gfx1201専用native convertを最大2候補で実装し、O0/O1/O2のload部分を比較する。
- bit-exactならN0、差があればN2/N3へ分類する。actual ISAとdispatch symbolを検査する。
- append/readのfamily短縮とAmdahl上限が不足する場合、codec単独production統合を止めC2のinput mechanismとしてのみ保持する。

### P30-A3: C2 decode packed-dot / wave-tile prototype

- baseline、packed-dot、wave-tiledをactual Qwen shapeで比較し、key当たりsync、VMEM、VALU、LDS、occupancyを取得する。
- context 9999/10000/10001/16384/32768で線形性とrouting境界を確認する。
- 5% full-modelへ届くcredible winnerを最大一つ選ぶ。FP16/FP8でmechanismが違えば別winnerを許すが、共通bodyを優先する。

### P30-A4: C3 prefill matrix prototype

- llama.cpp direct reuse、rocWMMA/CK Tile、custom WMMA/SWMMACの順に既存実装を検討し、source/provenanceを実装前に固定する。
- O3でodd/tile境界とactual matrix ISAを確認し、scalar/tiled providerと比較する。
- Q/Pの精度低下を伴う候補はN2として性能・品質結果を提示し、人間承認までproduction統合しない。

### P30-A5: bounded provider integration

- A2〜A4で固定したwinnerだけをcommon provider registryへ実装する。
- gfx1201/encoding/mode/bucketをstable keyへし、gfx1030とscope complementはbaselineを維持する。
- KV owner、transaction、VMM grow、cancel/recovery、prepared executionへtarget固有状態を漏らさない。

### P30-A6: correctness・numerical・resource verification

- O0/O4/O5、bounded scalar oracle、long-context sampled oracle、state publication、cancel/failure/cleanupを実行する。
- F0/F3/F4/F5でtoken/logit、visible output、usage、service semanticsを確認する。
- N0/N1は規則に従い処理し、N2は人間判断へ結果を返し、N3は棄却する。

### P30-A7: scoped performance adoption

- F1/F2を3 process以上のcounterbalanced bracketで実行する。
- scope内全pattern非悪化かつ任意代表pattern5%以上なら限定採用する。scope外baseline routingをsymbol、dispatch、wallで確認する。
- FP16/FP8、decode/prefill、Qwen/Gemmaを一括判定せず、S1〜S5とmodel scopeごとに採否を記録する。

### P30-A8: integration・closeout

- bounded summary/schema/test、integration review、findingのfocused re-reviewを完了する。
- 採用providerとrollback identity、ISA、性能、numerical classificationをhistoryと台帳へ記録する。
- main plan、runtime、compatibility、provenanceを同期してplanをarchiveする。
- 採用されなかったtemporary variant、debug routing、profile-only metadataをproduction sourceへ残さない。
- 次Phaseを自動開始しない。

## 停止・再計画条件

- native FP8 conversionが現行`kv-fp8-v1` contractと一致せず、bounded補正でも性能上限がない場合はC1をnegative closeoutする。
- matrix providerがQ/P dtypeまたはscale recipeを変え、N2として人間承認を得られない場合はbaselineを維持する。
- actual code objectまたはdispatchに期待ISAがない場合、native/matrix名称で性能測定を続けない。
- full-cache mirror、context二乗resident score、unbounded workspace、wrong token/state、fallback、GTT spill、cleanup failureがあれば候補を採用しない。
- operator改善があってもfull-model 5%へ届くAmdahl上限がない場合、追加tile/autotune探索を止める。
- R9700 DPM二峰性をpaired measurementで解消できない場合、wall採否を止め、測定contractを再計画する。
- 同じwork unitが2回reject、review時間が実装時間超、functional progressが1時間停止、verification/docsが30%超、
  見積り1.5倍超、acceptance変更時はvariant追加を止めて同じwork unitを再計画する。

[Phase 16計画](phase16-kv-cache-fp8-nvfp4.md)
[Phase 29計画](phase29-gdn-useful-workgroup-parallelization.md)
[数値・出力影響変更台帳](../../../../../compatibility/numerical-output-changes.md)
[Phase 30履歴](../../../../../history/2026/08/11-20/phase30-rdna4-native-attention-kv-optimization.md)
[Phase 30 bounded summary](../../../../../../ci/matrix/phase30-rdna4-attention-kv-summary-v1.json)
[メイン計画](../../../../main-plan.md)

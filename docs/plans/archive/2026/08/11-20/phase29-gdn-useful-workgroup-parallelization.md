# Phase 29: GDN useful-workgroup並列化最適化

> 状態: 完了（N1 shared candidate採用）
> 作成日: 2026-08-18

## 目的

Qwen3.5-4B dense BF16の通常decodeで、Phase 28採用済みのGDN recurrent gated normをbaselineに、
recurrent stateの全thread・waveへ有用な仕事を割り当てるtile構造を設計し、canonical V620 `gfx1030`と
R9700 `gfx1201`のGDN device時間を短縮する。

Phase 28後の探索では、現行32 workgroupに対して総workgroup数を16、64、128、256へ変えた単純分割を比較した。
16は1 workgroupが2 value headを直列処理し、64以上は1 value headを複数workgroupへ分けたが、次の構造上の制約があった。

- 16 workgroupは同時に処理できるvalue headと利用CU数を半減させる。
- 64/128/256 workgroupは各blockの有用state threadを64/32/16へ減らし、Q/K norm、gate、decay等をblockごとに重複する。
- block間同期がないため、64以上では現行のfused output RMSNorm/gateを別kernelへ移した。
- 有用なstate work総量を増やさず、CU数とworkgroup数を一致させただけなので、全variantが現行32より遅かった。

Phase 29はworkgroup数だけを調整する探索を繰り返さない。Q/K準備を一度にし、state tileの全waveへ有用な仕事を与え、
partial reductionとoutput finalizeを含むGDN family全体で短縮できるproviderを一つのcoherent work unitとして扱う。

## ユーザー決定とPhase固有の採用規則

- Phase 29の改善幅と採用可否は、full-model TPOT/E2Eではなく**GDN device時間だけ**で判定する。
- 5% threshold自体は維持し、Phase 29では`GDN device ns / committed decode step`へ適用する。
- Phase 29以外の一般的なfull-model 5%規則は変更しない。
- candidateの採用単位はstable dispatch keyで定義した`adoption scope S`とする。
- `S`内の全validation patternでGDN device時間にstableな悪化がなく、少なくとも一つの代表patternで5%以上短縮した場合、
  candidateを`S`へ採用する。
- 一方のexact targetだけが上記条件を満たす場合は、そのtargetのscopeへ限定採用し、他targetはbaseline providerを維持する。
  target splitは共通semantic descriptorとprovider registry内に閉じ込め、model graphを複製しない。
- 両targetで条件を満たす場合はshared providerを優先する。管理性を理由に、安全なscoped improvementを棄却しない。
- full-model TPOT、TTFT、post-TTFT E2E、host/API時間はdiagnosticとして記録するが、改善・非悪化を採用条件にしない。
- correctness、state publication、fallback、resource、cleanup、unsupported inputのfail-closedは引き続きhard conditionとし、
  GDN device時間の改善で相殺しない。

## 採用指標の厳密な境界

### Primary metric

`GDN device ns / committed decode step`を、通常target-only decodeの一つのcommitted output step内で実行された次のdevice kernel durationの
総和として定義する。

- Q/K L2 norm、BF16 RNE、Q scale等のGDN recurrent入力準備。
- recurrent stateのdecay、previous projection、beta update、current projection。
- tile間のpartial sum、reduce、temporary clear/copy。
- output RMSNorm、BF16 RNE、norm weight、SiLU gate適用。
- candidateが上記処理を分離、融合、renameした全kernel。

現行baselineでは上記が`sllm_linear_attention_recurrent_gated_norm_v1`一つに融合されている。candidateが複数kernelへ分けても、
その全durationを合算する。kernel名、launch数、別streamへの移動だけで時間を除外しない。

### Exclusionと移動防止

- linear/GDN input/output projection、causal convolution、embedding、attention、KV、samplingはprimary metricから除外し、controlとして記録する。
- Phase 29ではGDN recurrentとprojection、causal convolution、attentionのfusionを実装しない。primary境界を跨いで仕事を移さない。
- GDN用temporary bufferのfill/copyとprovider selectionにより発生したdevice処理はprimary metricへ含める。
- HIP API host duration、queue idle、H2D/D2Hはprimary metricへ含めず、wall diagnosticへ分離する。
- prefill、MTP draft/verify/replay、correctness-control requestのGDN callを通常target-only primaryへ混ぜない。

### 統計と非悪化判定

- 14 request × 16 output tokenの固定protocolから224 Argmaxをrequestごとに分け、各requestのprefill terminalを除く
  210 committed decode stepを集計する。
- 各stepで全layerのGDN family durationを合算し、processごとのp50/p90/max、layer p50、calls/stepを記録する。
- target/pattern/variantごとに独立processを3回以上実行し、baseline/candidateを`B-A-A-B`または`A-B-B-A`でcounterbalanceする。
- bracket両端のbaseline p50が2%を超えてdriftしたrunは採否へ使わず、health/clock/processを確認して取り直す。
- 0〜2%のcandidate悪化は同じbracketを一度だけ再実行し、final bracketで非悪化ならnoise、stableに残ればそのscopeで棄却する。
- 5%改善は`1 - candidate_p50 / bracketed_baseline_p50 >= 0.05`で判定する。p90、counter、full-model値は説明用であり、
  p50のhard判定を置き換えない。

## Primary scope

- model: fixed Qwen3.5-4B dense BF16 GGUFとderived lock。
- semantic op: GDN recurrent gated norm、FP32 recurrent state、BF16 input/output。
- shape: `qk_heads=16`、`value_heads=32`、`head_dim=128`、batch 1、decode `M=1`。
- target:
  - canonical V620 exact `gfx1030`、UUID `GPU-76a08c022586fed6`。
  - canonical R9700 exact `gfx1201`、UUID `GPU-a8e9ddefa2d60f55`。
- workload: warm single request、greedy、通常target-only decode、同一prompt/completion token列。
- state contract: FP32 recurrent matrix、previous/next double buffer、expected/committed length、cancel時非公開。
- baseline: Phase 28で採用したcopy/decay/previous projection統合済み32-workgroup fused kernel。

context lengthはGDN shapeを変えないため、実測後にcontext固有routing keyを作らない。17/28/255-token promptは同じscopeのvalidation patternとし、
mechanism上の新しいshape/layout境界が見つかった場合だけfinal run前にstable keyを再定義する。

## 既知baselineと探索結果

次の値は同一の正式採用runではなく、Phase 29候補を絞るためのrocprofv3探索値である。新しいA1 baselineを採否の正本にする。

| total workgroups | 構造 | V620 GDN ms/step | R9700 GDN ms/step | 判定 |
| ---: | --- | ---: | ---: | --- |
| 16 | 2 value head/WG、直列 | 3.191 | 2.936 | 両target悪化 |
| 32 | 1 value head/WG、現行fused | 1.354 | 0.620 | baseline |
| 64 | 2 WG/head、64 useful threads/WG | 1.413 | 0.840 | 両target悪化 |
| 128 | 4 WG/head、32 useful threads/WG | 1.397 | 0.693 | 両target悪化 |
| 256 | 8 WG/head、16 useful threads/WG | 1.418 | 0.702 | 両target悪化 |

64以上のcoreだけを見るとV620 128-workgroupで約2.1%短縮したが、output finalize追加分を含むGDN familyでは悪化した。
この結果は単純なrow splitを棄却するが、全waveに有用なtileを割り当て、重複準備を除いた構造を棄却しない。

## Candidate inventory

### C1: prepared Q/K + useful row tile

- Q/K normとBF16 RNEをqk headごとに一度だけprepareする。
- recurrent coreはtile幅に合うworkgroup sizeを使い、64-row tileなら64 threads、32-row tileならwave32を全thread有効にする。
- 2/4 tile per value headのpartial output norm sumをbounded scratchへ書き、headごとに一度だけoutput norm/gateをfinalizeする。
- GDN family合計でprepare/core/finalizeの追加launchを含め、現行fused 32-workgroup baselineと比較する。

### C2: wave column tile + deterministic reduction

- state columnまたはcolumn groupをwaveへ割り当て、各laneが複数rowを担当して連続state accessと有用wave数を増やす。
- previous/current projectionのpartial sumは固定順序でreduceし、atomicの非決定順序を使わない。
- current providerのtarget別state index contractを維持し、state transposeだけの既棄却candidateを再提案しない。
- provider-private layoutを変える必要がある場合はrequest開始時からそのlayoutでstateを保持し、decode stepごとのtransposeを追加しない。

### C3: fused tile finalize

- C1/C2でprepare/finalizeの固定費が支配的な場合だけ、cooperative launch、同一block tile、またはpersistent head assignmentで
  finalizeを再融合できるかboundedに検討する。
- global synchronizationの成立を仮定せず、portableでないgrid barrierやunsafe busy-waitを使用しない。
- C1/C2のmechanism結果なしにC3へ進まない。

### Variant方針

- compile-time variantはworkgroup size、row/column tile、waves/headを明示し、runtime autotuning DBを作らない。
- 最初のbounded matrixはbaseline、C1 row64、C1 row32、C2 column-waveの最大4 variantとする。
- total workgroup数64/128/256を目的にせず、有用thread率、waves/CU、state bytes/wave、reduction費からvariantを定義する。
- gfx1030/gfx1201で同じvariantが勝つ場合はshared mapping、異なる場合はexact-target provider mappingを使う。

## Fresh baseline・干渉防止contract

- source/tree、release binary、ROCm 7.14.0、LLVM 23、exact target/code object、GPU UUID/BDF、model lock、GGUF、runnerをdigestへ結ぶ。
- local Qwen serviceを停止し、single-V620構成へfallbackしない。
- targetをUUIDで一台だけ可視化し、性能runは一度に一GPUだけ実行する。他GPUで同時benchmark/profileを行わない。
- 各processのpre/during/postでforeign GPU process、temperature、clock、power、performance level、ECC、VRAM、loader rootを取得する。
- foreign active process、ECC、GTT spill、loader drift、明示throttle、cleanup残留があるrunを採否へ使わない。
- power/clockを変更せず、観測不能fieldを0として扱わない。dynamic clockの単発差だけを後付けhard gateにしない。
- profiler laneとprofilerなしwall laneを別processにし、rocprofv3 observer effect下のwall値を採否へ使わない。

## 受入基準

### Correctness

1. host descriptor/provider selectionはactual shapeに加え、`M=1,2,3`、非整列値、zero、overflow、unsupported layoutを検証する。
2. model-free GPU oracleは両targetでtoken count 1、3、17を実行し、finite分類、BF16 RNE stage、output order、padding非破壊を照合する。
3. previous/next recurrent state、expected/committed length、double-buffer swapをstepごとに照合する。
4. partial submission、timeout、cancel、query failureではnext stateを公開しない。retryable/durable cleanupを0へ戻す。
5. baseline/candidateでQwen target-onlyのprompt/completion token IDs、bounded logits、visible output、stop、usageを一致させる。
6. R9700 current MTP、prefill 17/255/256/257、fixed-seed sampling 3 profileをregression controlとして実行する。
7. CPU/backend fallback、別target code object、timeout、crash、zero test selectionをPASSにしない。

### GDN device performance

8. B0/B1/B2の各scopeでprimary metricを3 independent process以上取得し、baseline bracket driftを確認する。
9. shared adoptionは両target・scope内全validation patternでGDN p50非悪化、任意代表patternで5%以上短縮を要求する。
10. scoped adoptionは該当exact targetのscope内全patternでGDN p50非悪化、任意代表patternで5%以上短縮を要求する。
11. scope外はbaseline provider symbol、dispatch、GDN p50にstableな悪化がないことを確認する。
12. candidateがprepare/reduce/finalize/copyへ仕事を移した場合、その全kernelをGDN familyへ再分類して判定をやり直す。
13. operator microbenchmark、core kernel単体、workgroup count、CU occupancyだけでは採用しない。committed decode stepのGDN family合計を使う。
14. full-model TPOT/E2Eが改善しなくてもGDN条件を満たせば採用できる。悪化を含むwall結果は省略せずdiagnosticとして報告する。

### Resource・architecture

15. FP32 state、BF16境界、GGUF、public API、semantic graph順序を変更しない。
16. scratch、VGPR/SGPR/LDS、occupancy、grid/workgroup/wave、state traffic、launch数をbaseline/candidateで記録する。
17. decode stepごとのunbounded allocation、state transpose、host readback、device-wide synchronizeを追加しない。
18. request workspace増加はactual bytesと上限を記録する。workspace増加だけを自動棄却条件にしないが、allocation failureはhard failureとする。

### Evidence・closeout

19. raw trace、binary、model、full logits、生成全文を追跡しない。bounded aggregate、digest、runner/schema/test、plan/historyだけを残す。
20. affected host/build/GPU/full-model diagnostic、one integration review、changed findingだけのfocused re-reviewを行う。
21. 採用時はruntime/compatibility/provenance/main planを同期する。未達時はcandidateをproduction defaultへ残さずnegative completionできる。

## 計測matrix

| case | workload | 目的 | 採否への使用 |
| --- | --- | --- | --- |
| H0 | provider actual shape + boundary | descriptor、selection、scratch contract | correctness |
| G0 | token 1/3/17、両GPU | numerical/state oracle | correctness |
| B0 | prompt 17 + output 16、target-only | short-context GDN p50/p90 | hard GDN performance |
| B1 | prompt 28 + output 16、target-only | Phase 28 continuity | hard GDN performance |
| B2 | prompt 255 + output 16、target-only | same-scope context control | hard GDN performance |
| C0 | actual GDN shape + counters | traffic、VALU/VMEM、occupancy、stall attribution | mechanism |
| D0 | scope内代表値とscope外provider | routing coverage、candidate leakage | hard adoption |
| M0 | R9700 current MTP | target/draft/verify correctness | diagnostic correctness |
| F0 | prompt 17/28/255 + output 128 | TPOT、TTFT、E2E転化 | diagnostic only |
| P0 | prefill 17/255/256/257 | prefill/state regression | correctness |
| R0 | resident/workspace high-water | resource accounting | resource |

## 作業順序

### P29-A0: acceptance・metric freeze

- Phase 29だけ5% thresholdの対象をfull-modelからGDN device p50へ置き換える決定をmain plan、active plan、summary schemaへ固定する。
- GDN family kernel classification、committed step境界、prepare/reduce/finalize/copy包含規則をtestへ固定する。
- shared/scoped adoption key、primary target-only scope、context validation pattern、wall diagnosticの非gate扱いをmanifestへ記録する。

### P29-A1: interference-controlled fresh baseline

- Phase 28 production sourceからtarget別release binaryを作り、G0とB0/B1/B2を実行する。
- Qwen service停止、single-GPU sequential run、foreign process/clock/temperature/power/ECC/VRAM/loader snapshotを確認する。
- baseline p50/p90、layer分布、resource、counterを取得し、探索時の1.354/0.620 msとの差をrun drift、tool差、source identityへ説明する。

### P29-A2: useful-work prototype比較・candidate freeze

- baseline、C1 row64、C1 row32、C2 column-waveをmodel-free/actual-shape prototypeとして比較する。
- useful thread率、waves/head、state bytes/wave、prepare/core/finalize device time、VGPR/SGPR/LDS/scratchを分解する。
- 両targetでGDN family 5%へ届くcredible variantを最大一つずつ選ぶ。同じvariantならshared、異なればtarget-scoped候補とする。
- どのvariantも5%へ届かない場合はproduction統合へ進まずnegative closeoutする。

### P29-A3: provider/state contract・GPU oracle

- selected variantのprovider key、workspace、state layout/index、failure/cancel contractをhost testへ固定する。
- G0をbaseline/candidateで両GPU実行し、token 1/3/17、distinctive state、BF16 boundary、state publicationを照合する。
- llama.cppコードを直接reuseする場合は、実装前にexact source/path/hash/reuse modeをprovenanceへ記録する。

### P29-A4: bounded production implementation

- A2で固定したprepare/core/finalize構造だけをproduction registryへ実装し、baseline providerをrollback用に維持する。
- common semantic descriptorを保ち、exact-target差はprovider mappingへ限定する。
- projection、causal convolution、attention、MTP algorithm、GGUF/KV format、schedulerを変更しない。

### P29-A5: correctness・mechanism verification

- H0/G0/P0/M0、cancel/failure/cleanupを実行する。
- C0でstate traffic、useful waves、occupancy/stall、prepare/core/finalize、projection/conv controlを再取得する。
- wrong state、別familyへの仕事移動、unbounded scratch、fallbackがあればcandidateを棄却する。

### P29-A6: GDN-only adoption decision

- B0/B1/B2をsingle-GPU、3 process以上、counterbalanced bracketで両target実行する。
- scope内全pattern非悪化かつ任意代表pattern5%以上なら、そのscopeへcandidateを採用する。
- 一方だけ合格ならexact-target scoped adoption、両方合格ならshared adoption、両方未達ならbaselineを維持する。
- D0でscope外がbaselineへrouteされることをprovider symbol、dispatch、GDN p50で確認する。
- F0をprofilerなしで取得してwall転化を報告するが、Phase 29の採否を変更しない。

### P29-A7: integration・closeout

- bounded summary/schema/test、affected validation、integration reviewを完了する。
- main plan、history、runtime、compatibility、provenanceを実結果へ同期し、planをarchiveする。
- 採用しないvariantとtemporary dispatchをproduction sourceから除去する。
- 次Phaseを自動開始しない。

## Rollback・停止・再計画

- wrong token/logit/state、fallback、GTT spill、cleanup failure、unbounded allocationはcandidateを採用しない。
- GDN p50が5%へ届かないcandidateをfull-model偶然差、core単体値、CU occupancyだけで救済しない。
- GDN条件を満たしたcandidateをfull-model改善不足だけで棄却しない。wall悪化は明記してユーザーへ返す。
- 16/64/128/256の単純分割を同じ構造のまま再実行しない。新しいuseful-work mappingまたは干渉証拠がある場合だけcontrolに使う。
- 同じwork unitが2回reject、review時間が実装時間超、functional progressが1時間停止、verification/docsが30%超、
  見積り1.5倍超、acceptance変更時は追加variantを止め、同じwork unitを再計画する。

## 実行結果

- full Q/K/output wave reductionは、B0/B1/B2でV620を2.15〜2.20%、R9700を8.10〜9.21%短縮し、
  Phase固有のGDN-only性能条件を満たした。
- model-free oracleとoutput 16は一致したが、output 128の6 pattern中5 patternでbaselineとtoken列が分岐した。
- Q/K-onlyはR9700 B1で5.92%改善し、output-onlyは1.65%で5%未達だった。
- 初回closeoutではtoken完全一致条件によりcandidateを棄却した。その後のユーザー明示規則変更で、全て非負の二乗和を
  逐次深さ127から固定tree深さ概ね8へ変える本変更を、real-number semantic不変かつ解析的誤差bound低減のN1へ分類した。
- N1ではtoken差を台帳へ記録して数値gateを自動承認するため、full Q/K/output wave candidateをtarget splitなしでshared採用した。
- この改訂は上記Correctness 5のtoken完全一致をN1に限り観測項目へ変更する。state、fallback、cleanup等のhard conditionは維持する。
- 詳細値、binary/source identity、full-model診断はhistoryとbounded summaryを正本とする。

[Phase 28計画](../../../../archive/2026/08/11-20/phase28-decode-nonprojection-device-optimization.md)
[Phase 28 bounded summary](../../../../../../ci/matrix/phase28-nonprojection-summary-v1.json)
[Phase 29実行履歴](../../../../../history/2026/08/11-20/phase29-gdn-useful-workgroup-parallelization.md)
[Phase 29 bounded summary](../../../../../../ci/matrix/phase29-gdn-device-summary-v1.json)
[数値・出力影響変更台帳](../../../../../compatibility/numerical-output-changes.md)
[メイン計画](../../../../main-plan.md)

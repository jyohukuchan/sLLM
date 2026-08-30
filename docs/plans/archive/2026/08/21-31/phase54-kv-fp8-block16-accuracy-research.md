# Phase 54: KV FP8 block16精度改善研究

> 状態: 完了（2026-08-27、`no-improvement`。FP16 default／空mapping／descriptor v2を維持）
> primary target: exact `gfx1030`／V620／`kv-fp8-e5-block16`
> transfer target: exact `gfx1201`／R9700／`kv-fp8-e4-block16`
> MI300X: 候補が絞れ、他の実機項目と一括検証できるまでdeferred

## 目的

Phase 53で実装した16値単位のKV FP8形式を維持しながら、block16がstandard MXFP8 block32より最終logit品質で優位になる
recipeまたは固定変換を研究する。探索の主対象は差が大きくローカル実機を利用できるE5M2／gfx1030とし、再利用可能なfinalistだけを
E4M3／gfx1201へ移植評価する。

このPhaseは「候補を必ず採用する」ことを完了条件にしない。原因を特定し、候補を同一条件で順位付けし、勝者がなければFP16 defaultと
production descriptor v2を維持したまま棄却結果を閉じる。default変更は研究結果ではなく、Phase 46由来のfreeze済みpolicyを通過した
finalistに限る。

## 固定する製品制約

- canonical public nameは`kv-fp8-e4-block16`／`kv-fp8-e5-block16`のままとし、探索candidate名をpublic aliasへしない。
- 1 scaleは同一token、同一KまたはV plane、同一KV headのhead-dimension方向16値だけで共有する。
- tokenを跨ぐper-channel scaleとper-channel runtime処理は実装しない。scale方向はper-token block16だけとする。
- CDNA3／RDNA4はE4M3系、RDNA2はE5M2とする。OCP、FNUZ、software codecを混同しない。
- block size、E8M0 scale byte、packed value plane、K/V独立plane、BF16 append入力、FP32 attention accumulatorは維持する。
- FP16、candidate block16、MXFP8は同時常駐させず、各residentを解放してから次の経路を測定する。
- 探索中は省略時FP16、空default mapping、production descriptor v2／`StandardMxFloorPowerV1`を変更しない。
- stochastic rounding、request依存recipe、prompt依存routing、実行途中のencoding変更はstate再現性を弱めるため対象外とする。

staticなlayer別recipe、K/V別recipe、固定permutation／orthogonal transformは、実行時scaleが上記per-token block16のままであれば
研究対象にできる。ただし採用時はmodel metadata、descriptor version、state identityへ明示的に結合し、descriptor v2として偽装しない。

## Phase 53から引き継ぐ基準値

同じQwen3.5-4B BF16 model lock、derived lock、Phase 46 dataset／metricで比較する。

| target / candidate | KLD p99 | top-1 | long-context loss | 状態 |
| --- | ---: | ---: | ---: | --- |
| gfx1030 E5 block16 descriptor v1 | `0.04331390780013198` | — | — | superseded |
| gfx1030 E5 block16 v2 `StandardMxFloorPowerV1` | `0.03659844555378746` | `0.9` | `0.08333333333333337` | production control |
| gfx1030 standard MXFP8 E5 block32 | `0.03218873133110086` | `0.8` | `0.16666666666666663` | explicit comparator |
| gfx1030 local-MSE／parent32-guard | `0.04063529273873547` | `0.8` | `0.16666666666666663` | 棄却 |
| gfx1030 normalized fixed H16 | `0.03659844555378746` | `0.9` | `0.08333333333333337` | controlと同一、棄却 |
| gfx1201 E4 block16 v2 `StandardMxFloorPowerV1` | `0.006562189165612111` | `0.9` | `0.16666666666666663` | transfer control |
| gfx1201 standard MXFP8 E4 block32 | `0.004945428206833837` | — | — | transfer comparator |

H16診断ではK byte／scaleが実際に変化し、V不変、Q/K同時変換後のdirect attention numerical matchも確認した。したがって固定H16の
同一値は未適用ではなく、この評価集合で品質改善がなかった結果として扱う。診断raw reportは既存v2 schemaを再利用してcandidate identityを
持たないため、正式aggregateへ入力しない。

## 成功条件と判定区分

### 研究candidateの相対精度成功

E5M2 candidateは同じ一回測定で次をすべて満たす場合だけfinalistとする。

1. production block16 v2よりKLD p99が低い。
2. standard MXFP8 E5 block32のKLD p99 `0.03218873133110086`を下回る。
3. production controlに対してtop-1、task、long-contextのいずれも悪化しない。
4. finite output、HIP-only、fallback 0、cleanup 0を満たす。

探索一回の値は相対順位付けだけに使い、default採用証拠にはしない。同値は勝利と数えない。

### 正式採用

finalistはfresh immutable binaryで3 repeatを取得し、Phase 46由来の現行`kv-cache-default-v2` thresholdを変更せず評価する。
correctness、quality、resource、性能、state identityを満たしたexact targetだけ`adopt`できる。研究成功でもfrozen policy未達なら
`research-win-retain-fp16`とし、defaultへ昇格しない。

Phaseは`adopt`、`research-win-retain-fp16`、`no-improvement`のいずれでも完了できる。未達を隠すためdataset、percentile、sample、
baselineをcandidate確認後に変更しない。

## 作業単位

### A. candidate identityを持つ研究harness

1. 研究reportへ`candidate_id`、scale selector、rounding、K/V recipe、transform、calibration digest、descriptor compatibility、binary SHAを追加する。
2. diagnostic candidateをproduction descriptor v2／`StandardMxFloorPowerV1`として報告しない。正式descriptorを増やす前でも、report上の
   experiment identityを一意にする。
3. per-case KLD、top-1、NLL、long-context、最初のtoken／logit分岐、最大logit差と有限性を保存する。raw logitsや巨大traceは追跡しない。
4. runnerは探索時1 repeat、finalist時3 repeatを明示選択できるようにし、FP16→candidate→MXFP8の完全直列解放と
   terminal-zero cleanupを機械的に検証する。
5. production controlとMXFP8 comparatorを同じbinary、model、dataset、GPU session条件で再取得し、過去値だけとの比較にしない。

### B. 劣化箇所のattribution

1. boundedな固定caseについて、K-only量子化、V-only量子化、K+V量子化を分離して最終logitへの寄与を測る。
2. layerごとにK/V再構成誤差、QK score差、softmax確率差、attention output差、residual差を収集し、最終KLDとの相関を確認する。
3. scale exponent、amax、飽和、underflow、zero化、block16前半／後半と対応するMXFP8 block32 parentの分布を取得する。
4. short／longおよび15／16／17、31／32／33、255／256／257境界を含める。CPUだけの結果をGPU品質PASSにしない。
5. instrumentationは研究build限定とし、通常runtimeへ常時traceや同期copyを残さない。

この作業で「K、V、scale exponent、block境界、layer」のどこが主要因かを決めてから候補waveを選ぶ。全候補を総当たりしない。

### C. wave 1: strict block16 scale／rounding候補

優先順位はattribution結果で更新できるが、最初の候補集合は次とする。

1. `Floor` production control、`Ceil`、`NearestEvenExponent`を同一codecで比較する。
2. `e16-1/e16/e16+1`から、値SSEではなくQK scoreまたはattention output proxyを最小化する決定的selectorを試す。
3. KとVで別の有限recipeを選ぶ。KはQK／softmax誤差、Vはattention output／O projection感度を目的にするが、実行時scale axisは
   どちらもper-token block16のままにする。
4. calibrationでlayerごとに有限enumからrecipeを一つ選び、実行時探索を避ける。calibration promptとPhase 46評価datasetは分離する。
5. clippingを使う場合はblock内の決定的規則に限定し、NaN／Inf／signed zero／subnormalのcanonical contractを維持する。

local value MSEとparent32 exponentだけの候補はPhase 53診断で棄却済みのため、そのまま再実行しない。

### D. wave 2: attention-aware固定変換

wave 1でMXFP8を上回れない場合、またはattributionがchannel配置を主要因と示した場合に進む。

1. Q/Kへ同じ固定permutationまたはorthogonal transformを適用し、量子化前のQK積を保存する。
2. V側を変換する場合は対応する逆変換をO projectionへfoldし、量子化前モデルの意味を保存する。
3. per-layerの固定block packing／permutationをcalibrationで選び、outlierと通常値の16値groupingを調整する。
4. normalized H16単体は棄却済みcontrolとする。再利用する場合はpermutation、layer選択、K/V別処理など新しい仮説を一つだけ加え、
   何が効いたか判定可能にする。
5. preprocessまたはprojectionへfoldできる候補を優先し、attention内で各queryごとに16x16変換を再計算する実装は性能比較で費用を開示する。

固定変換はper-channel scaleではないが、model固有metadataとなる。採用候補はmodel lock／derived artifact／KV descriptor／checkpointへ
transform digestを結合し、異なる変換のstate reuseを拒否する。

### E. 候補の段階選別

1. host codec／数学oracleで非整列block、special value、QKまたはV/Oの変換前後不変性を確認する。
2. exact gfx1030のdirect GPU append／attention oracleでbyte、数値、dispatch、fallback、cleanupを確認する。
3. oracleを通過した候補だけQwen3.5-4B品質を一回測定する。controlより悪い候補、MXFP8未達、別metric悪化候補を棄却する。
4. 非支配finalistが複数ある場合はKLD、top-1／long-context、attention kernel費用、metadata量で比較し、最大2候補へ絞る。
5. 最大2候補だけfresh 3-repeat品質とresource／性能を取得する。同じ失敗candidateを反復回数だけ増やして再評価しない。
6. E5 finalistだけをgfx1201 E4へ移植し、E4のcontrol／MXFP8と同じ順序で一回比較する。target固有採否を許し、E4未達でE5を棄却しない。

探索用GPU実行はローカルV620/R9700で行う。MI300X `gfx942:sramecc+:xnack-`はfinalist、correctness matrix、測定項目が揃った後に、
他のMI300X検証と一括実行する。compile成功や他targetの結果をMI300X PASSへ読み替えない。

### F. finalist実装と採否

1. 勝者が現行recipeと意味的に異なる場合はdescriptor v3候補を追加し、v2 stateをhit／importしない。
2. public nameを維持しつつresolved descriptor、recipe、transform digest、selection reasonをreportとstate identityへ残す。
3. prefix cache、checkpoint、fork/COW、context shift、grow、offload/importの互換性とrollbackを検証する。
4. exact targetのquality policyをPASSした場合だけ通常5行＋長時間2行、logical／physical KV bytes、HBM/GTT settledを取得する。
5. default mappingの変更はtarget別N2 decisionとして数値変更台帳へ記録し、explicit FP16とdescriptor v2 rollbackを残す。
6. 勝者がなければdiagnostic dispatchを通常build／製品経路から除外し、production v2、空mapping、FP16 defaultを維持する。
   再現用sourceは明示的なresearch compile feature限定で保持できる。

## 対象外

- per-channel scale、tokenを跨ぐ統計、request内容によるscale選択。
- block size 16の変更、INT8／INT4／vector codebook／TurboQuantをblock16と称する置換。
- sparse outlier sidecar、low-rank residual、sink/recent FP16とのmixed storage。これらは別format／別Phase候補であり、strict block16の
  勝者がないことだけを理由に本Phaseへ自動追加しない。
- 新model family、training／fine-tuning、weight quantizationの同時変更。
- MI300X VMの新規確保と単独実行。ユーザー指示どおり検証項目がまとまるまで延期する。
- Phase 47の組込みtool/MCP、Phase 48のWebUI。

## 停止・再計画条件

- correctness/security defect、量子化前QKまたはV/O意味の不一致、fallback、cleanup残留があれば該当candidateを停止する。
- 同じwork unitが2回棄却、検証／docsが作業量の30%超、実装時間が見積りの1.5倍、または一時間以上functional progressが止まった場合は
  新候補を追加せずattributionへ戻る。
- 候補がper-channel scale、sidecar、公開format変更を必要とすると判明した場合はPhase 54へ暗黙拡張せず、別計画をユーザーへ提示する。
- frozen policy、dataset、model lock、metric集約を変更する必要が出た場合は既存結果と混ぜず、新policy提案として再計画する。

## 完了条件

- identity-safeなresearch harnessとK/V・layer・attention段階のattribution結果がある。
- 試した各candidateについて仮説、実装差、binary／model／dataset identity、direct GPU oracle、一回品質値、採否理由が残る。
- 最大2 final candidateへ絞り、該当する場合だけfresh 3-repeatとresource／性能を取得する。
- exact gfx1030について`adopt`、`research-win-retain-fp16`、`no-improvement`のいずれかを決定する。
- transfer可能なE5 finalistがある場合はgfx1201 E4を一回評価し、MI300X deferred項目を具体的な実行matrixへ追加する。
- 通常build／製品経路に診断分岐を入れず、採用時は新identityとrollback、非採用時はdescriptor v2／空mapping／FP16 defaultを維持する。
  再現用sourceを保持する場合は明示的なresearch compile featureへ隔離する。
- matching historyを作成し、このplanをarchiveへ移し、main planとroadmapを更新する。

## 非blockingな研究提案

finalistの性能選別ではattention時間のcontrol比`+5%`以内を望ましい目安とする。これはAI起案の研究優先順位であり、採用のhard gateではない。
scopeはPhase 54の候補絞り込みだけ、費用はfinalistごとの短い性能測定、expiryはPhase 54のtarget別採否時点とする。

## 進捗

- 2026-08-27: Phaseを開始した。production v2 contractを流用しないPhase 54専用report、1／3 repeat、同一binary内の
  FP16→production control→candidate→MXFP8完全直列実行をWork Unit Aの最初の実装単位とした。
- K/V-onlyのproduction mixed-plane stateは一つのencodingを共有する現行ABI／storageを広く変更するため、最初のattributionには
  採用しない。research feature限定のFP16-state block16 roundtrip surrogateを使い、K-only／V-only／K+Vの最終logit寄与を分け、
  actual production block16 K+Vとの一致を確認してからlayer展開する。
- wave 1はproduction Floorを同一binary controlとして残し、research-only append kernelでK/V別のCeil／NearestEvenExponentを
  選べる形を先に作る。candidate identityと実際のdispatchが一致しないreportは発行しない。
- same-binary探索1回とPhase 54 direct GPU oracleを完了した。Floor/Floorはproductionの全logitを完全再現し、Ceil／NearestEvenは
  K-onlyでtop-1を`0.9`から`0.85`へ、V-onlyでKLD p99を`0.03659844555378746`から`0.04331390780013198`へ悪化させた。
  4候補はMXFP8 `0.03218873133110086`にも届かず棄却し、組合せ総当たりへ進まない。GPU oracleは5 recipe pairすべてでK/V byte・
  scale exact、signed-zero-only、非整列境界、KV長2／非zero queryのK-sensitive attention、fallback 0、cleanup 0をPASSした。
- attribution用のproduction mixed-plane ABIは追加せず、feature限定のFP16-state block16 roundtripを最初のfull-attention layer 3へ
  注入する実装を追加した。K-only／V-only／K+Vのexact GPU最終logitを取得してから、wave 1のattention-aware selectorまたはwave 2の
  固定permutationへ進むかを決める。
- 同一attribution binaryでfull-attention 8層×K-only／V-only／K+Vの24 runを完了した。K-only KLD p99は全層
  `0.0003981–0.0010435`、V-onlyはlayer 19が`0.0398337`と突出し、layer 31もtop-1 `0.8`だったため、V、とくにlayer 19を
  主因と判定した。この測定は単層FP16-state roundtrip surrogateであり、全層production block16再現とは区別する。
- wave 2 control `phase54-kq-transpose16x16-all-full-v1`はdirect GPU oracleでK byte／scale exact、V不変、QK最大差`0`、
  fallback 0、cleanup 0をPASSしたが、一回品質はKLD p99がproductionと同じ`0.03659844555378746`、top-1が`0.8`へ悪化し、
  MXFP8 `0.03218873133110086`にも届かなかったため棄却した。次はlayer 19だけのV/O意味保存permutationを評価する。
- layer 19 V/O candidateはdirect GPU oracleをPASSし、一回品質KLD p99を`0.033918254226008415`へ改善した。top-1／task／
  long-contextはproduction同等だが、MXFP8 `0.03218873133110086`を下回らずfinalist条件未達である。attributionで次に寄与した
  layer 31を加えた`[19,31]`候補を最後の最小追加waveとして一回評価する。
- layers 19+31 V/O candidateもdirect GPU oracleをPASSし、KLD p99を`0.03337377972334127`へ改善したが、MXFP8未達に加えて
  top-1が`0.8`へ悪化したため棄却した。採用条件を満たす候補はなく、exact gfx1030を`no-improvement`で閉じる。finalistがないため
  3-repeat、resource／性能、gfx1201 transfer、MI300Xは実行しない。FP16 default／空mapping／descriptor v2を維持する。
- 完了後のユーザー指示によるfollow-upとして、32値のMXFP8 scaleを2個のblock16 childへ複製するresearch-only
  `parent32-duplicate`を追加した。OCP E4／E5 host oracle、exact gfx1030／gfx1201 direct GPU oracleをPASSし、Qwen3.5-4Bの
  全prefill／decode logitはsame-run MXFP8とFP32 bit列で完全一致した。これはMXFP8完全再現controlであり精度優位候補ではないため、
  Phase 54の`no-improvement`、production descriptor v2、FP16 default、空mappingは変更しない。
- V620での最初のE4M3／E5M2 attention比較はformat-neutralなscalar decoder同士で、E5M2がshort `1.885%`、long
  `3.431%`遅かった。ただしE5M2→FP16が単なる8 bit左shiftである性質を使っておらず、format選定には無効なbaselineと訂正した。
  E5M2をexact FP16 bitcast＋`v_cvt_f32_f16`、E4M3もnormal値のFP32 bit直接構築へ最適化したresearch-only再測定では、
  全case／全pairでE5M2が速く、short 6 caseの幾何平均で`5.259%`、long 6 caseで`11.870%`高速だった。全20 reportの
  numerical oracle／fallback 0／cleanup 0と旧新output hash一致を確認した。attention単体の結果なのでproduction mappingは維持し、
  次の判断単位を同一V620でのfull-model品質とappend込みend-to-end throughput A/Bとする。

[全体計画](../../../../main-plan.md) /
[Phase 37以降のロードマップ](../../../../active/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md) /
[Phase 53保存済み計画](../../../../archive/2026/08/21-31/phase53-kv-fp8-block16-default-adoption.md) /
[Phase 53履歴](../../../../../history/2026/08/21-31/phase53-kv-fp8-block16-default-adoption.md) /
[Phase 54履歴](../../../../../history/2026/08/21-31/phase54-kv-fp8-block16-accuracy-research.md)

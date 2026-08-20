# Phase 35: long-context Full Attention・GDN構造最適化

> 状態: 計画済み・未着手
> 作成日: 2026-08-20

## 目的

Phase 34後のfixed Qwen3.5-4B BF16、FP16 KV、10,001 input / 2 outputをcurrent sLLMとfixed llama.cppで
細粒度比較すると、sLLMのprojectionは既にpeerより短い一方、Full AttentionとGDNが残るdevice差の大半を占めた。
Phase 35はこの二つを同じPhase内の独立trackとして構造的に最適化し、projectionの長行hipBLAS利益を維持したまま
long-context TTFTを短縮する。

1. Full Attentionを1 query row単位のK/V走査から、複数query rowとGQA headがK/V tileを共有するproviderへ移す。
2. GDN recurrent coreをvalue head当たり1 workgroupからstate column並列へ移し、state shardをregisterに保持して
   GPU全体を利用する。
3. 各trackを単独で比較・採否した後、合成candidateでfull-model転化とfixed llama.cppとの差を再測定する。

Phase 35の目的はllama.cpp sourceを一括移植することではない。既存sLLMのsemantic op、transaction、vAttention、
chunked-prefill arena、数値分類を維持し、peer profileから確認できたprovider topologyの差をbounded candidateとして扱う。

## 開始根拠

### Phase 34後の同条件profile

canonical V620 exact `gfx1030`、Qwen3.5-4B BF16、FP16 KV、exact 10,001 input token / 2 outputの
rocprofv3 aggregateは次のとおりだった。sLLMはPhase 34 candidate、llama.cppはcommit
`f5919bf458ef190468b5c329bb293f8a54a1e69c`、33/33 layer offload、`n_batch=10001`、`n_ubatch=512`である。
GGUF byte identityは異なるためfull-model比較はE1 system-equivalentであり、operator topologyと絶対device差を実装候補の
根拠に使い、厳密な同一engine contractとは表記しない。

| family | sLLM device | llama.cpp device | sLLM/llama | 差 |
| --- | ---: | ---: | ---: | ---: |
| projection | 11.781 s | 12.772 s | 0.92x | -0.992 s |
| Full Attention | 10.857 s | 0.462 s | 23.48x | +10.395 s |
| GDN | 7.694 s | 0.622 s | 12.36x | +7.072 s |
| other GPU | 0.167 s | 0.136 s | 1.23x | +0.031 s |
| GPU kernel合計 | 30.499 s | 13.992 s | 2.18x | +16.506 s |

- profilerなしE2E中央値はsLLM `34.684 s`、llama.cpp `14.605 s`で、sLLMは2.375倍だった。
- Full Attention差はprofiled E2E差の約50.6%、GDN差は約34.4%を説明する。二つをpeer device timeまで短縮できた場合の
  sLLM GPU kernel楽観値は約`13.03 s`である。これは目標の方向を示すAmdahl estimateであり、採用thresholdではない。
- profile上のkernel外差約4秒は、sLLMがmessages/tokenizationを含みllama wrapperが直接token IDsを受け取る境界差を含む。
  Phase 35はこれをFull Attention/GDNの利益へ混ぜず、direct-token対応laneを診断controlとして分離する。
- projectionはsLLMがpeerより約7.8%短い。全prefillをllama.cpp同様の512-row graph chunkへ分割するとPhase 34のlarge-M
  hipBLAS利益を失う可能性があるため、projectionはfull-Mのまま維持し、stateful op内部だけをtile化する。

### 機構差

現行Full Attention C2は1 workgroupが1 query rowと1 KV headを所有し、GQA 4 headでK/V elementを共有するが、
別query rowとの共有はない。各rowはcausal endまでK/Vを走査し、keyごとに複数のworkgroup barrierを使う。
fixed llama.cppのtile providerは複数queryをまとめ、K/V tileとonline-softmax partialを再利用する。

現行GDN recurrent kernelは32 value headに対して32 workgroupだけを起動し、各threadが128 state elementをtoken順に
二回走査する。10,001 tokenでもworkgroup数は増えず、64/72 CUを埋めない。fixed llama.cppのcurrent providerは
wave32 x 4のworkgroupで4 state columnを所有し、gridのcolumn軸を32 groupへ広げるため、Qwen shapeでは
`32 heads × 32 column groups = 1,024 workgroups`となる。各waveは1 columnのstate row shardをregisterへ保持し、
token recurrenceだけを直列に進める。

## 権限・採否方針

- 固定改善率、5%規則、全pattern一律非悪化gateは使用しない。担当AIがadoption scopeごとに、絶対短縮時間、割合、
  confidence、一貫性、full-model転化、数値分類、resource、target分岐、実装・検証・保守費用、再利用性、rollbackを総合して採否する。
- Full AttentionとGDNは独立採否とする。一方が棄却またはN2判断待ちでも、他方の比較・採否を巻き戻さない。
- exact gfx1030/gfx1201で共通のsemantic、routing、state/publication、tile descriptorを優先する。target別thresholdまたは
  inner mathは実測利益が保守費用を上回る場合だけ既存registry keyへ閉じ込める。
- N0/N1は数値gateを通常または自動承認できる。N2は誤差、token、速度、scope、rollbackを提示してユーザー判断を得るまで
  production defaultへ接続しない。N3は採用しない。
- 本計画はcandidate実装とGPU実行を開始する権限をまだ与えない。ユーザーのPhase 35開始指示後に実装へ進む。

## Phase固有の範囲

### 対象

- primary model: fixed Qwen3.5-4B dense BF16 GGUF/derived lock。
- target: canonical V620 exact `gfx1030` UUID `GPU-76a08c022586fed6`、R9700 exact `gfx1201`
  UUID `GPU-a8e9ddefa2d60f55`。
- Full Attention shape: 8 layer、Q heads 16、KV heads 4、head dim 256、GQA ratio 4。
- GDN shape: 24 layer、Q/K heads 16、value heads 32、state/head dim 128、BF16 activation、FP32 recurrent state。
- mode: normal text prefillをprimary、long-prefix decode、chunk tail、MTP verify/replayをcorrectness/regression controlとする。
- KV encoding: Full AttentionはFP16をperformance primary、dynamic FP8をlow-bit primary、static FP8/NVFP4を
  direct-read correctness/resource controlとする。GDN stateはKV encodingから独立して同じproviderを使う。
- baseline: Phase 33 C1/C2、Phase 28 state-pass fusion、Phase 29 wave reduction、Phase 31 arena/chunk、Phase 34 shape-aware hipBLASを含む。
- fixed llama.cppは同一commitの`fattn-tile`/combineと`gated_delta_net_cuda<128,...>`をmechanism、launch geometry、
  profiler categoryの比較対象にする。直接adaptするsource expressionがある場合だけprovenance手順を適用する。

### 非対象

- Phase 34 projection、terminal LM head、Argmax、RMSNorm、一般elementwise、host/tokenizer/service residualの最適化。
- KV layout/providerのvAttentionからPaged Attentionへの変更、KV format/default、TurboQuant、prefix sharing、RadixAttention。
- DeepSeek V4/DFlash/MLA、Gemma sliding-window、MoE、continuous request batching、multi-GPU、Infinity Fabric/RDMA。
- GGUF/model-lock/public API変更、全score matrix、host softmax/combine、CPU fallback、requestごとのdevice allocation。
- GDN token recurrenceのsequence-parallel scan。Phase 35はstate column並列を対象とし、数学的prefix-scanへの置換は別候補とする。
- 全graphを512 tokenへ強制分割してprojectionを再実行する方式。stateful opだけを内部tile/chunk化する。
- gfx1200、gfx942、別SKU/model/head shapeへの性能一般化。gfx942は影響sourceのcompile-only controlに留める。

## 共通不変条件

- public semantic op descriptor、Rust/C++ versioned ABI、prepared execution、completion、transactional publicationを維持する。
- Full Attentionはcausal scaled-dot-product、GQA mapping、online softmax、BF16 RNE output、opaque KV stateを維持する。
- GDNはQ/K normalization、BF16 rounding stage、beta、decay、state update、projection、output RMSNorm、z SiLU、
  accepted-prefix state publicationを維持する。
- candidateのscratch/intermediateはPhase 31 request arenaのchecked lifetimeへ置く。model-resident duplicate、per-dispatch
  `hipMalloc`、full KV/state mirrorを作らない。
- providerはdispatch前のstable keyで選び、実行失敗後に別providerへsilent retryしない。cancel/error時はpartial output/stateを公開しない。
- unsupported target/shape、overflow、alias、resource不足はprepareまたはdispatch前にfail closedとする。

## Track A: Full Attention

### A-B0: Phase 33 C2 baseline

- exact gfx1030/gfx1201、`M>=64`、GQA4、head dim 256を1 query row × 1 KV head/workgroupで処理する。
- gridは`M × kv_heads`、K/Vは同rowの4 query headで共有するが、query row間で再利用しない。
- C1 decode wave8 splitと`M<=63`/scope外providerはPhase 35のcomplementとして固定する。

### A-C1: common multi-query-row K/V tiled provider

- 一つのworkgroupが同じKV headの`Q_TILE`連続query rowとGQA 4 headを所有する。K/Vを`K_TILE`単位で一度だけ
  direct decodeしてLDS/registerへ置き、`Q_TILE × 4`の独立online-softmax stateで共有する。
- causal upper boundは各query rowのabsolute positionから求め、nonzero chunk startとtail tileでもfuture keyを参照しない。
- full score matrixを保存せず、tileごとにmaximum、denominator、weighted VをFP32で更新する。
- `Q_TILE=2/4/8`、`K_TILE=32/64/128`、workgroup 256/512をboundedに比較する。最初のprimaryは16 logical attention rowを
  作れてmatrix-alignedな`Q_TILE=4`とし、register/LDS/occupancyで不成立なら2または8へ移る。
- FP16、FP8、static FP8、NVFP4は同じupper tile/routingを使い、codecだけを既存direct read helperへ閉じ込める。
- current C2よりK/V bytes、barrier/key、grid、VGPR/LDS、active blocks/CUがどう変わるかをdevice counterとISAで確認する。

### A-C2: barrier-reduced partial/merge

- A-C1がK/V再利用を得てもper-key workgroup barrierに支配される場合だけ、同じQ/K tile内でwave-owned partialと固定順mergeを比較する。
- merge表現はFP32 `(maximum, denominator, weighted value)`に限定する。global score buffer、atomic output、host combineは使わない。
- workgroup内mergeで収まるvariantを優先する。K方向を複数workgroupへ分ける必要がある場合はPhase 31 arenaへbounded partialを置き、
  scratch byteと追加dispatchを利益と独立に評価する。
- A-C1とA-C2を同時に無制限探索せず、counterで確認した最大残差機構に一候補だけ追加する。

### A-C3: exact gfx1201 matrix inner

- A-C1のstable tileが`Q_TILE>=4`で16 logical rowを持つ場合だけ、同じmask、tile、softmax、scratch、publication上で
  QK/PV innerをgfx1201 WMMA/SWMMAC/MFMA系primitiveへ置換する。
- actual code objectにmatrix instructionがある場合だけmatrix providerと呼ぶ。source primitiveやkernel名だけでは認定しない。
- gfx1030はA-C1/A-C2 common vector pathを維持する。gfx1201 matrix candidateが不採用でもupper tileを複製しない。
- FP16を最初に比較し、low-bit KVはtile decode/convert費用を含めてencoding別に判断する。full-context mirrorが必要なら採用しない。

## Track G: GDN

### G-B0: Phase 28/29 fused row-owner baseline

- gridはvalue head 32 workgroup、workgroup 128 thread。threadがoutput/state columnを一つ所有し、128 state rowをtokenごとに
  global memoryから読書きする。
- Q/K/output normはPhase 29 wave32 reduction、copy/decay/projectionはPhase 28の一pass統合を維持する。
- short/decode、scope外shape、candidate resource不成立時のcomplementとする。

### G-C1: preprocess + column-parallel recurrent core + postprocess

- recurrent stateのphysical layoutをprovider-owned transposed `[value_head, column, row]`へ固定し、logical shapeと
  transaction ABIは変えない。layout移行は単なるindex permutationとして独立oracleへ照合する。
- workgroupをwave32 x 4、gridを`value_heads × ceil(head_dim / 4)`とする。Qwen shapeでは1,024 workgroupとなる。
  各waveは1 state column、各laneは4 state rowを所有し、state shardをregisterへ一度loadしてtoken順に更新し、最後に一度storeする。
- token recurrenceは維持する。`S^T k`、delta、state update、`S^T q`をcolumnごとにwave reductionし、state rowの入力集合を変えない。
- inter-workgroup同期を必要とするQ/K norm、beta/decay生成、output RMSNorm/z gatingは、bounded preprocess/recurrent/postprocessの
  三段へ分ける。preprocessはPhase 29 reductionとBF16 rounding stageを維持し、postprocessはraw projectionのBF16 round後に
  current RMSNorm、norm weight、SiLU(z)を適用する。
- preprocessしたQ/Kはproducer bufferへ安全にaliasできるかlivenessを確認し、不可ならrequest arenaへbounded planeを置く。
  decayのFP32 scratchとraw BF16 outputをaccountし、postprocessは安全ならoutputへin-placeで書く。
- baselineの1 dispatchから最大3 dispatchへ増えるが、32から1,024 workgroupへの並列度、state global traffic削減、
  preprocess重複回避を含むfamily全体device timeで判断する。recurrent kernel単体の速度だけを採否へ使わない。

### G-C2: token span selection

- 最初は選択されたPhase 31 chunk全体を一つのrecurrent dispatchで処理し、projectionはfull-Mのまま維持する。
- recurrenceはtoken間依存を持つため、512 token等へ分けてもtoken並列にはならない。long-running block、register lifetime、
  fairness、state store/loadが実測上の問題になる場合だけ`256/512/1024/current-chunk`を比較する。
- token spanを分ける場合もstate carryはdevice residentのprevious/next transaction内で続け、host round-trip、projection再実行、
  per-span allocationを追加しない。llama.cppの`n_ubatch=512`を無条件にsLLM defaultへ移植しない。

### G-C3: llama.cpp bounded adaptationとprovenance

- fixed sourceのcolumn-owner launch geometry、register state shard、wave reductionを直接adaptする場合は、既存
  `llama-cpp-phase9-gdn-layout-001`が記録する「layoutだけ」のreuse範囲を上書きしない。Phase 35の新しいimport eventとして
  source blob、range、copyright、license、local path、変更内容、import commitを記録する。
- ggml tensor/runtime、generic CUDA dispatch、KDA/snapshot ABIを一括移植せず、sLLMのBF16 input、Phase 28/29 arithmetic、
  state transaction、MTP rewind/replayへ合わせたbounded adaptationとする。
- independent implementationになった場合もreader factsとimplementationを区別し、reuseしなかったことをhistoryへ明記する。

## Adoption scopeとrouting

| scope | target | mode/shape | candidate | complement |
| --- | --- | --- | --- | --- |
| S-A | gfx1030/gfx1201 | GQA4、head dim 256、採用M/KV bucket、encoding別 | A-C1/A-C2 | Phase 33 C2 |
| S-AM | exact gfx1201 | A-C1-compatible tile、採用encoding | A-C3 | A-C1/A-C2 |
| S-G | gfx1030/gfx1201 | Q/K 16、value 32、dim 128、採用token bucket | G-C1/G-C2 | Phase 28/29 G-B0 |
| complement | その他 | 全mode/shape | existing provider | なし |

- stable keyはexact target、semantic op、encoding、query/token count bucket、KV length bucket、head shape、alignment、
  arena/resource availabilityとする。
- threshold `B`を採用する場合は`B-1/B/B+1`とscope内の離れた複数代表値をfinal binaryで確認する。
- prompt内容、token ID、benchmark名、温度/clock、実測後の勝敗、個別request identityをrouting keyにしない。
- common upper sourceを優先するが、両target同じthresholdやmatrix innerを強制しない。target別差は既存registry selectionへ限定する。

## 数値・出力contract

### Full Attention

- real-number semantic、入力key集合、causal order、GQA mapping、FP32 online-softmax state、BF16 RNE outputを維持する。
- K/V共有だけで各query/headのQK reductionとkey更新順がcurrent C2へbit exactならN0とする。
- 同じ項をbalanced treeへ変更しdependency depthと標準boundが非増加ならN1とする。QK/weighted V/partial mergeのいずれかで
  boundが僅かに増える場合はN2とし、Phase 33 C1承認を自動流用しない。
- matrix accumulate、FP16 conversion、partial mergeはcandidate別にN0〜N3を分類する。NaN/Inf、signed zero、future-key poisonを含む。

### GDN

- state transposeは同じlogical elementのpermutationだけならN0である。
- currentのsigned 128-term逐次projectionを4 term/lane + wave treeへ変える場合、同じ項・FP32演算・丸めstageで
  dependency depthが127から概ね8へ下がり標準worst-case boundが非増加ならN1とする。
- Q/K normalization、beta BF16 round、decay、state element update、raw output BF16 round、output RMSNorm/z gatingの
  stageを一つでも省略・移動した場合は自動N1とせず再分類する。
- MTP off/on、accepted/rejected prefix、rewind/replay後のstate、normal decode、long prefillを比較する。

共通して、原因不明、非決定、bound外、causal/state publication違反はN3で棄却する。token差があってもN1なら台帳へ
原因と最初の差を記録して数値gateを自動承認できるが、性能採否は別に判断する。

## Resource・failure contract

- attention tile/partial、GDN preprocessed Q/K、decay、raw output、transposed stateのbyte、alignment、lifetimeをchecked arithmeticで計算する。
- request arena high-waterは10,001/16,385 tokenで記録し、Phase 31のmemory feasibilityを失わせない。scratch不足時はdispatch前に
  complementを選ぶかrequestをfail closedし、途中失敗後のretry fallbackを行わない。
- state layoutはprovider selection時に固定し、同一transaction中にbaseline/candidate layoutを混在させない。
- previous/next recurrent state、MTP snapshot/replay、cancel/error、completion drop、shutdownのowner lifetimeを確認する。
- output/input/state/scratch aliasを明示検証し、postprocess完了前にoutputまたはnext stateをpublishしない。
- VGPR/LDS、active blocks/CU、global traffic、dispatch数、library/kernel symbolをbounded summaryへ残す。

## 測定contract

### Identity・health

- Phase 34 final commit/sourceをbaselineに、source/build/tool/runner/binary、ROCm 7.14.0、LLVM 23、Code Object、
  exact GPU UUID/BDF、model/derived lock/GGUF、KV/chunk/providerを固定する。
- local Qwen serviceを停止してV620 pairを解放し、一度に一GPUだけをUUIDで可視化する。foreign process、VRAM/GTT、ECC、
  temperature、clock、power、loader root、cleanupをpre/during/postで確認する。
- timeout、crash、zero sample、CPU/backend fallback、partial offload、wrong target、GTT spillをGPU PASSにしない。

### Operator・profile

- baseline/candidateは同一binary、同一buffer/state、同一streamでcounterbalanceする。first-callとsteady stateを分ける。
- 初期値はwarmup 5、measured 21のHIP event device timeとするが、confidenceが得られた後の機械的追加取得をhard gateにしない。
  p50、MAD、p10/p90、min/max、absolute ns、relative差、dispatch/kernel、resourceを記録する。
- rocprofはfamily share、actual kernel、grid/workgroup、barrier、traffic、occupancy、ISAの診断に使い、profiler wallを採否へ使わない。
- attentionはB0/A-C1/A-C2/A-C3、GDNはB0/preprocess/recurrent/postprocess/family合計を分ける。

### Full model・peer

- final候補は`B0`、`Attention-only`、`GDN-only`、`combined`を同じsemantic identityで比較し、各trackの寄与と相互作用を分離する。
- profilerなしfull-modelはfresh processで順序をcounterbalanceし、TTFT、prefill、first token、TPOT、E2E、peak VRAMを記録する。
- sLLM direct-token/private evidence laneを用意できる場合はmessages/tokenizationを除いた境界を追加し、用意できない場合は
  kernel外残差をdevice candidateの成果へ数えない。
- fixed llama.cppは同じ10,001 token IDs / 2 output、model revision、BF16 logical weight、FP16 KV、commit、batch/ubatch、
  offload、MTP条件を記録する。artifact layout差が残るためE1とし、device familyとE2Eを併記する。

## Verification matrix

### H0: host/build/routing/provenance

- exact gfx1030/gfx1201 compile/link、gfx942 compile-only、wrong-target load拒否。
- A-C1/A-C2/A-C3、G-C1/G-C2、既存complementのstable routing、provider ID、logical/device symbol、actual launch一致。
- M/KV/token thresholdの`B-1/B/B+1`、head/ratio/dim `±1`、unknown target/shape、overflow、alias、resource不足。
- GDN transposed state index/size、previous-next transaction、MTP snapshot/replay、failure cleanupのhost test。
- direct llama.cpp adaptationがある場合は新規provenance entry、notice、source hash、import commitをrelease/push前に整合させる。

### G1: Full Attention operator correctness・performance

| class | M / start / KV | 目的 |
| --- | --- | --- |
| complement | M=1/2/3/31/32/33/63 | existing C1/C2 boundary |
| tile boundary | M=63/64/65、127/128/129、255/256/257 | Q tile tail、current threshold |
| chunk boundary | M=511/512/513、2047/2048/2049 | peer ubatch control、row/K tile scaling |
| long | M=4095/4096/4097、8191/8192/8193、10001 | K/V reuse、device scaling |
| nonzero start | selected M、start=1/3/511/16384 | absolute causal mask、chunk tail |
| special | NaN query、+Inf value、subnormal、future-key poison | classification、causality |

- 全直積は作らず、FP16/dynamic FP8をperformance primary、static FP8/NVFP4をcorrectness/resource controlとしてpairwiseに選ぶ。
- scalar FP64 oracle、causal/GQA、metadata、repeat、fallback、cleanupを確認する。
- A-C3は同じA-C1 tile/inputと比較し、matrix math以外を同時に変えない。

### G2: GDN operator/state correctness・performance

| class | token count | 目的 |
| --- | ---: | --- |
| decode/MTP | 1/2/3/4/7/8 | baseline complement、verify/replay |
| launch/tile | 15/16/17、31/32/33、63/64/65 | routing、column group、tail |
| prefill | 127/128/129、255/256/257、511/512/513 | state carry、peer ubatch control |
| long | 2047/2048/2049、4096、10001 | state traffic、GPU utilization |
| multi-span | 16384 then 1、selected 512-style span | state carry、chunk tail |

- model-free NumPy/F64 oracleはQ/K norm、beta、decay、state element、raw projection、output norm/gateを段階別に照合する。
- zero/nonzero previous state、signed/exponent-mixed state、NaN/Inf policy、state transpose、previous/next non-aliasを含む。
- Phase 29 baselineとcandidateのtoken列、state digest、first divergence、numerical classを記録する。
- GDN familyはpreprocess + recurrent + postprocessの合計device時間、dispatch数、arena byte、state trafficで採否する。

### G3: full-model attribution

| case | target | input / output | KV | 主目的 |
| --- | --- | ---: | --- | --- |
| F0 | gfx1030/gfx1201 | 29 / 32 | FP16 | short/decode complement |
| F1 | gfx1030/gfx1201 | 4108 / 32 | FP16/dynamic FP8 | Phase 30/33 continuity、TPOT |
| F2 | gfx1030/gfx1201 | 10001 / 2 | FP16 | primary TTFT、4-way attribution |
| F3 | selected target | 10001 / 16 | FP16/dynamic FP8 | long-prefix decode/state continuity |
| F4 | safe selected target | 16385 / 2 | FP16 | 16384+1 chunk、arena/state carry |

- 最小採用判断は両targetのF2と、各trackのrouting complementを示すF0またはF1とする。F3/F4はcandidate scopeと安全な
  preflightに応じてfreezeし、全rowの機械的実行をhard gateにしない。
- prompt/completion token、usage、KV/GDN committed state、all-HIP audit、fallback、VRAM/GTT、cleanupを記録する。
- N2 candidateはcombined defaultへ接続する前にユーザー判断へ戻す。

### G4: integration/lifecycle

- 採用candidateを通常CLIとOpenAI non-stream/SSEで代表10k+ promptから実行し、usage、first token、`[DONE]`、
  disconnect/cancel、直後recovery、graceful shutdownを確認する。
- shutdown後のcurrent/request/workspace bytes、retryable cleanup、durable quarantineを0とする。
- low-bit attentionを採用する場合はdynamic FP8 lifecycleを一つ追加する。GDNだけの変更でKV/API matrixを機械的に増やさない。

## 受入・採否基準

### Hard correctness・resource

1. Attention causal/GQA/softmax/outputとGDN recurrence/norm/gate/state transaction/public ABI/APIを維持する。
2. candidate固有oracleと既存toleranceを満たし、N0〜N3分類と根拠を持つ。toleranceを結果に合わせて拡張しない。
3. N3、wrong-target/shape誤route、metadata/actual launch不一致、CPU/runtime fallback、timeout/crash、zero testをPASSにしない。
4. unaccounted scratch、full score/state/KV mirror、GTT spill、OOM誘発、partial publication、cleanup failureを性能で相殺しない。
5. scope外入力はfinal sourceでPhase 33またはPhase 28/29 baselineへ事前routeされる。

### 担当AIによるperformance adoption

各scopeで次を総合し、`shared/scoped adoption`、`reject/negative completion`を理由付きで決める。

- operator familyとfull-modelの絶対短縮秒、割合、Amdahl転化。
- process/順序間の分散、baseline drift、改善の一貫性、利用頻度、long-context重要度。
- N0/N1/N2、token/state差、oracle margin、承認費用。
- LDS/VGPR/scratch/arena、dispatch/traffic、first-call、short/decode overhead。
- common sourceとtarget-specific inner/thresholdの保守費用、既存architectureとの整合、将来再利用性、rollback容易性。
- fixed llama.cppとの差がどこまで縮んだか。ただしpeer parity自体を単一hard thresholdにしない。

AttentionとGDNのcombined candidateが単独利益の和より小さい場合は、resource/occupancy/measurement境界を説明し、
単独scopeの採否を再評価する。projectionが悪化した場合はPhase 35利益で相殺せず、routingまたは実装上の干渉として修正する。

### Evidence・closeout

6. raw trace/DB、model、binary、full logits/state/outputを追跡しない。bounded summary/schema/test、aggregate、digest、plan/historyを残す。
7. track別の改善/悪化、confidence、採否理由、scope、complement、数値分類、resource、rollback、再検討条件を記録する。
8. 採用時はruntime、GPU/AMD/software compatibility、数値台帳、provenance、main planを影響範囲に応じて同期する。
9. 不採用candidateのforce flag、temporary timing seam、unused kernel/symbolをproduction sourceから除去する。
10. affected host/build/GPU/full-model checkと一回のintegration reviewを行い、findingがあれば変更箇所だけfocused re-reviewする。

## 作業順序

### P35-A0: acceptance・identity・fresh baseline

- Phase 34 final commit/buildをbaselineに固定し、current 10,001/2を両targetでfresh profileする。
- sLLM family boundaryをprojection、Full Attention、GDN recurrent/conv/pre/post、otherへ固定し、llama.cppの同一token profileを
  current identityへ再対応づける。direct-token sLLM timingの可否を確認する。
- Full Attention 8 layer、GDN 24 layerのshape、dispatch、grid/workgroup、barrier、traffic、VGPR/LDSを記録する。
- acceptance matrix、candidate toggle、summary/schema draft、数値分類項目を実装前にfreezeする。

### P35-A1: comparison seams・oracles・state layout contract

- production ABIへ露出しないB0/Attention/GDN/combined selection seamを用意する。
- Attention tile/partial oracleとGDN stage/state/layout oracle、routing/resource/failure host testを追加する。
- GDN transposed physical layoutとprevious/next/MTP snapshotのcontractを実装前に固定する。
- fixed llama.cpp GDN topologyをdirect adaptするかfacts-onlyにするか決め、reuseする場合は新規provenance eventをdraftする。

### P35-A2: Full Attention common tile

- A-C1のQ/K tile候補をboundedにscreenし、K/V reuse、barrier、resource、device timeから一つへ絞る。
- G1のcorrectness/metadataを通し、target/encoding/M/KV scopeとC2 complementをfreezeする。
- per-key barrierが最大残差の場合だけA-C2を一候補追加し、同じmechanismの無制限variant探索を行わない。
- final common vector candidateの数値分類とoperator採否を確定する。

### P35-A3: gfx1201 matrix inner

- stable common tileが16 logical row以上を持つ場合だけA-C3を実装する。
- actual ISA、FP16 oracle、common vector比device time、LDS/VGPR/occupancyを確認する。
- low-bitはdecode/convert費用込みで独立採否する。別tile/ABI、KV mirror、N3が必要ならC3を棄却する。
- N2ならproduction接続前にユーザー判断へ戻し、common vector trackは継続する。

### P35-G1: GDN column-parallel pipeline

- preprocess、column-parallel recurrent、postprocessとtransposed state layoutを実装し、Phase 28/29 arithmetic/rounding stageを保持する。
- G2をshortからlongへ段階実行し、32→1,024 workgroup、register state、global traffic、3-dispatch family合計を比較する。
- token spanはcurrent chunkを先に採用し、実測残差がある場合だけG-C2を比較する。
- numerical class、token/state、resourceからS-G scopeとB0 complementをfreezeし、担当AIがoperator採否を確定する。

### P35-A4: production routing・単独full-model

- 採用候補だけをstable registry keyへ接続し、temporary overrideを除去する。
- Attention-onlyとGDN-onlyでF0/F1/F2の必要最小rowを実行し、単独のTTFT/TPOT/E2E転化と干渉なしを確認する。
- gfx1030/gfx1201 common upper path、target-specific inner/threshold、scope外baselineをfinal dispatch evidenceへ固定する。

### P35-A5: combined・peer comparison・integration

- B0/Attention-only/GDN-only/combinedをcounterbalanceし、F2 primaryと必要なF3/F4を実行する。
- final combined profileをfixed llama.cppの同token profileと同じfamily表で比較し、projection維持、attention/GDN gap、
  kernel外boundaryを分ける。
- contextual adoptionをtrack/scopeごとに確定し、通常CLI/API lifecycle、wrong-target、cleanupを実行する。

### P35-A6: closeout

- `phase35-attention-gdn-summary-v1.json`、schema/test、history、数値台帳、runtime/compatibility/provenance/main planを同期する。
- 一回のintegration reviewとfindingのfocused re-reviewを行う。
- plan/historyを相互linkしてarchiveし、raw artifactと不採用sourceを除外して必要最小限のcommitへ整理する。

## 停止・再計画条件

- Attention tileがcausal/GQA semanticを維持できず、full score matrix、host combine、KV layout変更、Paged Attentionを必要とする。
- GDN column providerがstate transaction/MTP rewindを維持できず、public ABI変更またはsequence-parallel scanが必要になる。
- candidateがPhase 31 memory feasibilityを失わせ、bounded arena reuseまたはbaseline complementで回避できない。
- N3、非決定、bound外、wrong-target route、silent fallback、GTT spill、cleanup defectが解消できない。
- 同じmechanismが2回rejectされ、counter上の新しい根拠なしにtile/workgroup/span variantだけを増やす状態になる。
- review時間が実装時間超、機能進捗が1時間停止、verification/docsが30%超、見積り1.5倍超、acceptance変更時は
  新規variant/reviewを止め、残るtrackを別Phaseへreplanする。

[Phase 28計画](../../../../archive/2026/08/11-20/phase28-decode-nonprojection-device-optimization.md)
[Phase 29計画](../../../../archive/2026/08/11-20/phase29-gdn-useful-workgroup-parallelization.md)
[Phase 31計画](../../../../archive/2026/08/11-20/phase31-chunked-prefill-memory-foundation.md)
[Phase 33計画](../../../../archive/2026/08/11-20/phase33-full-attention-structural-optimization.md)
[Phase 34計画](../../../../archive/2026/08/11-20/phase34-v620-long-prefill-bf16-matmul-provider-optimization.md)
[runtime architecture](../../../../../architecture/runtime.md)
[数値・出力影響変更台帳](../../../../../compatibility/numerical-output-changes.md)
[provenance](../../../../../provenance/README.md)
[メイン計画](../../../../main-plan.md)

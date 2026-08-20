# Phase 33: Full Attention構造最適化

> 状態: 完了（C1/C2限定採用、C3棄却）
> 作成日: 2026-08-19

## 実施状況（2026-08-19）

- A0のfresh traceで、Phase 32報告がnative append encode中の`M=1` attentionを集計から漏らしていたことを確認した。
  10,001 inputのB0はprefill 18.648秒に加え`M=1` 35.275秒、16,385 inputではprefill 50.532秒に加え
  `M=1` 94.854秒であり、Full Attention全体が主要支配時間だった。
- C1はglobal scratchとsecond combine launchを必要とするmulti-workgroup案から、1 workgroup内を8 waveの連続KV区間へ分割し、
  partial online-softmaxを区間順にmergeするscratch-free案へbounded replanした。`M=1`、KV長1,024以上、head dim 256、
  exact gfx1030/gfx1201だけへrouteする。KV=1,024〜8,193のFP16 device中央値はR9700で約53〜58%、V620で約64〜65%短縮した。
  QK和の依存深さが概ね8段から12段へ増えるN2であることを維持し、2026-08-20のユーザー承認により限定採用した。
- C2は`Q_TILE=1`のまま同じKV headを使うGQA 4 headでK/V decodeを共有するscratch-free providerへ絞った。
  `M>=64`、GQA ratio 4、head dim 256、exact gfx1030/gfx1201へrouteし、M=64〜257のFP16 device中央値を
  R9700で約21〜47%、V620で約38〜54%短縮した。gfx1201は既存wave treeとbit exactのN0、gfx1030は同じ8段boundの
  固定tree変更としてN1であり、担当AI判断では共通限定採用が妥当である。
- C1/C2の最終候補はFP16/dynamic FP8/static FP8/NVFP4 × 2 target × 29 case、計232 caseを独立scalar FP64 oracle、
  causal/GQA、metadata、fallback、cleanup込みで全PASSした。FP16最大絶対誤差は`2.3841858e-7`、FP8は`4.7683716e-7`、
  NVFP4は`1.1641532e-9`だった。KV=1,024のC1特殊値とM=64のC2特殊値も含む。
- R9700 10,000 promptの同条件CLIはB0 105.800秒からC1+C2 75.162秒へ28.96%短縮した。profiler aggregateでは
  Full Attention 53.922秒から23.101秒へ57.16%短縮し、最終C1はLDS 8,256 byte/VGPR 104/scratch 0、
  C2はLDS 192 byte/VGPR 48/scratch 0だった。V620 4,108 promptはFP16 39.097→36.500秒、FP8
  39.304→36.669秒で約6.6〜6.7%短縮した。全caseでtoken、usage、HIP-only、fallback false、cleanup 0を維持した。
- C3はC2の実採用tileが1 query row × 4 GQA headで4 rowしかなく、gfx1201 FP16 WMMA primitiveの16×16×16最小shapeに
  合わない。12 row paddingまたはQ_TILE=4の別layoutが必要となり「同じC2 tileのinner mathだけ」というscopeを越えるため棄却した。
  最終gfx1201 C2 code objectのmatrix instructionは0であり、matrix providerとは呼ばない。llama.cpp等からのcode reuseは行っていない。
- dynamic FP8のgfx1201 serverで10,012 prompt tokenのnon-stream/SSE、`[DONE]`、client disconnect、直後のrecovery、
  graceful shutdownをPASSした。shutdownはcurrent/request/workspace byte、retryable cleanup、durable quarantineがすべて0だった。
- 承認後の最終identityで両target・4 encoding・29 caseの232/232 oracle、R9700 10,000-prompt、V620 4,108-prompt、
  R9700 dynamic FP8 API lifecycle、wrong-target拒否を再実行し、HIP-only、fallbackなし、cleanup 0を確認した。

## 目的

Phase 31で10k+ inputとchunked prefillが通常経路で成立し、Phase 30で改善したwave reductionだけでは
long-context Full Attentionの支配時間を解消できないことが明確になった。Phase 33では、現行causal/full attentionの
semantic op、vAttention `virtual-contiguous` KV、KV encoding、transaction/publication contractを維持したまま、次を独立に比較する。

1. decode/small-queryでKV列を複数workgroupへ分割し、GPU全体を利用するsplit-KV provider。
2. prefillで複数Q行とGQA headがK/V tileを共有するtiled online-softmax provider。
3. 共通prefill tile上でexact `gfx1201`だけQK/PV内部をmatrix instructionへlowerするprovider。

各candidateは独立したadoption scopeとして採否を決める。単一の巨大なFA4互換実装、特定target専用の上位graph、
full score matrixは作らない。共通のdispatch、scratch、online-softmax merge、correctness contractを優先し、target差は
inner mathと既存native codecへ閉じ込める。

## 開始根拠と変更された前提

- Phase 23の選択traceではfull-attention/KV shareは7%未満で、短contextの最優先候補ではなかった。
- Phase 30はexact `gfx1201`のnative FP8 readとwave32 reductionを`M=1`/`M>=32`へ採用し、4108 inputでTTFT 9.60%、
  prefill 9.72%、decode throughput 7.86%を改善した。一方、prefill matrix providerは別tile/layoutを要するため延期した。
- Phase 31はworkspace arenaを約86.79%縮小し、10,001-token one-chunkと16,385-token `16,384+1`を通常経路で成立させた。
- Phase 32のlocal diagnostic traceでは、10,001 inputのcausal attentionは17.652秒、GPU kernel時間の72.65%、
  full-model timingの59.46%相当だった。内訳は8 prefill dispatchが17.584秒、8 decode dispatchが68.426 msだった。
- 16,385 inputでは24 attention dispatchが48.441秒、GPU kernel時間の82.11%、full-model timingの75.23%相当だった。
  8 main-prefill dispatchが48.217秒、second one-token chunkとdecodeの計16 dispatchが224.514 msだった。
- 上記raw traceはrepositoryへ追跡しないdiagnosticであり、Phase 33の採用証拠へ直接流用しない。A0でcurrent identityから
  fresh aggregateを再取得し、bounded summaryへdevice share、wall-equivalent share、dispatch shapeだけを残す。
- 現行kernelは`query_count × q_heads` blockを起動し、各blockがkey positionを0からcausal endまで逐次走査する。
  decode `M=1`は16 blockしかなく、64/72 CUをkey方向に利用できない。prefillはquery/headごとにK/Vを再読込し、
  GQA ratio 4のhead間共有、Q-row tile、matrix instructionを使用しない。

## Phase固有の範囲

### 対象

- primary model: fixed Qwen3.5-4B BF16 GGUF/derived lock、8 full-attention layer、Q heads 16、KV heads 4、head dim 256。
- target: Radeon Pro V620 exact `gfx1030` UUID `GPU-76a08c022586fed6`、Radeon AI PRO R9700 exact `gfx1201`
  UUID `GPU-a8e9ddefa2d60f55`。
- KV encoding: FP16を共通algorithmのprimary、dynamic FP8をlow-bit primary、static FP8/NVFP4をdirect-read correctness controlとする。
- mode: `M=1` decode、small query/prefill、one-chunk long prefill、`16,384+1` chunked prefill。
- current Phase 30 native FP8 read/wave provider、Phase 31 chunk selector/liveness arena、Phase 32 native FP8 appendをbaselineへ含める。
- model-freeではhead dim `255/256/257`、GQA ratio `1/2/4`、非整列KV長、chunk開始位置を扱う。
- fixed llama.cpp sourceの`fattn-vec`、`fattn-tile`、`fattn-mma-f16`、parallel-block combineをprovider topology、
  occupancy、scratch、GQA reuseの比較対象にする。直接reuseが合理的なら通常のprovenance手順を適用する。

### 非対象

- sliding-window/Gemma専用attention、DFlash、MLA、Paged Attention、RadixAttention、prefix sharing、continuous batching。
- vAttentionの置換、KV layout/scale recipe変更、low-bit KV default昇格、TurboQuant、新KV encoding。
- Q/K/V projection、RoPE、KV appendとのfusion、full-attention layer間fusion、HIP Graph化。
- attention score `M × N`全体を保存する実装、requestごとの`hipMalloc`、host softmax/combine、CPU fallback。
- gfx1200、gfx942、別RDNA/CDNA SKU、別model/head shapeへの性能一般化。
- FlashAttention 4との名称上の互換性または同等性能claim。Phase 33はsLLMのbounded provider改善である。

## Baselineとcandidate

### B0: current online-softmax provider

- 256-thread、query/head当たり1 block、key positionを逐次走査する。
- exact gfx1201の`M=1`/`M>=32`はPhase 30 wave providerとnative FP8 readを使用する。
- gfx1030とgfx1201 `M=2..31`は既存LDS reductionを使用する。
- semantic equation、BF16 output RNE、KV transaction、public ABI、dispatch evidenceをbaselineとして固定する。

### C1: common decode split-KV + deterministic combine

- primaryは`M=1`、secondaryは小さい`M`とし、KV key rangeを`P`個のtileへ分割する。
- 各partialはFP32の`local_max`、`local_sum`、`local_weighted_value[head_dim]`を出力する。combineは
  `global_max = max(local_max)`、`global_sum = Σ local_sum * exp(local_max-global_max)`、
  `output = Σ local_weighted_value * exp(local_max-global_max) / global_sum`を固定順で実行する。
- outputへのatomic加算は使用せず、partial publication後の一つのcombine kernelで決定的に出力する。
- scratchはPhase 31 request arenaのcompletion boundaryへ追加し、`P × M × q_heads × (head_dim+2) × sizeof(float)`を
  checked arithmeticでaccountする。per-dispatch allocation、persistent full-context mirrorは作らない。
- `P=1/2/4/8/16`とoccupancy/CU/KV tile数から求めるauto selectionを比較する。CU数だけで固定せず、active block数、
  wave tail、combine overhead、KV長を含むstable pre-dispatch keyで選ぶ。
- gfx1201はpartial内で既存wave/native codecを、gfx1030は同じsplit/merge構造と既存vector codecを使用する。

### C2: common prefill tiled online softmax + GQA reuse

- 一つのworkgroupがQ row tileと同じKV headを使うGQA query head群を所有し、causal endまでK/V position tileを順に処理する。
  K方向も複数workgroupへ分けるvariantではC1のpartial/combine contractをそのまま使う。
- K/V tileをglobal memoryから一度読み、同じtile内の複数Q行・GQA headで共有する。FP8/static FP8/NVFP4は
  scaleをtile内で共有し、register/LDSへdirect decodeする。request全体のFP16/BF16 KV mirrorは作らない。
- causal upper boundは各Q行のabsolute positionから求め、chunk開始位置が0でない場合もfuture keyを参照しない。
- `Q_TILE=1/2/4/8/16`、`K_TILE=32/64/128`をbounded prototypeで比較する。採用値はtarget/encoding/head contractごとに
  final manifestへ固定し、prompt内容や実測結果をruntime keyにしない。
- score全体を保存せず、K tileごとのonline maximum/denominator/weighted-valueをFP32で更新する。
- C1のpartial/combine表現を再利用できる場合は共有し、別のsoftmax ABIやscratch familyを増やさない。

### C3: exact gfx1201 matrix inner provider

- C2と同じQ/K/V tile、mask、online-softmax、scratch、dispatch contractを使い、QK/PV inner mathだけを
  gfx1201で利用可能なWMMA/SWMMAC/MFMA系compiler primitiveへ置換する。
- 最初はFP16 KVでmechanismと数値差を分離する。FP8/NVFP4はdirect decodeからmatrix inputを作る費用を別計測し、
  full-context mirrorまたはKV format変更が必要ならC3 scopeへ採用しない。
- actual code objectにmatrix instructionがある場合だけmatrix providerと呼ぶ。source-level primitive、rocWMMA include、
  kernel名だけをhardware pathの証拠にしない。
- gfx1030へmatrix instruction、target symbol、選択overheadを混入させない。上位routingとC2 fallbackは共通に保つ。
- Q/K入力変換やaccumulator precisionがcurrent real-number implementationへ与える影響を独立に分類する。

C1/C2/C3は独立採否とする。C1採用・C2棄却、C2採用・C3棄却、encodingまたはtarget別の限定採用を許す。

## Adoption scopeとrouting key

| scope | target | mode | encoding | candidate |
| --- | --- | --- | --- | --- |
| S-D | gfx1030/gfx1201 | decodeまたはsmall `M`、採用KV長bucket | FP16/FP8/static FP8/NVFP4の合格encoding | C1 |
| S-P | gfx1030/gfx1201 | prefill、採用M/KV長bucket | 合格encoding | C2 |
| S-M | exact gfx1201 | prefill、C2-compatible tile | 合格encoding | C3 |
| complement | 上記scope外 | 全mode | 全encoding | B0または下位合格provider |

- stable keyはexact target、encoding、query count bucket、KV length bucket、head dim、GQA ratio、alignment、scratch availabilityとする。
- `S-M`のcomplementはC2、`S-P`/`S-D`のcomplementはB0へ戻す。runtime失敗後のsilent fallbackではなく、
  dispatch前のcapability/resource selectionとしてproviderを決める。
- threshold `B`を使う場合は`B-1/B/B+1`とscope内の複数値をfinal performance前に固定する。
- DPM状態、温度、個別prompt、token列、benchmark名、実測後の勝敗をrouting keyにしない。

## 数値・出力規則

- real-number semanticはcausal scaled-dot-product attention、online softmax、GQA mappingのまま維持する。
- partial softmax merge、QK/PV tile、matrix accumulationによる演算順変更を数値台帳N0〜N3へcandidate別に記録する。
- baselineとbit exactならN0とする。固定tree化について解析的な誤差boundが非増加と示せる部分だけN1とする。
- signed Vのweighted sum、input dtype変換、matrix accumulateが非増加boundを保証できない場合はN2とし、既存NumPy FP64 oracle、
  output error、logit/top-1/token影響を定量化してproduction採用前にユーザー判断へ戻す。専用production高精度pathは要求しない。
- 原因不明、非決定、非有界、causal mask違反、NaN propagation contract破壊はN3として棄却する。
- 既存N1 token差承認を新しいC1/C2/C3へ自動流用しない。candidateごとの原因と演算treeを台帳へ記録する。

## Resource・lifetime contract

- vAttention `virtual-contiguous` KV pointerとtoken-major value/scale planeをそのまま使用する。Paged Attentionとの競合は発生させない。
- scratchはrequest arena内のchecked intervalとして確保し、attention completionまで保持する。chunk間・layer間でlifetimeが重ならない
  scratchは再利用し、model-resident allocationへしない。
- preflightはcandidate scratch、alignment、high-waterを含める。scratch不足時はdispatch前にB0/C2 complementを選ぶか、
  request自体をfail-closedする。途中allocation failureから別providerへsilent fallbackしない。
- partial buffer、output、query、KV value/scale planeのaliasを拒否する。combine完了前にKV length/outputをpublishしない。
- cancel、timeout、completion drop、kernel failureではpartialを公開せず、既存KV committed lengthを維持する。

## 測定contract

- current `main` commit、kernel/tool source SHA-256、ROCm 7.14.0、LLVM、exact target、code object、release flags、
  fixed model/lock、selected KV/chunk/providerをidentityへ固定する。
- local Qwen serviceを停止してV620 pairを解放し、一度に一GPUだけ測定する。GPUはUUIDで単独可視化する。
- pre/during/postのforeign process、VRAM/GTT、ECC、temperature、clock、power、loader root、cleanupを記録する。
- operatorは同一process内でB0/C1/C2/C3をcounterbalanceし、warmup 5以上、measured 21以上、HIP event device timeを使う。
  p50、MAD、p10/p90、absolute ns、bytes read、scratch、dispatch数、workgroup/grid、VGPR/LDS、active blocks/CUを記録する。
- profilerはfamily share、memory/occupancy、actual kernel/ISAのdiagnosticに限定し、profiler wallを性能採否へ使わない。
- full modelはprofilerなしの3 independent processでbaseline/candidateをcounterbalanceする。10,001 inputは各process 1 warmup + 5 measured、
  16,385 inputは1 warmup + 3 measuredを初期値とし、noiseまたはbaseline driftが判断を妨げる場合だけ追加取得する。
- decodeはlong prefixから16または32 outputを生成し、prefill/TTFT、first committed token、TPOT、decode token/s、E2Eを分離する。
- fixed llama.cppとの数値速度比較は同一operator input/boundaryを作れる場合だけE1とし、token stream、KV dtype/layout、timing boundaryが
  異なるfull modelはE2 diagnosticとしてratioを採否根拠にしない。

## Verification matrix

### H0: host/build/ISA・routing

- exact gfx1030/gfx1201 compile/link、wrong-target load拒否、candidate symbol/metadata/actual launch一致。
- C1 split数、C2 tile、C3 matrix provider、B0 complementのstable keyと境界unit test。
- checked scratch arithmetic、alignment、arena lifetime、alias、capacity、zero/overflow、resource不足のfail-closed test。
- gfx1201 C3採用候補はactual matrix ISA、gfx1030は同ISA/target symbol 0。native FP8 read既存ISAも維持する。

### G1: C1 decode split-KV

| case | M | KV length | 目的 |
| --- | ---: | --- | --- |
| D0 | 1 | 1/2/3/31/32/33 | combine不要／短KV overhead |
| D1 | 1 | 255/256/257、1023/1024/1025 | tile・split境界 |
| D2 | 1 | 4095/4096/4097、8191/8192/8193 | CU occupancy・long decode |
| D3 | 1 | 9999/10000/10001、16383/16384/16385 | 実long-context |
| D4 | 2/3/17/31/32/33 | 1024/10001 | small-M routing境界 |

- canonical Qwen shapeは各target・合格encodingで実行し、head dim `255/256/257`とGQA ratio `1/2/4`はpairwise boundary matrixで
  B0/NumPy oracleと比較する。全次元の機械的な直積は作らない。
- `P=1/2/4/8/16/auto`のoutput、partial metadata、combine、scratch、device timeを記録する。

### G2: C2/C3 prefill tile

| case | M/chunk | start / expected KV | 目的 |
| --- | --- | --- | --- |
| P0 | 1/2/3/17/31/32/33 | 0 / M | decode・small-prefill complement |
| P1 | 255/256/257 | 0 / M | Phase 30・tile境界 |
| P2 | 511/512/513、2047/2048/2049 | 0 / M | chunk bucket・GQA reuse |
| P3 | 4095/4096/4097、8191/8192/8193 | 0 / M | long prefill scaling |
| P4 | 10001 | 0 / 10001 | current one-chunk通常経路 |
| P5 | 16384 then 1 | 0 then 16384 / 16385 | multi-chunk absolute causal mask |

- FP16/dynamic FP8をperformance primary、static FP8/NVFP4をcorrectness/resource controlとする。
- causal future-key poison、nonzero start、non-aligned head dim、GQA ratio境界を独立oracleで確認する。
- C3はC2と同じtile/inputで比較し、matrix math以外の差を同時に入れない。

### G3: full model performance・correctness

| case | target | input | output | KV | 主指標 |
| --- | --- | ---: | ---: | --- | --- |
| F0 | gfx1030/gfx1201 | 29/267 | 32 | FP16 | short complement、TPOT |
| F1 | gfx1030/gfx1201 | 4108 | 32 | FP16/FP8 | Phase 30連続性、TTFT/TPOT |
| F2 | gfx1030/gfx1201 | 10001 | 16 | FP16/FP8 | one-chunk TTFT、decode split |
| F3 | gfx1201 | 16385 | 16 | FP16/FP8 | `16384+1` TTFT、long decode |
| F4 | gfx1030 | 実行可能な最大fresh長 | 16 | FP16/FP8 | control resource境界 |

- baseline/candidateのprompt token、generated token、usage、state length、all-HIP audit、fallback、cleanupを記録する。
- token差がある場合は数値分類と最初の分岐layer/opを説明し、N2ならproduction採用前にユーザー判断へ戻す。

### G4: integration/lifecycle

- 採用候補を通常CLIとOpenAI non-stream/SSEで10k+ promptから実行し、usage、first token、`[DONE]`、cancel/disconnect、
  recovery、shutdown request/workspace/current byte 0を確認する。
- provider evidenceはselected ID、actual symbol、split/tile、target、encoding、fallback falseを報告する。
- server同時要求、Paged Attention、prefix cacheはPhase 33で追加しない。

## 受入・採否基準

### Hard correctness・resource条件

1. causal mask、GQA mapping、query/start/KV length、output shape、KV transaction/publication、public ABI/APIを維持する。
2. candidateは独立NumPy FP64 oracleとcandidate固有toleranceを満たし、数値差をN0〜N3へ分類する。
3. N3、unsupported targetへの誤route、CPU/runtime fallback、zero test selection、timeout/crash、resource/cleanup破壊をPASSにしない。
4. scratchはaccounted arena内に収まり、full score matrix、per-dispatch allocation、full KV mirror、GTT spillを導入しない。
5. provider metadata、actual dispatch、ISA、target、split/tile、fallbackが一致する。

### Performance adoption

6. 固定改善率thresholdまたは全pattern一律非悪化gateは置かない。担当AIがS-D/S-P/S-Mごとにoperator/full-modelの
   絶対短縮量と割合、confidence、一貫性、利用頻度、長context寄与、scratch/launch overhead、target分岐、実装・検証・保守費用、
   architecture再利用性、revert容易性を総合して理由付きで採否を決める。
7. 短KV/Mでstableに不利でも、事前に説明可能なthresholdでB0 complementへrouteでき、scope内利益が管理費用を上回る場合は
   scoped採用できる。悪化値、境界、selection overheadを隠さない。
8. C1/C2の共通構造を優先するが、両target同一providerをhard gateにしない。target別inner mathまたはthresholdが合理的なら限定採用する。
9. C3はmatrix ISAの存在だけで採用しない。C2比の実利益、数値分類、LDS/VGPR/occupancy、encoding範囲、保守費用を判断材料にする。
10. operator改善がfull-modelへ転化しない場合も自動棄却せず、絶対時間、将来batch/long-context再利用、複雑性との釣り合いを記録する。

### Evidence・closeout

11. raw trace、DB、model、binary、full logits/KV、生成全文を追跡しない。summary/schema/test、aggregate、digest、plan/historyを残す。
12. candidate別の採否理由、改善/悪化、測定限界、scope、complement、再検討条件をbounded summaryへ記録する。
13. direct llama.cpp reuseを行う場合はcopyright/license、完全commit、upstream/local path、hash、変更点、notice、import commitを
    release前にprovenance正本へ記録する。facts-onlyならその境界を記録する。
14. 採用時はruntime、AMD/software compatibility、数値台帳、main planを同期する。不採用candidateのforce switch、debug timing、
    unused symbolはproduction sourceから除去する。

## 作業順序

### P33-A0: acceptance・identity・fresh long-context baseline

- current source/build/model/GPU/toolchain identityとB0 routingを固定する。
- 10,001/16,385 traceをfresh取得し、attention family、prefill/decode/small-chunk、dispatch shape、Amdahl shareをbounded aggregate化する。
- B0のM/KV/encoding別device time、block/CU、memory traffic、barrier、occupancy、VGPR/LDSを取得する。
- fixed llama.cppのvec/tile/matrix/split provider構造をcurrent lockへ再固定し、reuseかfacts-onlyかをcandidate実装前に記録する。

### P33-A1: partial-softmax merge・scratch contract

- C1/C2で共有可能なpartial metadata、fixed merge equation、scratch layout、checked sizing、arena lifetimeを定義する。
- tiny CPU/NumPy oracle、host ABI/routing/resource testを先に作り、split 1がB0 semanticへ一致することを確認する。
- combine単独のGPU correctnessとoverheadをD0/D1で測り、設計がlong-contextにだけ有効な場合はthreshold候補を固定する。

### P33-A2: C1 decode split-KV prototype・採否

- `P=1/2/4/8/16/auto`を同一binaryで比較し、D0〜D4を両targetで実行する。
- auto selectorはoccupancy、CU、KV tile数、wave tail、combine overheadから決め、exact dispatch metadataへ記録する。
- S-Dのencoding/target/KV bucketを独立判断し、不採用bucketはB0 complementへ固定する。

### P33-A3: C2 common prefill tile prototype・採否

- Q/K tile、GQA reuse、causal/nonzero-start mask、direct low-bit decodeを順に導入し、P0〜P5を実行する。
- `Q_TILE/K_TILE`候補をboundedに絞り、C1のmerge/scratchを再利用できない重複設計を増やさない。
- S-Pをtarget/encoding/M/KV bucketごとに判断し、短Mまたはresource不足をB0へrouteする。

### P33-A4: C3 gfx1201 matrix inner prototype・採否

- C2のcorrectnessを満たすstable tile contractが定まった場合に限り、C2の採否にかかわらず同じtileへmatrix inner mathを
  一候補追加する。別上位providerやscore bufferは作らない。
- actual ISA、FP16 correctness、C2比device time、LDS/VGPR/occupancyを確認する。
- FP8/NVFP4はdirect decode costを含めて独立判断し、KV mirrorまたはN3差が必要ならC3 scopeから外す。
- 数値がN2なら測定結果をまとめてユーザー判断へ戻し、承認前にproduction defaultへ接続しない。

### P33-A5: production routing・full model

- operator段階で各scopeのwinnerを最大一つへ絞り、その候補だけをregistryへ接続してS-D/S-P/S-MとB0 complementをstable keyで実装する。
- F0〜F4を一度に一GPUだけでcounterbalance実行し、prefill/TTFT/TPOT/E2E、token/state、resource、fallback、cleanupを比較する。
- operator/full-model結果から担当AIが各scopeの最終採否を理由付きで確定する。

### P33-A6: integration・closeout

- 採用候補についてG4通常CLI/API lifecycleを実行する。
- `phase33-full-attention-summary-v1.json`、schema、host test、history、数値台帳、runtime/compatibility/main planを同期する。
- plan/historyを相互linkしてarchiveし、temporary profiler seam、force flag、不採用providerを除去する。

## 停止・再計画条件

- partial mergeまたはtileがcausal/GQA semanticを維持せず、N3差を解消するためpublic attention equation変更が必要になる。
- scratchがPhase 31の10k+/16,385 memory feasibilityを失わせ、bounded arena reuseまたはB0 complementで回避できない。
- candidateがfull score matrix、requestごとのallocation、host combine、KV mirror、Paged Attention/vAttention変更を必要とする。
- C1/C2が同じ機構で2回棄却され、variant追加だけが続く。新しいmechanism evidenceなしにtile/split候補を増やさない。
- C3がC2と独立したlayout/ABIを要求する、matrix ISAを生成しない、またはN2判断前にproduction接続が必要になる。
- verification/docsが作業の30%を超える、機能進捗が1時間以上止まる、見積りが1.5倍を超える、受入条件が変わる場合は
  新規variant/reviewを停止し、残るscopeを別Phaseへ分離する。

[Phase 30計画](../../../../archive/2026/08/11-20/phase30-rdna4-native-attention-kv-optimization.md)
[Phase 31計画](../../../../archive/2026/08/11-20/phase31-chunked-prefill-memory-foundation.md)
[Phase 32計画](../../../../archive/2026/08/11-20/phase32-native-fp8-kv-append-revalidation.md)
[runtime architecture](../../../../../architecture/runtime.md)
[KV memory decision](../../../../../architecture/kv-memory.md)
[数値・出力影響変更台帳](../../../../../compatibility/numerical-output-changes.md)
[provenance](../../../../../provenance/README.md)
[メイン計画](../../../../main-plan.md)
[実施履歴](../../../../../history/2026/08/11-20/phase33-full-attention-structural-optimization.md)
[bounded summary](../../../../../../ci/matrix/phase33-full-attention-summary-v1.json)

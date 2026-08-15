# Phase 15O: FP8/NVFP4 model量子化path最適化

> 状態: complete
> 作成日: 2026-08-15
> 完了日: 2026-08-15

## 目的

Phase 16のKV cache量子化へ進む前に、model weightのFP8 W8A8とweight-only NVFP4について、
decodeとprefillを別provider・別計測laneとして最適化する。VRAM削減を維持しながら、現在の量子化pathにある
activation量子化、packed decode、library solution選択、small-M/large-M不一致を解消する。

本Phaseは量子化formatを増やすPhaseではない。既存のFP8/NVFP4 sidecar、encoding、数値contractを維持し、
同じmodel graphのlinear bindingとprepared providerを改善する。Phase 16は本Phaseのcloseoutまたはユーザーによる
明示的な打切り・順序変更まで開始しない。

## 完了結果

- FP8は既存W8A8/hipBLASLt contractを維持し、BF16 activationのper-row dynamic量子化をwave reductionと
  native pair conversionへ置き換えた。exact `gfx1201`の代表M=1/M=32 shapeで旧kernel比7.48〜29.36%低遅延、
  Qwen3.5-4B 32/32でprefill `+5.89%`、decode `+10.69%`、E2E `-9.27%`となったため採用した。
- NVFP4 decodeのwave/scale共有候補は改善しなかったため棄却し、M=1は従来packed-dequant実装をprovider ID 8として
  維持した。prefillは一つのpacked weight K tileを最大8 M rowで共有するprovider ID 9を採用し、代表M=32 shapeで
  R9700は59.29〜59.51%、V620は51.21〜56.68%低遅延となった。全weightのBF16展開とresident/peak増加はない。
- FP8 `gfx1201`は`opt-in production`、`gfx1030` emulationは`correctness-only`、NVFP4はaccuracy budget超過を
  維持して両targetとも`correctness-only opt-in`とした。新しい`gfx942` candidateはMI300X不在のため有効化していない。
- operator oracle、Qwen accuracy、CLI/OpenAI non-stream/SSE/Unicode/stop/連続request、cleanup、workspace tests、
  manifest/docs validationを完了した。詳細な数値と棄却理由は対応履歴を正とする。

## 現状と開始baseline

- FP8はtext-linear weightをper-output-row scale付きE4M3FNで保持し、各linear入力のBF16 activationを
  実行直前にper-row FP8へ動的量子化する。exact `gfx1201`/`gfx942`はhipBLASLt W8A8、`gfx1030`は
  correctness-only emulationを使う。
- 現在のFP8 activation kernelは各rowを最大値取得と量子化で二度走査し、各linearで独立launchする。
  prepared hipBLASLt planはzero-workspace solutionを使い、decode M=1とprefill M>1を別の性能providerとして
  tuningしていない。
- R9700 Qwen3.5-4B 32/32ではFP8はBF16比でprefill約8.5%、decode約14.7%、E2E約16.7%遅い。
  MI300XではFP8 decode throughputがBF16の約75〜81%である一方、prefill-longは834.51対832.14 tok/sで
  BF16と同等だった。large-M FP8 GEMM自体よりsmall-M、量子化、launchの寄与を先に分離する。
- NVFP4はpacked E2M1、K-axis block 16 E4M3FN scale、tensor FP32 scaleをresidentに保持するが、
  現providerはdecode/prefillとも一つのworkgroupで一つのoutput elementを計算する
  `packed-dequant` kernelである。M>1でもweightとscaleをrowごとに再読込し、tiled GEMMになっていない。
- Qwen3.5-2BのNVFP4はresidentを52.43%削減したが、decodeはV620/R9700とも約20〜22%遅い。
  R9700 prefillはshort-oddで86.35対558.52 tok/s、32/32で87.01対755.68 tok/sと大幅に遅い。
- NVFP4は既存accuracy budgetを超えているため、性能改善だけで`correctness-only opt-in`からdefaultへ昇格しない。
  FP8もtarget/provider単位で採用判断を維持する。

開始時にfresh baselineを取り直し、上記履歴値と大きく異なる場合は新しい値を正本にする。過去値を現在candidateの
性能証拠として再利用しない。

## 対象と非対象

### 対象

- FP8 decode: exact `gfx1201`のOCP E4M3FNと、実機を利用できる場合のexact `gfx942` FNUZ。
- FP8 prefill: hipBLASLt solution、activation量子化、prepared workspace、shape routing。
- NVFP4 decode: exact `gfx1030`/`gfx1201`のpacked weight直接消費GEMV。
- NVFP4 prefill: exact `gfx1030`/`gfx1201`のpacked-dequant tiled GEMMまたは実測で確認したnative/library path。
- Qwen3.5のproduction shape、Qwen3.5-2B NVFP4、Qwen3.5-4B FP8、Gemma 4の代表linear shape。
- provider identity、prepared cache、workspace、dispatch audit、CLI/OpenAI pathの回帰。

### 非対象

- KV cache FP8/NVFP4、FP8/FP4 attention、MTP、vision、MoE、multi-GPU。
- NVFP4 activation、W4A4、別のmixed-bit format、calibration/imatrix、accuracy threshold緩和。
- sidecarからGGUFへの移行。GGUF統一はPhase 19のまま維持する。
- runtime自動tuning DB、requestごとのweight変換、全weightのBF16 materialization、暗黙fallback。
- FP8 emulationを`gfx1030`の性能providerへ昇格する作業。

## 固定する実行契約

- decodeは`M=1`、prefillは`M>1`として別provider ID、別shape guard、別prepared cache identityを持つ。
- FP8はweight/activation FP8、FP32 accumulation、BF16 outputを維持する。E4M3FNとFNUZを再解釈せず、
  activation scaleとweight scaleを明示的に適用する。
- NVFP4は既存packed value、block scale、tensor scaleを直接消費し、requestごとのunpackや全weight BF16複製を行わない。
- activation量子化結果を共有する場合は、source buffer ID、view、content generation、M/K、FP8 encoding、scale layout、
  exact targetをcache keyに含める。異なるtoken、state generation、OCP/FNUZ間で再利用しない。
- providerはprepare時に確定し、runtime failure後にBF16、別quantization、別providerへ成功扱いでfallbackしない。
- optimizationでsidecar bytesまたはscale規則を変更しない。format変更が必要になった場合は本Phaseを止め、
  converter/model-lockを含む別の変更として再計画する。

## 外部参照とprovenance

- AMD/ROCmのdatatype、ISA、hipBLASLt API・solution queryと独立数値oracleを実装上の正本にする。
- vLLM/SGLangのFP8は、dynamic quantization、producer fusion、shape routing、benchmark条件という技術的事実だけを
  reader記録へ固定する。implementationではその記録を使い、source expressionや近い構造をcopy、adapt、portしない。
- llama.cpp Q8/MMV/MMQはdecode/prefill分離、vector load、reduction、packed weight直接計算の実装候補として参照できる。
  直接reuseする場合は完全commit、path、hash、reuse区分、noticeをprovenanceへ記録する。
- FP8 matrix engineとinteger/block-Q8は同じ演算経路とみなさない。Q8由来の構造を採用してもFP8 hardware利用を
  dispatch evidenceで別に確認する。

## 受入条件

### correctness/security blocker

1. FP8/NVFP4の既存format oracle、sidecar hash、loader、provider fail-closed contractを維持する。
2. production shape、非整列K/N、M境界のoperatorを独立FP32 oracleへ照合し、NaN/Inf、zero、scale極値、tailを含める。
3. FP8 activation量子化を並べ替えまたは融合する場合、既存RNE、finite saturation、zero規則と生成scale/valueを
   byteまたは明示した数値許容差で照合する。
4. exact target、provider ID、logical/device symbol、fallback、workspace、quantization回数をauditできる。
5. cache/reuseはbuffer generationとrequest lifetimeを越えず、cancel/error/timeout時に未完了outputを公開しない。
6. full-model fixed/Unicode/stop generation、連続request、OpenAI non-stream/SSE、cleanupを最終candidateで回帰確認する。
7. FP8 accuracyは既存top-1/KLD budgetを維持する。NVFP4は既存budget超過を隠さず、optimization後に悪化を記録し、
   thresholdを緩めない。

### 性能candidateとproviderの採用判断

性能判断を、実装candidate、lane全体、provider default昇格の三段階に分ける。単一の固定率を三段階へ流用しない。

#### 実装candidateの採用

- primary比較は同一candidate、同一model、同一sidecar、同一GPU tupleの変更前量子化pathとする。BF16は製品判断用の
  controlとして併記する。
- O1のwarmup 3＋measured 3はcorrectnessと改善方向のscreeningだけに使い、candidateの最終採用根拠にしない。
- 最終採用はO2のwarmup 3＋measured 10を基本とし、baseline/candidateの実行順を反転またはcounterbalanceする。
  median、MAD、p10/p90、run順のdriftを比較し、primary caseの改善がそのtarget/caseで観測したnoise envelopeを
  越える場合に採用する。全target/case共通の3%等の固定floorは置かない。
- guard caseも同じtarget/case固有のnoise envelopeで判定し、説明不能な退行がnoiseを越えるcandidateは採用しない。
  別GPU、別model、decode/prefillの結果を同じnoise envelopeへ混ぜない。
- kernel単体の改善が小さくても、支配時間、呼出回数、複数model/targetでの一貫性からfull-modelへの累積効果を
  説明できる場合は採用できる。単一最良runまたは測定誤差内の符号だけでは採用しない。
- resident weight削減率を維持する。追加workspace、LDS/register、model load、peak VRAMは必ず記録し、速度改善を
  隠れた全weight展開で作らない。

#### lane全体の改善評価

- FP8 decode、FP8 prefill、NVFP4 decode、NVFP4 prefillごとに、開始baselineからBF16 gapをどれだけ回収したか、
  quantized path内でどのcandidateが寄与したかを分けて記録する。
- 各laneは最有力candidateを実装・計測し、採用または理由付きで棄却すれば完了できる。個々の有効な小改善を
  固定率未満という理由だけで捨てず、反対に大きな未解決gapを小改善だけで実用化済みと表記しない。
- 全pathのBF16超えをPhase完了条件にはしない。残る差をquantization、GEMV/GEMM、launch、memory、providerへ
  分解してhandoffする。

#### providerのdefault昇格

- 実装candidateの採用はproviderのdefault昇格を意味しない。accuracy、resident/peak VRAM、対象case全体の性能、
  exact target、fallback/auditを合わせて`default / opt-in production / correctness-only / converted`を判断する。
- FP8/NVFP4がBF16より遅くても、memory節約providerとして明示opt-inで残せる。defaultへ昇格する性能条件を
  release hard gateにする場合は、複数O2/O3履歴と再現性を揃え、main planの未解決事項に従って別途ユーザー承認を得る。
- NVFP4は既存accuracy budget超過が解消されない限り、性能改善だけで`correctness-only opt-in`から昇格しない。

### 目標値（nonblocking）

- FP8 decode: R9700/MI300Xで現状のBF16比15〜25% gapを回収し、可能ならBF16同等以上を目指す。
- FP8 prefill: large-Mの同等性能を維持し、short/medium promptのactivation量子化とsolution選択の損失を減らす。
- NVFP4 decode: 現状の約20〜22% gapを回収する。
- NVFP4 prefill: R9700の非tiled kernelによる大幅退行を解消し、packed residentを維持した実用的tiled pathを作る。

## 実装・検証順序

### P15O-A0: fresh profile、shape inventory、reader記録

- Qwen/Gemmaのlinearをdecode/prefill、M/K/N、共有activation、encoding、target/provider別に集計する。
- current FP8をactivation quantize、hipBLASLt、scale/output、launch/eventへ、NVFP4をvalue/scale load、decode、dot、
  reduction、launchへ分解する。rocprof traceはrepository外へ置き、文書にはbounded summaryとhashだけを残す。
- R9700でFP8 Qwen3.5-4BとNVFP4 Qwen3.5-2B、V620でNVFP4 2Bをfreshに測る。MI300Xは新しいVMが
  利用可能になった場合だけexact tupleを再取得し、過去VMの結果を新candidateのPASSにしない。
- vLLM/SGLang reader factとllama.cpp/AMDの実装候補を分離した短いreference noteを作り、以降の実装basisを固定する。
  固定結果は[Phase 15O reader記録](../../../../../references/phase15o-model-quant-optimization-reader.md)を正とする。

### Decode lane

#### P15O-D1: FP8 dynamic quantizationの重複除去

- 同一BF16 activationをQ/K/V、gate/up等の複数linearが消費する箇所を実測し、現状の量子化重複回数を数える。
- source buffer generationが同じ間だけ、FP8 activation value/scaleをrequest-owned prepared workspaceで一度生成して
  sibling linear間で再利用するcandidateを作る。
- RMSNormまたはactivation producerからFP8 value/scaleを直接出すfusionは、BF16 consumerの有無と追加writeを測り、
  standalone quantizeより有利なshapeだけに限定する。
- quantize kernel自体はvectorized BF16 load、wave shuffle reduction、scale write、second passのmemory accessをprofileし、
  OCP wave32とFNUZ wave64を別guardで扱う。

#### P15O-D2: FP8 M=1 provider

- hipBLASLtのsupported solutionをzero/nonzero workspace、algorithm、layout、alignment別に列挙し、production K/Nで
  prepared-timeに選ぶ。最初のheuristicを無条件採用しない。
- library M=1が支配的に遅い場合は、FP8 weight/scaleを直接読むcustom GEMVを検討する。activation quantizeとの
  launch統合、LDSに保持するquantized activation、複数output列/一workgroupを候補にし、matrix/FMA命令の実使用を監査する。
- `gfx1201` OCPと`gfx942` FNUZは別binary/providerとして測り、一方のsolution IDやtileを他方へ流用しない。

#### P15O-D3: NVFP4 M=1 provider

- 現在の256-thread/一output reductionをbaselineに、vectorized packed load、block 16ごとのscale再利用、tensor scale hoist、
  wave単位output、複数N列/一workgroupを順に比較する。
- activationを複数output列で共有し、packed nibble decodeとscale loadをregister/LDSへ置く。K tailとodd nibbleを
  分岐外のfast pathと明示tailへ分ける。
- exact `gfx1030`/`gfx1201`でwave、cache policy、occupancyを別に決める。native FP4命令/libraryはcapability probeが
  実shapeで成功した場合だけ別candidateとし、packed-dequantをnativeと呼ばない。

#### P15O-D4: decode統合判断

- operator micro後、Qwen3.5-2B/4B short-odd、32/32、decode-longを3 warmup＋10 measuredで比較する。
- TPOT、decode tok/s、quantization kernel count、linear wall time、launch/event、resident/peak/workspaceを記録する。
- FP8/NVFP4を独立に採用・棄却し、decode provider IDとshape guardを固定する。decode改善をprefillへ自動適用しない。

### Prefill lane

#### P15O-P1: FP8 activation量子化とshape bucket

- Mをshort、medium、largeの実shape bucketへ分け、per-row quantization、hipBLASLt、host/launch比率を測る。
- Decode laneのshared activation cache/fused producerをM>1で再評価し、追加workspaceとwrite量が有利なbucketだけ採用する。
- quantize kernelは複数row/workgroup、vector load/store、row reductionをtarget別に比較し、M=2/3/17、255/256/257を含める。

#### P15O-P2: FP8 M>1 hipBLASLt provider

- production shapeごとにsupported solution、必要workspace、alignment、warm/cold差を測り、M bucketでproviderを選ぶ。
- Q/K/Vまたはgate/upのcompatible GEMM groupingは、同じactivation量子化を共有でき、sidecar/graph複製なしで
  launchとmemory trafficが減る場合だけ候補にする。
- MI300X prefill-longの既存同等性能をguardとし、small-M最適化でlarge-M solutionを置き換えない。

#### P15O-P3: NVFP4 M>1 tiled provider

- `[M,K] x [N,K]^T`をM/N/K tileへ分け、activation tile、packed weight、block scaleをworkgroup内で再利用する。
- weightはpacked residentのまま読み、tile内だけE2M1とscaleをregister/LDSへ展開する。M rowごとの全weight再読込を廃止し、
  full BF16 weight bufferを作らない。
- `gfx1201`では利用可能なBF16/FP8/FP4 matrix pathをcapability queryとmicroで確認し、packed formatから直接接続できない
  libraryへ無理に全weight変換しない。`gfx1030`は独立したtiled scalar/vector pathとして扱う。
- M=2/3/17/32、255/256/257とproduction K/N、非整列K/Nでtile、tail、scale reuseを数値oracleへ照合する。

#### P15O-P4: prefill統合判断

- short-odd、32/32、prefill-longを3 warmup＋10 measuredで比較し、TTFT、prefill tok/s、GEMM/quant time、
  launch/event、resident/peak/workspaceを記録する。
- FP8/NVFP4とM bucketごとにproviderを採用・棄却する。prefillで速いtemporary expansionがpeak VRAMと
  resident削減の製品価値を失う場合は、別`converted-bf16` providerとして明示し、NVFP4 native/packed性能に含めない。

### P15O-I0: integration、GPU別判定、closeout

- decode/prefill providerを同一model graphへ接続し、prepared cache key、workspace lifetime、request cancellation、
  dispatch auditを回帰する。
- R9700/V620のaffected operator、full generation、CLI/OpenAI smokeを行う。gfx942 candidateを有効化した場合は、
  新しいMI300X VMのexact tupleでoperatorと代表full-model rowをfail-closedに再検証する。VMがなければ
  gfx942への新candidate enableを行わず、既存providerを維持する。
- target/encoding/laneごとに`default / opt-in production / correctness-only / converted`を決める。
- runtime、model lock、GPU/software compatibility、provenance、main plan、historyを同期し、1回のintegration reviewと
  findingだけのfocused re-reviewを行う。
- 本planをarchiveしてからPhase 16 KV cache FP8/NVFP4を開始可能にする。

## 計測matrix

| lane | target | encoding/model | 必須case | 主指標 |
| --- | --- | --- | --- | --- |
| operator decode | `gfx1201`、`gfx1030`、有効化時`gfx942` | FP8/NVFP4 | M=1、production K/N、K/N境界前後 | kernel時間、quant/dequant、bandwidth、provider |
| operator prefill | 同上 | FP8/NVFP4 | M=2/3/17/32/255/256/257、production/non-aligned K/N | tile効率、solution、workspace、数値誤差 |
| full decode | R9700/V620、再取得時MI300X | Qwen 2B/4B | short-odd、32/32、decode-long | TPOT、decode tok/s、E2E、launch |
| full prefill | 同上 | Qwen 2B/4B | short-odd、32/32、prefill-long | TTFT、prefill tok/s、GEMM/quant比率 |
| model guard | 利用可能target | Qwen＋Gemma slice | accuracy set、fixed/Unicode/stop | top-1、KLD、fallback、cleanup |

O0はoperatorと1回のcorrectness control、O1はcandidate screeningのwarmup 3＋measured 3とする。最終採否のO2は
baseline/candidateそれぞれwarmup 3＋measured 10を基本とし、実行順を反転またはcounterbalanceしてmedian、MAD、
p10/p90、driftを比較する。長いfull-model/GPU matrixはcandidate選定前に繰り返さない。

## 再計画・停止条件

- profileでlinear以外が支配要因だったlaneは、原因を記録して無関係なkernel tuningを続けない。
- sidecar format、scale semantics、accuracy policy、public ABIの非互換変更が必要なら本work unitを止め、別計画にする。
- 同じwork unitの2回reject、review時間が実装時間超、1時間以上の機能進捗停止、検証/docs 30%超、見積り1.5倍超、
  gate変更時は追加candidate探索を止めて再計画する。
- timeout、crash、CPU fallback、zero test selection、unsupported solutionはGPU PASSにしない。
- MI300X不在はlocal `gfx1201`/`gfx1030` workを止めない。新しいgfx942最適化のPASSだけを保留し、別targetの結果を移植しない。

[対応する履歴](../../../../../history/2026/08/11-20/phase15o-model-quant-path-optimization.md)

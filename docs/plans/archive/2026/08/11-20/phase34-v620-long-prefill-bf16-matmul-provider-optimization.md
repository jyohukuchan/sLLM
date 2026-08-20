# Phase 34: V620長行prefill BF16 matmul provider比較・最適化

> 状態: 完了（gfx1030長行shape限定採用、MTP correctness修正を併合）
> 作成日: 2026-08-20
> 完了日: 2026-08-20

## 目的

Phase 33後のdynamic FP8 KV、10,001 input / 2 output診断では、Qwen3.5-4Bの内部BF16 projection 248回が、exact `gfx1030`
V620で`matmul.bf16_fp32.tiled16.v2`を合計66.561秒、exact `gfx1201` R9700で`matmul.hipblas.gemm_ex.v2`を
合計0.642秒使用した。`M=1` projectionは22.42/17.29 msで1.30倍差に留まるため、長い`M>1`のprovider/solution差が
主要因という強い仮説を持つ。ただし約104倍にはGPU世代とlibrary solutionの能力差も含まれるため、寄与は同一V620上のB0/C1比較で確定する。

Phase 34はPhase 23 `P23-O5`を変更後の前提で再開し、次を一つのwork unitとして完了する。

1. V620の実production shapeでcurrent tiled16と既存hipBLAS BF16 GEMMを同一条件比較する。
2. 短い`M`と長い`M`のcrossover、shape依存性、数値差、library初回費用を測り、担当AIが採否scopeを理由付きで決める。
3. 採用する場合は既存共通hipBLAS実装を再利用し、exact targetとstable `M/K/N`だけを使うshape-aware routingをproductionへ実装する。
4. V620 long-prefill full modelで短縮のwall転化を確認し、R9700/gfx942、短prefill、decodeの既存経路を回帰する。

新しいGEMM kernelを先に作るPhaseではない。既存hipBLASが長いV620 shapeで勝つかを最初に反証し、勝つscopeだけを実装する。
全実shapeで勝たない場合はshape familyまたは`M` bucketごとにB0 complementを残す。既存hipBLASにも採用可能scopeがなく、
一つの明確な次kernel機構も固定できない場合は、推測的なGEMM実装を追加せずnegative completionとする。

## 開始根拠と前提変更

### 現時点の診断値

同一dynamic FP8 KV、10,001-token one-chunk、248 prefill projection、250 `M=1` projectionを持つfresh rocprofv3診断は次の値を示した。
raw DBとprofiler wallは採用証拠としてGit管理せず、Phase 34 A0でfinal Phase 33 identityから再取得する。

| 区分 | V620 `gfx1030` | R9700 `gfx1201` | V620/R9700 |
| --- | ---: | ---: | ---: |
| `M>1` prefill projection、248回 | 66.5609 s、tiled16 | 0.6417 s、hipBLAS | 103.72x |
| `M=1` projection、250回 | 22.419 ms、decode v4 | 17.289 ms、decode v4 | 1.30x |
| projection合計 | 66.5833 s | 0.6590 s | 101.04x |
| profiler E2Eからprojectionを除いた残差 | 23.525 s | 20.086 s | 1.17x |

- V620のprojectionはprofiler E2E 90.108秒の73.89%、R9700では20.745秒の3.18%だった。
- 両runはone chunkで、projection call数と全体submission/dispatch topologyが一致した。Phase 32/33の独立診断でも
  V620 prefill projectionは66.11/66.56秒、R9700は0.645/0.642秒であり、一般的なforeign processやclock noiseだけでは説明できない。
- V620のtiled16は16x16 output tile、K tile 16、K区間ごとに2 barrier、threadごとのBF16 decodeとscalar FP32 accumulationを使う。
  `K=9216`ではworkgroup当たり576区間、1,152 barrierとなる。
- R9700で観測したTensile kernelは`MT128x128x32_MI16x16x1`で、large macro tile、vector load、matrix-instruction系providerを使う。
  GPU世代差は実在するが、同じdecode/attention/GDNが概ね1.2〜1.3倍差であるため、約104倍をhardware差だけに帰属しない。

### Phase 9/23から変わった前提

- Phase 9はreal `M=17`だけを比較し、V620 hipBLASの主要projection 1.4〜2.2 msとvocab 32.4 msがcustom tiled16を
  上回らなかったため、V620 `M>1`をtiled16へ維持した。この判断は短行shapeについては有効だった。
- 現在の`select_variant(m,k,n,target)`は`k/n`を明示的に捨て、`gfx1030`の`M>8`をすべてtiled16へ送る。
  したがってPhase 9の短行判断が10,001行にもそのまま適用され、library初期費用を償却できる長行crossoverを表現できない。
- Phase 23 `P23-O5`は「V620がtiled16 matmulに支配される」ことを観測し、terminal-row除去後にtarget-specific `M>1`
  GEMMを再profileするよう求めたが、Phase 24以降は上位候補を先に実施し、O5自体は完了していない。
- Phase 24後はterminal vocabulary projectionが通常prefillで`M=1`になった。Phase 34の248回はvocabの巨大行列ではなく、
  transformer内部projectionだけであり、terminal-row最適化を戻さない。

## 権限・採否方針

- 本文書はユーザーの「比較・判断・実装まで含めた次Phase計画」作成指示に基づく。Phase 34のscopeと順序を確定するが、
  production source変更とGPU実行はユーザーの開始指示後に行う。
- 固定5%規則、全pattern一律非悪化規則は使用しない。担当AIがscopeごとの絶対短縮量、割合、分散、critical-path寄与、
  full-model転化、数値分類、library/resource費用、target/shape分岐、保守費用、将来再利用性、rollback容易性を総合して採否する。
- あるshape/bucketでC1が妥当でも別bucketで不利なら、勝つscopeだけC1へ送り、他はB0を維持できる。別scopeの悪化を
  候補全体の自動棄却理由にも、勝つscopeの自動採用理由にも使わない。
- common semantic/runtime pathを維持し、target差は既存provider registryのselectionへ閉じ込める。gfx1030/gfx1201でgraph、
  tensor layout、public ABIを複製しない。
- 数値分類は`docs/compatibility/numerical-output-changes.md`のN0〜N3を適用する。N0/N1は数値gateを通常または自動承認できる。
  reduction順変更が決定的で、candidate固有の解析的上限を示し、誤差bound増加が既存tolerance内へ有界と説明できる場合だけN2とし、
  性能・誤差・token結果を提示してproduction接続前にユーザー判断へ戻す。内部順またはboundを特定できない場合はN3であり採用しない。

## Phase固有の範囲

### Primary対象

- model: fixed Qwen3.5-4B dense BF16 GGUF/derived lock、通常text prefill。
- primary target: Radeon Pro V620 exact `gfx1030`、UUID `GPU-76a08c022586fed6`。
- regression target: Radeon AI PRO R9700 exact `gfx1201`、UUID `GPU-a8e9ddefa2d60f55`。
- compile control: exact `gfx942`/ROCm 7.14契約。実機性能claimは追加しない。
- operation: BF16 `[M,K] x transposed [N,K] -> BF16 [M,N]`、FP32 computeの内部projection。
- mode: current one-chunk prefillとautomatic chunked prefill。`M=1` decodeと`M=2..8` serial-row pathは回帰controlだけとする。
- providers: current `PrefillTiled16`と既存`HipBlas` runtime path。gfx1030には現状model-context hipBLAS handleがないため、
  C1比較と採用にはgfx1030でhipBLASだけをcontext-lifetimeに作るbounded prerequisiteを含む。hipBLASLtはgfx1201/gfx942限定を維持する。
- full-model KV: FP16を性能primary、dynamic FP8をmatmul routeがKV encoding非依存であることのcontrolとする。
- fixed llama.cppはprovider topologyとE1 system gap closureの診断controlに使う。直接reuseが生じる場合だけprovenance手順を適用する。

### 非対象

- BF16 `M=1` matvec、serial rows、terminal LM head、Argmax、RMSNorm、attention、GDN、elementwise kernelの再最適化。
- FP8/NVFP4/MXFP4 weight matmul、activation quantization、KV format/default、TurboQuant、DeepSeek V4、MoE/grouped GEMM。
- gate/up fusion、projection間activation共有、weight repack/layout変更、operator数削減、HIP Graph、event/completion最適化。
- chunk selector、vAttention、Paged Attention、prefix cache、continuous batching、multi-GPU、public API、GGUF/model-lock変更。
- gfx1200、別RDNA2/RDNA4 SKU、別model shapeへの性能一般化。
- hipBLASが不採用だった場合に、根拠なく複数のcustom tile、hipBLASLt solution、llama.cpp kernelを同じPhaseへ追加すること。

## Production shape inventory

Phase 24後の各main prefill chunkは、terminal vocabulary projectionを除く次の248 BF16 projectionを持つ。`M`はselected chunk rowsである。

| family | K | N | chunk当たり回数 | 10,001診断での優先度 |
| --- | ---: | ---: | ---: | --- |
| MLP gate/up | 2,560 | 9,216 | 64 | 最上位、N=9216 aggregate 28.16 s |
| MLP down | 9,216 | 2,560 | 32 | 上位、N=2560 aggregateへ含む |
| GDN/full q系 | 2,560 | 8,192 | 32 | 上位、12.47 s |
| GDN z | 2,560 | 4,096 | 24 | 中位、4.69 s |
| GDN/full out | 4,096 | 2,560 | 32 | 上位、N=2560 aggregateへ含む |
| full k/v | 2,560 | 1,024 | 16 | control、0.78 s |
| GDN b/a | 2,560 | 32 | 48 | small-N control、0.084 s |

上位四shape familyをscreenのprimaryとし、N=4096/1024/32をselector complementと全体加重時間のcontrolに使う。
`calls_family × median_device_time(M,K,N)`を合計した予測projection aggregateを作り、fresh full-model profileの実aggregateと照合する。
通常terminal LM headはfinal chunkで別の`M=1,K=2560,N=248320`、intermediate chunkでは省略される。明示all-logits pathは
同じwide-vocabulary K/Nを`M>1`で要求し得るため、Phase 34のproduction K/N membershipへ含めず現行providerを維持する。

## Baseline・candidate・routing

### B0: current gfx1030 tiled16

- exact `gfx1030`、`M>8`を`matmul.bf16_fp32.tiled16.v2`へ送る現行default。
- 16x16 output/K tile、256 thread、BF16 input/weight、source-level scalar FP32 accumulation、BF16 RNE output。
- `M=1` decode v4、`M=2..8` serial rows、gfx1201/gfx942 hipBLASはB0のまま固定する。

### C1: existing hipBLAS BF16 GEMM on gfx1030

- existing `hipblasGemmEx` runtime pathへ、candidate buildでgfx1030 context-lifetime hipBLAS handleを追加して接続する。
  weight transpose、BF16 A/B/C、
  `HIPBLAS_COMPUTE_32F`、`HIPBLAS_GEMM_DEFAULT`を変更しない。
- gfx1030には`hipblasCreate`だけを追加し、FP8用hipBLASLt handle作成条件は変更しない。per-request handle、weight repack、
  persistent duplicate、explicit workspace、fallback retryを追加しない。
- comparison時は同一binaryからB0/C1を明示選択できるprivate evidence seamを使い、first-callとwarm-callを分離する。
  既存`SLLM_MATMUL_FORCE_BASELINE=1`はtiled16ではなくscalar v1を選ぶため比較switchへ使わない。環境変数やforce switchを
  production sourceへ残さない。
- actual selected Tensile kernel、solution name、grid、library statusを記録し、`hipblasGemmEx` labelだけで同一mechanismとみなさない。
- original B0 context create、handle付きcandidate context create、first GEMM、steady GEMMを分離し、handle作成の失敗面、時間、
  resourceとmodel-ready影響をC1 costへ含める。

比較上は、production B0（handleなし+tiled16）、diagnostic B0H（gfx1030 hipBLAS handleあり+tiled16）、C1（同じB0H contextで
hipBLAS）の三つを区別する。B0対B0Hでcontext/model-ready cost、B0H対C1でkernel差、B0対final C2でuser-visible総差を測る。
B0Hは原因分離用でありproduction providerとして残さない。

### C2: shape-aware production selection

C1の勝つscopeが確定した場合だけ実装する最終candidateである。compute implementationはC1を再利用し、selectionだけを追加する。

1. 全actual shapeで同じcrossoverが安定する場合も、`exact target + measured internal K/N membership + common M threshold`を使う。
   Mだけの一般ruleにはせず、明示all-logitsの`K=2560,N=248320`や未知shapeを変更しない。
2. shape依存なら同じ7 production K/N familyにfamily別M thresholdを持つ小さなstatic tableを使う。
3. threshold `B`は測定後に固定し、`B-1/B/B+1`、historical `M=17`、long production値をfinal binaryで検証する。
4. 表にないK/N、unsupported target、短M、library/capability不成立はdispatch前にB0を選ぶ。実行失敗後のretry fallbackは行わない。
5. prompt内容、token値、KV encoding、DPM/clock、benchmark名、実行時の自己計測をrouting keyにしない。

C2はgfx1030専用の別graphではなく、既存共通`select_variant(m,k,n,target)`が既存共通HipBlas variantを返す範囲の追加である。
gfx1201/gfx942のcurrent `M>8` HipBlas選択はsource上もdispatch evidence上も不変にする。

## Comparison freezeとcrossover探索

### 二段階のM/shape matrix

全M/K/Nの直積は作らない。最終routeを決める前に、次のbounded matrixを同じprovider pairで測る。

1. **screen**: 全7 production shapeを`M=17/256/2048/10001`でB0/C1比較する。Phase 9、Phase 23、2K、現行long caseを接続する。
2. **scaling**: 上位4 shapeだけを`M=8/9/16/17/64/255/256/257/511/512/513/1024/2047/2048/2049/`
   `4095/4096/4097/8191/8192/8193/10001/16383/16384/16385`からboundedに比較する。8/9はserial/tiled境界、
   非整列三点は既存/automatic chunk境界を表す。全点を全shapeへ機械的に掛けず、screenとcrossover区間から必要なpairだけ選ぶ。
3. **refinement**: winnerが切り替わる区間だけ追加点を取り、各採用scopeの最終threshold `B`を一つ決める。final binaryでは
   `B-1/B/B+1`とscope内の離れた2点以上を再確認する。
4. **complement**: 実shapeの`K/N`から一つずらした非production shape、odd/non-aligned small shape、M=1/2/8を使い、
   exact-shape tableが未知入力を誤ってC1へ送らないことを確認する。

`M=10001`は一dispatchの現行通常経路、`M=16384`はPhase 31 auto chunk上限、`M=17`は旧棄却根拠である。
総prompt 16,385はproductionでは`M=16384`と`M=1`へ分割され、`M=16385`一dispatchではない。後者はselector/tail controlに限定する。
短M悪化を隠さず、長Mの勝ちと同じ図表へ載せる。Phase 23 O5の旧「10〜40%期待」や「residual 10%」は前提変更前の
探索estimateであり、Phase 34のthreshold、採用gate、期待値へ使わない。

### fixed llama.cpp control

- current source-lockのllama.cpp `b10453` / `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`をfacts-only primaryとする。
- fixed sourceの通常BF16 pathは、gfx1030で`M=2..8`をMMVF、`M>=9`をhipBLAS、gfx1201の整列Qwen shapeで
  `M=2..3`をMMVF、`M=4..16`をWMMA MMF、`M>=17`をhipBLASへ送る。この事実はV620 C1を測る根拠と
  `M=8/9` boundary controlに使うが、source topologyを性能勝敗そのものへ読み替えない。
- llama.cpp library boundaryはBF16 weight、F32 activationをBF16 temporaryへ変換、F32 compute/outputであり、sLLMの
  BF16 activation/output contractとは一致しない。`GGML_CUDA_CUBLAS_COMPUTE_TYPE`はunset/autoを固定し、operator数値・速度を
  同一contractとして直接比較しない。
- peer full-model/prompt-processingを実行する場合は、Qwen source revision、BF16 logical weights、token IDs、prompt length、GPU、
  batch/chunk、MTP/KV、build coverage、timing境界をmanifestへ固定する。artifact layoutまたはfrontendが異なる結果はE1/E2とし、
  sLLM provider採否のhard evidenceへ使わない。
- C1は既存vendor-library callを再利用するため、llama.cpp sourceのdirect reuseはなく、新規provenance eventもない。
  C1が失敗し、固定sourceから一つの具体的custom mechanismを次候補にする場合だけ、別work unitとしてreuse/provenance costを提示する。

## 数値・出力contract

- real-number equationは`C[row,column] = Σ_k A[row,k] * W[column,k]`、BF16 input/weight、FP32 accumulate、
  BF16 round-to-nearest-even outputのまま維持する。row/column order、shape、tensor layout、state publication、public ABIを変えない。
- B0はKを昇順にsource-level scalar accumulateする。C1は同じdtype/compute contractだがTensile solutionのtile/reduction/FMA順が
  変わり得るため、bit exactでないこと自体をdefectにしない。
- model-free oracleはdecoded BF16 productをF64で求め、既存Phase 8の`gamma_K * Σ|a_i w_i| + BF16 half-ULP` bound、
  finite/NaN/Inf classification、signed zeroを維持する。toleranceをcandidate結果を見て拡張しない。
- large production shapeは全F64 outputをCPUで再計算しない。bounded distinctive row/column sampleをF64 oracleへ照合し、
  全outputはGPU-sideまたはreadback aggregateでfinite分類、baseline/candidate mismatch count、max absolute/relative error、repeat digestを確認する。
- Kの加算順、actual library solution、dependency depthを特定でき、worst-case boundが非増加と示せる場合だけN1とする。
  measured bit exactかつ演算/丸めstageも不変ならN0とできる。決定的な演算順とcandidate固有boundを特定でき、僅かな増加が
  既存tolerance内に有界ならN2とする。内部順・誤差方向・boundを説明できない、非決定、bound外、semantic変更はN3とする。
- full-modelではprompt/completion token IDs、最初のtoken/logit分岐、top-1 margin、usage、KV/GDN committed lengthを記録する。
  token差がある場合はbounded op digestで最初に差が現れるlayer/projection familyを特定し、provider reduction順から説明できるか確認する。
  raw activation、full logits、生成全文は追跡しない。
- N2はPhase 33 C1と同様に分類をN1へ見せかけず、数値差、速度、scope、rollbackを提示してユーザー承認前にdefault routeへ接続しない。

## Resource・failure contract

- C1/C2 candidateはgfx1030 contextへhipBLAS handleを一つ追加し、既存context destructor、mutex、queue stream設定を再利用する。
  gfx1030 hipBLASLtは作らず、requestごとのhandle生成、per-dispatch `hipMalloc`、duplicate weight、repack、persistent tuning DBを導入しない。
- handle create/destroy、context failure cleanup、model-ready time、resident/host resourceをB0/C1で測る。短requestがC1を選ばなくても
  context handle costは発生するため、long kernel利益から独立したadoption costとして扱う。
- hipBLAS内部のlazy initialization、code object load、solution cache費用はfirst callとして別計測し、warm kernel timeへ隠さない。
  model-ready、first prefill、steady requestのどの境界へ費用が載るかを記録する。
- model-resident bytes、request workspace high-water、VRAM/GTT、library workspace、allocation countをB0/C1で比較する。
  unaccounted allocation、GTT spill、OOM、resource lifetime破壊を局所速度で相殺しない。
- providerはprepare/dispatch前に決める。C1 submissionが失敗した後でB0へsilent retryせず、output/stateをpublishしない。
- existing handle mutexによるqueue間直列化を記録し、Phase 34のsingle-request latency claimをconcurrent throughputへ一般化しない。
- wrong target、unsupported shape、`int`へ収まらないM/K/N、dimension overflow、null handle、stream設定失敗はdispatch前または既存境界で
  fail closedとし、unchecked `uint64_t`→`int` castを新scopeへ広げない。
- cancellation、timeout、completion drop、queue failure、shutdownでpending outputを公開せず、retryable/durable cleanupを0へ戻す。

## 測定contract

### Identity・health

- Phase 33 final source/buildをbaselineとし、source tree、candidate diff、kernel/runtime/evidence source、binary、ROCm、LLVM、
  Code Object、release flags、model/derived lock、GGUF、exact GPU UUID/BDFをdigest-boundにする。
- software tupleはcurrent compatibility正本のUbuntu/kernel/amdgpu/ROCm 7.14.0/LLVM 23を使い、異なるtupleを同一claimへ混ぜない。
- local Qwen serviceを停止してV620 pairを解放し、各runは一度に一GPUだけをUUIDで可視化する。foreign process、VRAM/GTT、
  temperature、clock、power、ECC、throttleをpre/during/postに記録する。
- CPU/backend fallback、partial offload、timeout、crash、zero sample、別target、GTT spillをGPU PASSにしない。

### Operator timing

- kernel比較はB0H/C1を同一candidate binary、同一buffer、同一stream/handle lifetimeでcounterbalanceする。original B0とB0Hは
  context/model-ready costの別laneで比較する。temporary force seamのselection overheadを
  kernel時間へ含めず、final routeのprepare/dispatch overheadは別に測る。
- operatorの初期測定targetはfirst callを独立記録し、その後warmup 5、measured 21のHIP event device timeとする。
  これはPhase 33 operator protocolを起源とするnonblocking sampling proposalで、scope freeze時にconfidenceが十分なら機械的な追加取得を
  completion gateにしない。median、MAD、p10/p90、min/max、absolute ns、relative差、provider/kernel nameを残す。
- large tensorのH2D/D2HとCPU oracleはtimed regionから外す。activation/weight/output allocationとinitializationは両providerで共有する。
- top gap familyだけrocprof/counter laneを別runで取得し、DRAM/L2 traffic、VALU/matrix instruction、barrier、wave stall、occupancy、
  VGPR/LDS、actual Tensile solutionを比較する。profiler/counter wallをproduction performanceへ使わない。
- providerごとのweighted projection aggregateを248-call inventoryから算出し、instrumented full-model aggregateと照合する。

### Full-model timing

- profilerなしのproduction direct/CLI laneを正とし、baseline/candidate順をfresh process間でcounterbalanceする。
- full-modelの初期sampling proposalはlong lane各variant 3 independent process、processごとに1 warmup + 3 measured、短lane
  3 warmup + 10 measuredとする。originはPhase 32/33のprocess/DPM drift、costは数十分規模、expiryはP34-A2 scope freezeとし、
  fixed sample数自体をhard completion gateにしない。判断に必要なconfidenceを得た時点で止め、分散で判断不能な場合だけ追加する。
- TTFT、prefill span、projection device aggregate、attention/GDN、first token、E2Eを分ける。2 output primaryはprefill転化を、
  32 output controlはdecode route非変更を確認する。
- raw profiler runとprofilerなしwallを混ぜず、同じsemantic/build identityの対応だけを使う。

## Verification matrix

### H0: host/build/routing

- exact gfx1030/gfx1201 build/link、gfx942 compile-only、wrong-target code object load拒否。
- B0/C1/C2 provider ID、logical/device symbol、M/K/N、row/normalized size、dispatch count、fallback metadataがactual launchと一致する。
- `M=0/1/2/8/9`、actual shape、K/N `±1`、`INT_MAX`/dimension multiplication境界、unknown targetをhost/prepare evidenceで確認する。
- gfx1030 candidate contextのhipBLAS create/destroy/failure cleanupと、gfx1030 hipBLASLt非作成を確認する。
- final threshold/tableについて各`B-1/B/B+1`、scope内2点以上、scope外shapeを確認する。
- gfx1201/gfx942 `M>8` HipBlas、gfx1030 `M=1` decode v4と`M=2..8` serial rowsの選択が不変である。

### G1: BF16 numerical/provider evidence

| class | representative M/K/N | 目的 |
| --- | --- | --- |
| tiny/special | M=1/2/3/8/9/17、K/N=1/3/17/31/32/33/255/256/257 | NaN/Inf/subnormal/tie/tail、existing oracle |
| production-K | M=9/17/255/256/257、actual K/N 7 family | provider境界、実reduction長 |
| 2K boundary | M=2047/2048/2049、上位4 shape | historical P23-O5境界、crossover |
| long | M=4095/4096/4097、8191/8192/8193、10001、16383/16384/16385からbounded、上位4 shape | sampled F64、全output aggregate、repeat |
| final threshold | 各B-1/B/B+1、採用shapeとK/N±1 | production selectorとcomplement |

全caseでnumerical bound、finite classification、provider metadata、repeat determinism、fallback false、cleanup 0を確認する。
large matrixはpairwise/bounded sampleとし、全shape/MのF64 Cartesian productを作らない。

### G2: operator performance

- gfx1030はscreen/scaling/refinement matrixを実行し、family別B0/C1、weighted 248-call aggregate、first/warm costを採否へ使う。
- gfx1201はcurrent HipBlasのrepresentative short/long shapeとfinal selector overheadだけを確認し、V620候補とのcross-target絶対速度を
  採否gateにしない。
- fixed llama.cppを実行する場合はV620/R9700の256、2048、10001 prompt-processingをE1/E2区分付きで記録し、
  sLLM B0/C2のsystem gap closureをdiagnosticとして示す。

### G3: full model

次表はcandidate poolであり、全rowの実行をhard gateにしない。P34-A2で実際のcrossover/adoption scopeを確認後、
primary long、short complement、代表R9700 regressionと、scopeを説明する最小のscaling/boundary rowへfreezeする。

| case | target | input / output | KV | 目的 |
| --- | --- | ---: | --- | --- |
| F0 | gfx1030/gfx1201 | 17 / 32 | FP16 | historical short complement、decode不変 |
| F1 | gfx1030/gfx1201 | 255/256/257 / 2 | FP16 | old prefill boundary、短M route |
| F2 | gfx1030 | 2047/2048/2049 / 2 | FP16 | crossover/2K continuity |
| F3 | gfx1030/gfx1201 | 4108 / 2 | FP16 | Phase 30/33 continuity |
| F4 | gfx1030/gfx1201 | 10001 / 2 | FP16 | primary one-chunk long-prefill採否 |
| F5 | gfx1030 | 10001 / 2 | dynamic FP8 | KV encoding非依存route control |
| F6 | 実行可能target | 16385 / 2 | FP16 | 16384+1 chunkとsecond-chunk complement |

- 必須最小laneはV620 F4、V620 F0、R9700 F4とする。F1/F2/F3からfinal thresholdまたはscalingを説明する一つ以上を選ぶ。
  F5はKV encoding横断claimが必要な場合、F6はadoption scopeが`M=16384`を含む場合だけ追加する。F6はpreflight/health上安全なtargetだけで
  実行し、V620でのOOMを性能FAILへ読み替えない。
- baseline/candidateのprompt/completion token、visible output、stop、usage、KV/GDN state、selected provider count、HIP-only、
  fallback、VRAM/GTT、cleanupを記録する。
- gfx1201はselector/current provider不変を確認するfocused controlであり、V620と同じfull performance matrixを要求しない。

### G4: integration/lifecycle

- C2採用時だけ、V620の通常CLIとOpenAI non-stream/SSEで代表10k+ promptから1〜2 token生成し、usage、first token、`[DONE]`、
  disconnect/cancel、直後recovery、graceful shutdownを確認する。
- shared service/runtime codeを変更しない場合、R9700は通常CLI long regressionとwrong-target/load controlに限定し、Phase 33 API matrixを
  機械的に再実行しない。
- shutdown後のcurrent/request/workspace bytes、retryable cleanup、durable quarantineを0とする。

## 受入・採否基準

### Hard correctness・resource条件

1. BF16 matmulのshape/layout、real-number equation、FP32 compute、BF16 RNE output、state/publication、public ABI/APIを維持する。
2. candidateはPhase 8 F64 bound、finite/special classification、repeat determinismを満たし、N0〜N3分類と根拠を持つ。
3. N3、wrong-target/unsupported-shape誤route、provider metadata不一致、CPU/runtime fallback、timeout/crash、zero test selectionをPASSにしない。
4. unaccounted workspace、duplicate weight、GTT spill、OOM誘発、partial publication、cleanup failureを性能で相殺しない。
5. C1 failure後のsilent B0 retryを行わず、失敗したsubmissionのoutput/stateを公開しない。
6. gfx1201/gfx942、gfx1030 decode/serial rows、scope外shapeはfinal sourceで既存providerへrouteされる。

### 担当AIによるperformance adoption

固定改善率や全pattern非悪化を置かず、次をscopeごとに総合する。

- device timeの相対差だけでなく、1 callと248-call weighted aggregateの絶対短縮秒。
- F4 10,001-token TTFT/prefill/E2Eへの転化、F2/F3 scaling、first-call lazy cost。
- bracket drift、MAD/p10/p90、複数process間の一貫性、clock/thermal confidence。
- 実productionでのfamily頻度、採用M範囲、chunk policyとの整合、Amdahl share。
- numerical N0/N1/N2、token/logit影響、oracle margin、数値承認cost。
- selector tableの単純さ、既存HipBlas再利用、resource/handle/cache費用、ROCm solution依存性。
- exact target splitの保守費用、scope外B0 complement、将来GPU/model shapeへの再利用性、rollback容易性。

判断は次の形で行う。

1. **shared long-M scoped adoption**: 全actual shapeに一つのstable thresholdが適合し、利益と単純さが費用を上回る。
2. **shape-family scoped adoption**: 一部familyだけstableに勝ち、K/N tableの保守費用をその絶対短縮が上回る。他familyはB0。
3. **reject/negative completion**: 改善が不安定、first/full-modelで相殺、N3/resource defect、または細かい表の保守費用が利益を上回る。

operatorが大幅改善してもfull-modelで転化しない理由を説明できなければ自動採用しない。一方、scope外でC1が悪化しても、
final selectorがその入力をB0へ確実に隔離し、採用scopeの利益が管理費用を上回るなら採用できる。gfx1201の改善率をV620採用の条件にしない。

### Evidence・closeout

7. final source/tree、binary、runner、ROCm/GPU/model identity、routing table、selected solution、numerical class、raw artifact digestを
   bounded summaryへ固定する。
8. raw model、binary、trace DB/counter dump、full output/logits、生成全文をGitへ追跡しない。summary/schema/test、aggregate、
   plan/history/台帳/compatibilityだけを残す。
9. candidate別の改善/悪化、測定限界、採否理由、adoption scope、B0 complement、rollback、再検討条件を明記する。
10. affected host/build/GPU/full-model checkと一回のintegration reviewを行い、findingがあれば変更箇所だけfocused re-reviewする。
11. 採用時はruntime、GPU/AMD/software compatibility、数値台帳、main planを同期する。不採用時はforce seam、unused variant/table、
    debug timingをproduction sourceから除去し、negative resultだけを記録する。

## 実装方針

- 最小production変更は、`public_runtime.hip.cpp`でexact gfx1030へcontext-lifetime hipBLAS handleだけを作成し、
  `select_variant(m,k,n,target)`で合格したproduction K/N/M scopeだけ`KernelVariant::HipBlas`を返す二点である。
  `matmul_runtime.inc`のexisting `hipblasGemmEx` call、handle mutex、stream設定、dispatch/completion contractを再利用し、
  `matmul_kernel.hip.cpp`、Qwen graph、public ABIは変更しない。
- `k/n`を捨てる現行codeは廃止し、selector helperへ7 production shapeのmembershipと共通またはfamily別M thresholdを置く。
  判定はchecked integral compareだけとし、allocation、string生成、runtime timing、library queryをprepare hot pathへ追加しない。
- final provider ID/device symbolは既存`matmul.hipblas.gemm_ex.v2`/`hipblasGemmEx`を使い、同じcompute pathへ新IDを重複追加しない。
- evidence runnerはprovider override、production-K/N、threshold boundary、large performance modeを比較できるよう拡張する。
  huge matrixのCPU全output oracleとperformance timingを同じmodeへ詰め込まず、correctness/performance reportを分離する。
- Phase 22 matvec evidenceのdevice-resident buffer、interleaved sample、cleanup patternをrunnerのtemplateに使うが、M=1固定やall-one入力を
  そのまま流用しない。current matmul G1のF64/special/provider identityをcorrectness側へ維持する。
- final sourceではtemporary overrideを除去し、production selectorのactual dispatch evidenceを取得する。rollbackはgfx1030の
  K/N/M ruleを除去してB0 tiled16へ戻し、gfx1030 contextのhipBLAS handle作成を除去する二点とする。
- hipBLAS内部solutionを固定APIで選べずROCm updateで変化する場合は、ROCm/toolchain identityとactual kernel nameをevidenceへ残す。
  runtime autotunerや永続solution cacheはPhase 34へ追加しない。
- direct llama.cpp code reuseは予定しない。必要になった場合はPhase 34を無断拡張せず、exact source/blob/license、reuse mode、
  notice、implementation/verification costを示して別candidateの承認を得る。

## 作業順序

### P34-A0: acceptance・identity・fresh baseline

- Phase 33 final semantic/source identity、current provider selector、Qwen shape/call inventory、Phase 9/P23-O5履歴を固定する。
- local Qwen serviceを停止し、canonical V620/R9700を解放する。GPU healthとsoftware/model identityを記録する。
- V620のF4相当10,001 fresh profileとF0 short controlを取得し、`M/K/N/role/provider`別device aggregate、call count、
  E2E residual、first-call境界を再確認する。R9700 fresh long baselineはPhase 33 final identityと対応する証拠を再利用できない場合だけA0へ追加する。
- 診断の約104倍差またはV620 10k projection支配が再現しない場合は、旧DBを採用根拠にせずscopeをreplanする。
- numerical tolerance、operator matrix、full-model cases、performance判断項目を実装前にsummary/schema draftへ固定する。

### P34-A1: same-binary B0/C1 comparison seam・oracle

- candidate buildでgfx1030 contextへhipBLAS handleだけを追加し、create/destroy/failure/resource/model-readyを測れるようにする。
- production ABIへ露出しないprivate evidence selectionを追加し、同一candidate binaryでgfx1030 tiled16 B0/C1を選べるようにする。
  scalar v1へ切り替える既存force-baseline flagは使わない。
- existing 18-case matmul evidenceを弱めず、tiny special、actual K/N、large sampled-oracle、provider metadataを追加する。
- gfx1030/gfx1201のB0 selectionとC1 mechanismを確認し、first/warm call、allocation/workspace、actual library kernelを記録する。
- H0/G1を通し、C1の暫定N0〜N3分類を行う。N3なら性能比較で救済せずC1を棄却する。

### P34-A2: staged provider comparison・scope freeze

- gfx1030 screen、上位shape scaling、crossover refinementを順に実行し、全直積を避ける。
- weighted 248-call aggregate、first-call、counter/profile mechanism、full-model Amdahl見積りを作る。
- current pinned llama.cppのgfx1030/gfx1201 BF16 provider topologyを固定し、必要な場合だけE1/E2 prompt-processing controlを取得する。
- 担当AIがC1の採用候補scopeと最小routing keyを決める。7 production K/N membership内の共通thresholdを先に検討し、
  必要な場合だけfamily別thresholdへ狭める。Mだけでunknown/all-logits shapeへ一般化しない。
- final threshold/table、B0 complement、B-1/B/B+1、final numerical classをproduction実装前にmanifestへfreezeする。
- C1がN2なら、ここでoperator/full-model予測、oracle/token差、scope、rollbackをユーザーへ提示し、承認までP34-A3のdefault接続を止める。

### P34-A3: C2 production routing実装

- frozen scopeだけを`select_variant`へ実装し、gfx1030 context-lifetime hipBLAS handleと既存HipBlas runtime pathへ接続する。
  gfx1030 hipBLASLtとgfx1201/gfx942 handle ruleは変更しない。
- selector/evidence expected provider、logical/device symbol、grid/row metadata、unknown shape/target complementを更新する。
- temporary force seamを除去し、final selectorからB0/C1双方を到達可能にする。
- host/build/compile、gfx1030/gfx1201 focused numerical、gfx942 compile-onlyを実行する。

### P34-A4: final operator・full-model・regression

- final binaryでthreshold boundary、scope内代表、scope外shape、weighted aggregateを再取得する。
- V620 F4とF0、R9700 F4を最小final setとし、F1/F2/F3からfrozen threshold/scalingを説明する最小rowを追加する。
  F4のprojection/TTFT/E2E転化、short complement、R9700 existing HipBlas route、token/state、resource、cleanupを確認する。
- F5はKV encoding横断claimが必要な場合、F6はfinal scopeが`M=16384`を含みsafe preflightを満たす場合だけ追加する。
  OOM可能なV620 long laneを無理にrelease gateへしない。
- provider audit、fallback false、GTT spillなし、cleanup 0をfinal identityへ結び付ける。

### P34-A5: contextual adoption・integration・closeout

- 担当AIがscopeごとにadopt/rejectを確定し、絶対/相対改善、悪化、confidence、数値/resource、分岐/保守費用を理由付きで記録する。
- 採用winnerだけ通常CLI/API lifecycleを実行する。不採用scopeのforce switch、debug instrumentation、unused sourceを除去する。
- `phase34-v620-prefill-matmul-summary-v1.json`、schema、host test、history、数値台帳、runtime/compatibility/main planを同期する。
- 一回のintegration reviewとfindingのfocused re-reviewを行い、plan/historyを相互linkしてarchiveする。

## 停止・再計画条件

- fresh post-Phase-33 profileでV620 long-prefill projectionが支配的でない、または診断差がproviderへ再現可能に局所化できない。
- C1がactual long shapeでB0に勝たず、追加候補に必要な機構がweight layout、新GEMM family、hipBLASLt autotuning等へ広がる。
- N3、非決定、bound外、wrong-target route、silent fallback、unaccounted workspace/GTT、state/cleanup defectが解消できない。
- stable crossoverを作れず、prompt/benchmark/clock依存keyまたは過度に細かいK/N/M tableでしか勝ちcaseを表現できない。
- 同じC1 work unitが2回reject、review時間が実装時間超、functional progressが1時間停止、verification/docsが30%超、
  見積り1.5倍超、acceptance変更時はvariant追加を止め、同じwork unitをreplanする。

## 完了結果

existing hipBLASをexact gfx1030の長行production shapeへ限定採用した。主要5 familyは`M>=128`、Full Attention K/Vの
`K=2560,N=1024`は`M>=1024`で切り替え、`N=32`、未知shape、all-logits vocabulary shape、短Mは従来providerを維持する。
gfx1201/gfx942 selector、gfx1030 hipBLASLt、graph、public ABIは変更していない。

10,001行の248 projection加重値はtiled16 62.526秒からhipBLAS 11.081秒へ82.28%短縮した。final full-modelは
89.249秒から34.684秒へ61.14%短縮し、生成token `[2064, 5686]`、HIP-only、fallback false、cleanup 0、同一workspace arenaを
確認した。R9700は75.316秒でPhase 33 final 75.553秒と整合し、既存route不変だった。N=32のcrossoverは不安定かつ寄与が小さいため
棄却し、scope外へ利益を一般化していない。

hipBLAS solutionはGSU1でglobal split/atomic combineを使わず、両providerとも同じBF16入力、FP32 compute、BF16 RNE出力、同じK項を
一度ずつ使用する。stress入力ではbit差を観測したが、repeatは決定的でPhase 8標準bound違反は0だったためN1とした。

R9700 long controlで露呈したMTP verify row不具合も、MTP decode blockだけterminal全行を保持する限定修正として併合した。
通常prefill compactionは不変で、専用回帰テストと全`-p sllm-core -p sllm-hip`テストをPASSした。詳細結果とidentityは履歴および
bounded summaryを正とする。

final gfx1030 serverの10,001-token OpenAI non-stream/SSE、disconnect=`cancelled`、small recovery、graceful shutdownもPASSし、
HIP-only、fallback/cleanup 0を確認した。

[Phase 9履歴](../../../../../history/2026/08/11-20/phase9-engine-structural-optimization.md)
[Phase 23 bounded summary](../../../../../../ci/matrix/phase23-performance-discovery-summary-v1.json)
[Phase 24計画](../../../../archive/2026/08/11-20/phase24-prefill-terminal-row-projection-optimization.md)
[Phase 31計画](../../../../archive/2026/08/11-20/phase31-chunked-prefill-memory-foundation.md)
[Phase 33計画](../../../../archive/2026/08/11-20/phase33-full-attention-structural-optimization.md)
[runtime architecture](../../../../../architecture/runtime.md)
[数値・出力影響変更台帳](../../../../../compatibility/numerical-output-changes.md)
[provenance](../../../../../provenance/README.md)
[メイン計画](../../../../main-plan.md)
[実施履歴](../../../../../history/2026/08/11-20/phase34-v620-long-prefill-bf16-matmul-provider-optimization.md)
[bounded summary](../../../../../../ci/matrix/phase34-v620-prefill-matmul-summary-v1.json)

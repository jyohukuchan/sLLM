# Phase 9: 実行エンジン構造最適化

> 状態: completed
> 作成日: 2026-08-14
> 完了日: 2026-08-14

## 目的

Phase 8で大幅に改善したQwen3.5 BF16単一request pathを、model本体のFP8 W8A8を追加する前に
構造から最適化する。主対象は数値形式に依存しないdecode graph/segment実行、host completion境界、
M=1 matvec dispatch、Qwen3.5 GDNと周辺fusionである。BF16だけの一時的な高速化ではなく、Phase 10の
FP8とPhase 14のWeight NVFP4が同じ高速な実行骨格を利用できる状態を作る。

llama.cppと同じmodel revision、target、dtype、token条件で差を測り、既にAMD/HIPで成立している小さな
実装単位は積極的に直接reuseする。llama.cpp全体のgraph/runtimeは移植せず、sLLMのRust service、semantic
op、vAttention KV ownership、scheduler、versioned ABI、transactional state/error契約を維持する。

## 開発順序の変更

ユーザー指示により、本作業をPhase 8.5ではなく正式なPhase 9とする。旧Phase 9以降は次のように
一段繰り下げる。

| 新Phase | 内容 | 旧Phase |
| --- | --- | --- |
| 9 | 実行エンジン構造最適化 | 新規 |
| 10 | model本体FP8 W8A8 | 9 |
| 11 | FP8/BF16のCDNA3移植 | 10 |
| 12 | MI300X単体実機確認 | 11 |
| 13 | google/gemma-4-12B | 12 |
| 14 | Weight NVFP4 | 13 |
| 15 | KV cache FP8/NVFP4 | 14 |
| 16 | MTP、vision | 15 |
| 17 | Gemma4またはQwen3.5 MoE | 16 |
| 18 | 残りの初期version機能 | 17 |
| 19 | 人間によるREADME整備・発表 | 18 |

## 開始時点の事実

- Phase 8 short-odd 17/17のsLLM decodeはV620 `gfx1030`で約1.87 tok/s、R9700 `gfx1201`で
  約1.95 tok/sである。同じ固定llama.cppは約41.00/52.27 tok/s、E2E差は約20.4/26.9倍残る。
- short-oddは7,956 submission / 8,364 kernel、prefill-longは59,904 / 62,976、decode-longは
  119,808 / 125,952である。short-oddからoutput tokenあたり約468 submission / 492 kernelとなる。
- Phase 8のprepared semantic cacheはprepare重複を除いたがdispatch数は変えていない。単純な
  same-stream host wait batchingは改善せずdefaultへ採用しなかった。
- `qwen_execution.rs`はsemantic submission、KV append、full attention、linear attention等を個別にwaitし、
  position依存attention preprocessはdecode stepごとにdescriptorを準備する。
- Qwen3.5-4Bは32 layer中24 layerがlinear attention/GDN、8 layerがfull attentionである。Phase 8の
  focused attention時間だけでは約0.51秒/tokenのTPOTを説明できず、full attentionの追加最適化は
  最初の支配候補ではない。
- 固定llama.cpp `f5919bf458ef190468b5c329bb293f8a54a1e69c`にはHIP Graph capture/replay、
  graph property update、floating MMVF/GEMV、gate/bias/GLU fusion、Qwen GDN kernel、RMSNorm系fusionがある。
  llama.cppからの直接reuseはproject方針で許可されている。

数値と比較条件の正本は`ci/matrix/phase8-profile-summary-v1.json`、Phase 8の判断は
[Phase 8 history](../../../../../history/2026/08/11-20/phase8-bf16-optimization.md)とする。

## 優先順位

1. dtype・modelに共通するhost orchestrationとgraph/segment実行。
2. dtypeに共通するcompletion、parameter update、ownership境界。
3. BF16 M=1 GEMV/MMVFとshape/target dispatch。
4. 24 linear-attention layerへ効くQwen3.5 GDNとstate更新fusion。
5. 全32 layerへ効くMLP GLU、residual、RMSNorm fusion。
6. prefill provider/packingの実shape再評価。
7. profileが変化した場合だけfull attentionまたはRDNA4固有tuning。

RDNA4向けFA3-likeは既存の非blocking将来taskとして維持する。Phase 9のprofileでattentionが支配要因へ
変わらない限り、Phase 9へ取り込まない。

## スコープ

- Qwen3.5-4B BF16、single GPU、batch 1のprefill/decode production path。
- stable decode graphまたは安全に分割したsegmentのmodel-resident準備とrequest間再利用。
- HIP Graphのcapture、instantiate、update、replay可能性と、非対応部分の明示的segment境界。
- token position、KV長、pointer、scalar等の動的入力を更新するdevice parameter blockまたはgraph node update。
- M=1 BF16 GEMV/MMVF fast pathとshape/target別provider selection。
- Qwen3.5 GDN、MLP GLU、residual/RMSNorm等のprofile-driven fusion。
- real prefill shapeに対するhipBLAS/rocBLAS/hipBLASLt/既存custom provider比較。
- canonical V620/R9700でのdirect engine測定、固定llama.cpp比較、service smoke。

次はPhase 9に含めない。

- FP8、NVFP4、KV量子化、load-time quantization。
- multi-request/continuous batching、chunked prefill、multi-stream scheduling。
- Paged Attention、prefix sharing、RadixAttention、複数GPU。
- Qwen vision、MTP、MoE、他model architecture。
- 汎用graph optimizer、JIT compiler、永続autotuning DB。
- llama.cppのtensor graph、allocator、scheduler、service runtime全体の移植。
- RDNA4向けFA3-like。ただしprofile上の支配要因がattentionへ移った場合は次task候補を更新する。

## 外部実装の利用単位

### 直接reuse候補

固定llama.cppから次の単位だけをboundedに調査し、exact/adapted/portedの区分で採用する。

1. `ggml/src/ggml-cuda/common.cuh`と`ggml-cuda.cu`にあるHIP Graphのcompatibility判定、warmup、
   capture、executable update/replayの状態遷移。
2. `ggml/src/ggml-cuda/mmvf.cu`のfloating matvec kernel、gate/bias/GLU fusion、shape dispatch条件。
3. `ggml/src/ggml-cuda/gated_delta_net.cu`のwave reduction、state layout/access、Qwen GDN dispatch。
4. `ggml-cuda.cu`のRMSNorm+mul/add、RMSNorm+mul、SSM conv+SiLU等の小規模fusion dispatch。

移植時はupstream URL、完全commit SHA、upstream/local path、source blob/hash、license、変更内容、
import commitを記録し、source headerと`THIRD_PARTY_NOTICES.md`を更新する。draft中の調査はrelease用記録の
完成をblockerにしないが、release/distributionまでに解消する。

### no-copy参考

vLLM、SGLang、ATOM、CK/AITER等はarchitecture、dispatch predicate、測定方法の参考に限定し、code表現を
持ち込まない。llama.cpp由来でもCUDA portability macroや閾値を無条件採用せず、V620/R9700の実測で選ぶ。

## 設計境界

1. Rust側のsemantic graphは正しさ、audit、unsupported pathの基準として維持する。
2. native側へmodel-residentな`execution graph/segment plan`を追加し、Rustはeligibleな連続nodeを一回で
   submitする。public semantic descriptorを消さず、graph非対応nodeの境界を明示する。
3. graph planはtarget、dtype/encoding、model graph fingerprint、shape class、alignment、provider集合、
   workspace上限をkeyとする。request tokenやpositionそのものをcache keyへ含めない。
4. 動的値はcapture済みhost pointerの偶然の安定性へ依存せず、device parameter blockまたは検証済み
   node parameter updateで変更する。更新値とgenerationをaudit可能にする。
5. completion ownerは全input/output/state/workspace/queue/graph executableをterminalまで保持する。
   cancellation、timeout、errorでは未完了stateを公開せず、既存の先行stateを保持する。
6. HIP Graphが一部op/libraryで安定しない場合は黙って逐次実行へfallbackせず、そのnodeでsegmentを切る。
   capture/replay可否と選択理由をdispatch metadataへ残す。
7. baseline kernelと逐次semantic pathは数値oracle、debug、unsupported shape用に残す。production fast pathの
   実行時失敗をbaseline成功へ読み替えない。

## 受入条件

1. P9-A0で同一4B BF16 17/17とdecode surrogateを両GPUで測り、CPU submission/wait、GPU idle、
   kernel/provider、launch数、transfer、memory bandwidthをwall timeへ分解する。sLLMと固定llama.cppの
   trace条件を記録し、raw traceはlocal-only、bounded summaryだけを追跡する。
2. HIP Graph PoCは両canonical targetでkernel-onlyとhipBLAS混在segmentを確認し、warmup、capture、
   instantiate、replay、pointer/scalar更新、error/cancel/cleanupを区別する。timeout、crash、未実行、CPU
   fallbackをPASSにしない。
3. production eligible decode pathではsemantic opごとのhost waitを廃止し、token terminalまたは明示した
   graph segment境界へcompletionを集約する。readback、cancellation check、transactional state publish以外の
   host waitを残す場合は理由とcostを記録する。
4. graph/segment planとworkspaceをmodel-residentに再利用し、requestごとのgraph instantiate、全weight repack、
   library handle生成を行わない。動的KV長/position/pointer更新後も同じ実行planを再利用できる。
5. M=1 BF16 fast pathはQwen3.5の実shapeを含む独立float64 oracleでcurrent providerと比較し、非整列値、
   dispatch境界B-1/B/B+1、NaN/Inf classification、target mismatch、unsupported shapeを検査する。
6. GDN/fusionは未融合semantic pathとのdifferential、linear state/KV stateの世代・rollback、alias、workspace、
   cancel/dropを検査する。Qwen GDNを最優先とし、残りはP9-A3後のprofile上位だけを実装する。
7. optimized pathのexact target、HIP-only、no silent fallback、loader root、ECC/health、process/VRAM cleanup、
   fixed/Unicode/stop generationを確認する。
8. 通常iterationはO0/O1だけを使う。2B/9B、canonical long、固定llama.cpp、OpenAI non-stream/SSEは
   P9-A6またはmodel/semantic意味が変わった時だけ実行する。
9. case別のTTFT、prefill/decode token/s、TPOT、E2E、resident/peak/workspace VRAM、host wait、submission、
   kernel/graph node数、replay hit率、選択providerをPhase 8と固定llama.cppへ比較する。
10. 未承認の一律倍率やparityをhard gateにしない。各candidateはrepeated O1/O2で対象caseを改善し、他の
    canonical caseへ有意な退行を起こさない場合だけdefaultへ昇格する。単一最良runでは判断しない。
11. direct engineを性能の正本とし、serviceはtransport/cancellation/cleanupが構造変更後も成立することを
    確認するsmokeに限定する。
12. affected host/compile/GPU test、1回のintegration review、指摘箇所だけのfocused re-review、provenance、
    runtime/compatibility/main plan/history同期を完了し、完了時にこのplanをarchiveへ移す。

## 実装順序

### P9-A0: gap accountingとtrace固定

- Phase 8 summaryを基準に、4B short-odd 17/17と32/32 surrogateを両GPUで再取得する。
- host側はprepare、FFI、submission、event query/wait、readback、state publishを区間化する。
- GPU側はprovider/kernel別duration、launch間idle、transfer、wave occupancy、bandwidthを代表sampleだけ取得する。
- 固定llama.cppも同じmodel/token条件の代表traceを取得し、graph replay有無、launch数、上位kernelを比較する。
- wall timeを`host/launch + GEMV + GDN + full attention + prefill GEMM + transfer/other`へ分解し、A2〜A5の
  優先順位を更新する。計測自体をproduction requestの常時overheadにしない。

### P9-A1: llama.cpp bounded readerとHIP Graph PoC

- 固定llama.cppのgraph state machine、MMVF、GDN、fusionからreuse候補のsource range、依存、license、
  AMD portability前提を記録する。直接reuse候補とno-copy事実を混ぜない。
- standalone native PoCで、sLLM kernelだけのsegmentとhipBLAS GEMMExを含むsegmentを個別にcapture/replayする。
- tokenごとに変わるposition、KV length、input/output pointer、parameter generationを更新し、同じexecutableを
  再利用できるか確認する。
- stream capture support、graph instantiate/update cost、初回warmup回数、replay latency、node count、破棄後の
  VRAM/resource復帰をV620/R9700で測る。
- 結果から、全decode graph、layer segment、op-family segmentのどこをproduction境界にするか決定する。
  非対応nodeは明示的segment cutとし、graph全体のsilent disableにはしない。

### P9-A2: production graph/segment実行とcompletion集約

- versioned C ABIへopaque graph/segment plan、execute、completion、dispatch metadataを追加する。
- existing prepared semantic planとbuffer ownerを束ね、model load後にstable segmentを一度準備する。
- request-local device parameter blockを導入し、position、token/KV length、active pointers、state generationを
  一括更新する。parameter update完了とgraph launchを同一stream順序へ置く。
- `qwen_execution.rs`はeligible node列をsegment submitへ置換し、per-op `wait_submission_success()`を
  terminal/segment completionへ集約する。argmax/logits readback、cancel check、state publishは明示的境界に残す。
- submission/kernel数、host wait、graph capture/replay/update countをauditとprofileへ追加する。
- forced error、timeout、drop、disconnect、partial segment failureで未完了KV/linear stateをpublishせず、
  buffer/queue/graph executableが早期解放されないことをhost contractとreal GPUで確認する。

### P9-A3: BF16 decode GEMV/MMVF

- Qwen3.5のM=1 projection shapeごとにcurrent custom wave reduction、hipBLAS、llama.cpp由来MMVF candidateを
  microbenchmarkする。cold library initializationとsteady stateを分ける。
- `mmvf.cu`の必要最小kernel、type conversion、wave reduction、gate/bias/GLU optionだけをsLLM registryへ
  port/adaptし、ggml tensor/runtime依存を持ち込まない。
- gate/up projectionを同一launchで処理できる場合は、weight pointer/layoutをmodel-resident planへ固定し、
  SiLU multiplyまでのfusion有無をshape別candidateとして比較する。
- target、M/K/N、alignment、output width、gate/bias modeごとにdispatchし、V620とR9700で別の閾値を許す。
- 改善しないshapeはcurrent providerを維持し、candidateの存在だけでdefaultを置換しない。

### P9-A4: Qwen3.5 GDNと高頻度fusion

- 24 linear-attention layerのconv+SiLU、recurrent state update、normalization、z gate、state write/copyを
  P9-A0/A3後のprofileで分解する。
- llama.cpp GDNのwave-owned state shard、reduction、dispatchをsLLMのBF16 input/FP32 stateと既存transactionへ
  adaptする。state layout変更が有利な場合はrequest-local stateを一つのcanonical layoutへ移行し、恒久的な
  二重stateを作らない。
- 最初にrecurrent GDN fast path、次にconv+SiLU/state history、必要なら両者と周辺projectionのfusionを比較する。
- その後も上位costに残る場合だけ、MLP gate/up+SiLU multiply、residual+RMSNorm(+scale)、Q/K norm+RoPE+KV
  appendの順で実装する。候補を一律には実装しない。
- 各fusionは元semantic op列との数値differential、alias、NaN/Inf、state generation、cancel/rollbackを維持する。

### P9-A5: prefill provider再評価と残差最適化

- real Qwen shapeのM>1に対し、existing tiled16、hipBLAS/rocBLAS、hipBLASLt、利用可能なCK candidateを
  target別に比較する。library存在をsolution supportと読み替えない。
- model-resident packed weightが有利な場合はresident representationを置換する設計を優先し、元weightとの
  無条件二重保持を避ける。Phase 10のFP8 layoutにも同じplan/registry境界を再利用できるようにする。
- A2〜A4後のprofileで新たに支配的になったcostだけを追加最適化する。full attentionやRDNA4 FA3-likeが
  支配的でなければ、このwork unitでは実装しない。
- residual gapをhost/launch、GEMV、GDN、attention、prefill、memoryへ再分解し、Phase 10へ渡す
  dtype固有backlogと、別の共通最適化backlogを分離する。

### P9-A6: canonical統合、llama.cpp比較、計画同期

- canonical V620 `gfx1030` / R9700 `gfx1201`で4B O2を実行する。semantic/model意味が共通する変更は
  2B/9B short spot checkを追加する。
- fixed llama.cpp wrapperと同じmodel revision、commit、target、BF16、input/output token条件でTTFT、TPOT、
  token/s、E2E、peak VRAM、launch/graph構造を再比較する。
- OpenAI non-stream/SSE、stop、disconnect cancellationを同一production pathでsmokeし、server shutdown後の
  request/state/workspace/VRAM/process cleanupを確認する。
- Phase 8比、llama.cpp比、採用/rejectしたcandidate、graph replay coverage、残差backlogをversioned summaryへ残す。
- 累積integration reviewとaffected final gates後、runtime/compatibility/provenance/main plan/historyを同期し、
  このplanをarchiveへ移す。

## 計測lane

| lane | Phase 9での使用 |
| --- | --- |
| micro | 対象kernel/graph PoC、独立数値oracle、steady-state latency。full modelを起動しない |
| O0 | 変更対象GPU、4B short-odd、warmup 1 + measured 3。correctness、fallback、cleanup、方向性を確認 |
| O1 | O0 + 32/32 surrogate。通常iterationでsubmission、wait、graph hit、TPOTを比較 |
| O1-boundary | graph/dispatch/tiling/alignment/VMM境界を変えた時だけB-1/B/B+1を追加 |
| O2 | A2〜A5の統合とA6でcanonical 4B 7 case、warmup 3 + measured 10を実行 |
| O3 | release/nightlyまたは意味変更時だけdual-GPU、2B/9B、llama.cpp、serviceを広く実行 |

同一candidateのV620/R9700 runはGPUをUUIDで分離し、他processの干渉がない場合は並列実行できる。
rocprof traceはA0と支配要因が変わったcheckpointの代表sampleだけに限定し、各最適化の必須測定にしない。

## 性能判定

- Phase 8 profileを開始baselineとし、direct engineのmedian、p10/p90、MADを比較する。
- 数値の一律performance hard gateは置かない。llama.cpp parity未達だけで正しい構造改善をrejectしない。
- candidateをdefaultへ昇格するには、対象caseでrepeated O1/O2の改善があり、別canonical caseの有意な退行、
  silent fallback、過大なresident/peak VRAM増加がないことを確認する。
- Phase 9完了時にllama.cppとの差が残る場合は、kernel time、host wait/idle、launch、memory bandwidth、
  provider、graph coverageのどこに残るかを定量化する。Phase 10にはFP8固有最適化だけを混ぜ、未解決の
  dtype非依存問題は別backlogとして明示する。
- Phase 14 Weight NVFP4開始前に最新profileで支配要因を再確認する。これはPhase 9中の広範再測定を
  増やす条件ではなく、古いbottleneck判断のまま新形式を増やさないための短いcheckpointである。

## Rollbackと再計画

- graph/segment、provider、kernel、fusionはregistry/dispatch entry単位で無効化できるようにし、逐次semantic
  baselineを削除しない。
- numerical mismatch、silent fallback、state publication破壊、use-after-free、cleanup不良は該当work unitの
  correctness blockerとする。性能だけが改善しないcandidateはdefaultへ採用せず、他work unitをblockしない。
- HIP Graphが両GPUの主要segmentで成立しない場合は、同じ同期削減目的をnative command-list/segment submitで
  実現するようA2を再計画する。PoC失敗をCPU fallbackや無検証graph claimで埋めない。
- 同じwork unitの2回reject、review時間が実装時間超過、1時間以上の機能進捗停止、検証・文書が30%超、
  見積り1.5倍超、gate/受入条件変更のいずれかで追加review・測定を止め、ユーザーへ報告して再計画する。

[対応するhistory](../../../../../history/2026/08/11-20/phase9-engine-structural-optimization.md)

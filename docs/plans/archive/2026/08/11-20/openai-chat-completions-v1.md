# Phase 6: KV memory方式選定とOpenAI-compatible Chat Completions profile v1実装計画

## 目的

Phase 6の最優先課題として、AMD GPU上でvAttention型の仮想連続KV cacheを再現できるかを
standalone HIP PoCで確認する。続いてKV memory方式とattention kernelを直交する軸として扱い、
vAttention上の連続KV FlashAttention系kernelと、block tableを読むPaged Attention系kernelを比較して、
Paged AttentionとvAttentionのどちらを初期KV memory方式にするか決定する。
選択した方式を上位serviceから隠すKV allocation/layout境界を先に確定したうえで、
完成済みQwen3.5 text generation pathをmodel-resident serviceとして再利用し、
[`sLLM OpenAI-compatible Chat Completions profile v1`](../../../../../api/openai-compatibility.md)の
`GET /v1/models`、`POST /v1/chat/completions`、non-stream、SSE stream、strict error behaviorを
Rust serverとして実装する。

HTTP外形だけでなく、profile v1が列挙するtemperature、top_p、presence/frequency penalty、stop文字列、
usage、disconnect cancellationを実行可能にする。llama.cppから再利用価値の高いsamplingとtest資産を
provenance付きでport/adaptし、vLLM、SGLang、TensorRT-LLM、LMDeployはno-copy readerとして
service境界、streaming、usage、cancellationの技術的要点だけを参照する。

vAttentionの公開実装はCUDA/A100を対象とする研究実装であり、AMDでの再現性を証明しない。
Microsoft vAttention、vLLM等の非llama sourceはno-copyの技術参考に限定し、PoCはAMD HIP VMMの
公式APIと論文の公開事実から独立して実装する。

FlashAttention-2のAMD実装候補はROCmのComposable Kernel/CK Tileとし、PyTorch/Tritonをproduction
dependencyへ追加しない。upstream FlashAttention-3はHopper向け、FlashAttention-4はCuTe DSLで
Hopper/Blackwell向けであり、V620/R9700上で同一実装を実測したとは主張しない。A1ではFA2相当のAMD実測と、
FA2/3/4の公開algorithm・KV interfaceから行う設計比較を分離する。

## 仕様正本と現在の公式仕様

- payload互換性の正本は現在のprofile v1と、そこから固定されるOpenAI OpenAPI commit
  `117ce5680e4269f6656a4fd70d28f9755630d938`である。
- 2026-08-11時点の公式OpenAI OpenAPIは`POST /v1/chat/completions`を引き続き提供し、JSON responseと
  `text/event-stream`を定義する一方、新規projectにはResponses APIを推奨している。
- current specとの差分はreader記録とnegative test候補へ反映するが、profile v1のpinを実装途中で
  silently更新しない。pin更新またはResponses API追加は別versioned decisionとする。

## 前提と依存関係

- [Phase 4 Qwen3.5-2B・9B互換性確認計画](../../../../archive/2026/08/11-20/qwen35-2b-9b-compatibility.md)と
  [Phase 5エンジン性能baseline計画](../../../../archive/2026/08/11-20/engine-performance-baseline.md)の完了後に開始する。
- Phase 3のmodel lock、typed renderer/tokenizer、Qwen execution、dispatch audit、stop token policyを再利用する。
- initial serviceは単一GPU、単一model-resident instance、1 active generationとbounded queueに限定する。
- Rust MSRV 1.85.0、offline dependency closure、network-isolated H0〜H2を維持する。
- ROCm 7.14.0の直接queryではcanonical V620 `gfx1030`とR9700 `gfx1201`の
  `hipDeviceAttributeVirtualMemoryManagementSupported`がともに1だった。ただし、これはreserve/map/unmapの
  正しさ、粒度、latency、既存kernelとの統合を証明しないため、A0の実機PoCを方式選定の根拠とする。
- A0はPyTorch、full model、model weight、HTTP serverを使用しない。C++/HIPのmodel-free executableと
  小さな数値oracleでVMM自体を先に検証し、選択後にだけproduction KV pathとfull modelへ進む。
- local ROCm 7.14.0には`amdrocm-ck7.14`とCK Tileのcontiguous/paged-KV FMHA headerが存在するが、
  Qwen3.5の`q_heads=16`、`kv_heads=4`、`head_dim=256`とexact `gfx1030`/`gfx1201`で利用可能な
  instance、数値正しさ、性能は未確認である。A1ではcompile/dispatch feasibilityからfail-closedに確認する。

## server構成

### crateとdependency固定

- `crates/sllm-server`: OpenAI DTO、strict validation、model registry、request queue、HTTP/SSE adapter、binary。
- `sllm-core`/`sllm-frontend`: CLIから抽出するtransport非依存generation service、sampling、stop、usage。
- A2で固定したdirect dependency（2026-08-13）:
  - `axum = 0.8.9`（MSRV 1.80、MIT）
  - `tokio = 1.53.1`（MSRV 1.71、MIT）
  - `tower-http = 0.7.0`（MSRV 1.65、MIT）
  - `tokio-stream = 0.1.19`（MSRV 1.71、MIT）
  - `futures-util = 0.3.33`（MSRV 1.71、MIT OR Apache-2.0）
  - `serde_path_to_error = 0.1.20`（MSRV 1.61、MIT OR Apache-2.0）
- `crates/sllm-server`をworkspaceへ追加し、上記direct dependencyをexact versionでroot `Cargo.lock`と
  `ci/dependencies/rust-workspace-v1.json`へ固定した。closureは132 package、308 edgeで、全registry packageの
  checksum/license/MSRV、全edge、resolved/requested featureをoffline validatorで再現する。axumは
  `http1,json,tokio`、tokioは`macros,net,rt-multi-thread,signal,sync,time`、tower-httpは`limit,trace`、
  tokio-streamは`sync`、futures-utilは`std`だけをdirect edgeから要求する。

### runtime境界

1. model registryはserved aliasをimmutable model-lock fingerprintとmodel-resident ownerへ結合する。
2. requestごとにrenderer/tokenizer、sampling state、stop matcher、Qwen request-local stateを新規作成する。
3. model load/uploadはrequest間で再利用し、request failure/disconnectでmodel-resident ownerを破棄しない。
4. initial schedulerは1 active requestを実行し、bounded FIFOを超えたrequestを429で拒否する。
5. cancellationはHTTP disconnect、server shutdown、timeoutからgeneration ownerへ伝播し、KV/linear state、
   pending completion、channelをcleanupする。

## 既存engineの利用方針

### llama.cpp: direct reuse候補

固定commit `f5919bf458ef190468b5c329bb293f8a54a1e69c`から、次を実装開始前にexact blob/license単位で確定する。

| upstream path | 用途 | reuse分類候補 |
| --- | --- | --- |
| `src/llama-sampler.cpp`, `src/llama-sampler.h` | temperature、top-p、penalty、sampling order | Rustへの`ported` |
| `tests/test-sampling.cpp` | tiny logits、boundary、deterministic RNG test | Rust testへの`adapted` |
| `tools/server/tests/unit/test_chat_completion.py` | request/response、usage、finish reason | profile v1 subsetだけ`adapted` |
| `tools/server/tests/unit/test_stream.py` | chunk順、SSE終端、disconnect | profile v1 subsetだけ`adapted` |
| `tools/server/tests/unit/test_security.py` | body/path/header negative cases | 該当caseだけ`adapted` |
| `tools/server/tests/utils.py` | HTTP/SSE test harness | 最小部分を`adapted`またはclean実装 |

`server-http.cpp`、`server-stream.cpp`、`server-context.cpp`、`server-task.cpp`はC++ server architectureと
Rust/axumの差が大きいため、原則`facts-only`とする。protectable structureを近接移植した場合は
`ported`へ分類を上げる。

最初のdirect import時に`THIRD_PARTY_NOTICES.md`を作成し、各local source header、upstream URL、
完全SHA、source blob、local path、imported SHA-256、copyright、MIT license、reuse mode、変更内容、
import commitを[provenance正本](../../../../../provenance/README.md)どおり記録する。

### 他engine: facts-only reader

- Microsoft vAttention:
  論文`arXiv:2405.04437`と公開repository。A0開始前に参照revisionを完全commit SHAで固定し、
  CUDA/A100で観測された設計事実だけを使う。source code、test body、CUDA固有実装はcopy/adapt/portしない。
- Dao-AILab FlashAttention、ROCm Composable Kernel/CK Tile、ROCm AITER:
  contiguous/paged KV interface、対応architecture、公開algorithm、kernel capabilityのfacts-only readerとする。
  upstream CUDA/CuTe DSL、Triton、CK/AITER source codeやtest bodyをcopy/adapt/portしない。ROCm 7.14.0に同梱された
  MIT-licensed CK headerをproductionへ直接取り込む場合は、別途dependency/provenance判断を行う。
- vLLM:
  `vllm/entrypoints/openai/chat_completion/{protocol.py,serving.py,api_router.py}`、
  `models/{protocol.py,serving.py}`、`engine/protocol.py`。A1のattention比較では追加で
  `vllm/v1/attention/backends/{flash_attn.py,fa_utils.py}`とROCm attention selector/testの
  interface・capability事実だけを参照する。
- SGLang:
  `python/sglang/srt/entrypoints/openai/{protocol.py,serving_chat.py,sse_utils.py,usage_processor.py}`と
  registered OpenAI server tests。
- TensorRT-LLM:
  `tensorrt_llm/serve/{openai_protocol.py,openai_server.py,openai_service.py}`とOpenAI app tests。
- LMDeploy:
  `lmdeploy/serve/openai/{protocol.py,api_server.py,serving_chat_completion.py,utils.py}`。

Microsoft vAttentionからはvirtual address reservation、physical commit、page activation、mapping lifetimeの
技術的事実だけを抽出する。他engineからはvalidation layering、async generator、disconnect propagation、
usage accounting、stream/non-stream共通化、backpressure、error分類の技術的事実だけをreader記録へ抽出する。
いずれもsource code、test body、型定義をcopy/adapt/portしない。

## 作業順序と依存関係

1. A0のAMD vAttention再現性PoCは2026-08-13に完了した。
2. A1のFA2相当AMD実測、FA2/3/4設計比較、初期方式選択、KV memory契約、最小production pathは
   2026-08-13に完了した。初期方式はvirtual-contiguous KV（vAttention型）である。
3. A2のAPI profile drift、reader、dependency、provenance設計は2026-08-13に完了した。
4. A3 generation serviceは2026-08-14に完了し、A1のKV lease/view契約だけに依存してVMM pointerや
   block tableを直接扱わない境界を維持した。
5. A4とA5は2026-08-14に一つの実装バッチとして完了し、共通DTO、generation event、scheduler、
   error mappingを一度だけ確定した。受入結果はA4/A5ごとに判定した。
6. A6は2026-08-14に独立したintegration batchとして完了し、compatibility fixture、
   differential、canonical GPU、service overheadのevidenceを取得した。

A0/A1は、後続のcancellation、capacity/admission、continuous batching、KV量子化でKV所有権と
addressingを作り直さないためPhase 6へ前倒しした。full dynamic/continuous batching自体はPhase 6の非対象を維持する。

### A4〜A6のバッチ境界（2026-08-14決定）

- A4とA5は一括実装する。non-streamとstreamを同じtyped request、model registry、bounded scheduler、
  generation event、usage/finish resultから構築し、A4だけの一時的なunbounded・non-cancellable境界を作らない。
- A4とA5の受入条件は統合しない。strict HTTP/non-stream matrixとSSE/cancellation/backpressure matrixを
  個別に満たしたことを確認し、未達項目を一括完了へ埋没させない。
- A6は分離する。外部engine differential、official client、full-model GPU、VMM cleanup、service overheadは
  host API実装とは失敗原因、実行時間、evidence identityが異なるため、A4/A5の実装反復へ毎回混ぜない。
- A0〜A3が想定より早く完了した事実はA4/A5の共有境界を早期に統合する根拠にはなるが、
  canonical GPUと外部互換性を含むA6まで一つの不可分な作業単位にする根拠にはしない。

## 作業単位

### A0: AMD vAttention再現性確認とstandalone HIP PoC（2026-08-13完了）

実績:

- standalone source `ci/tools/vattention_a0_probe.hip.cpp`、fail-closed runner
  `ci/tools/run_vattention_a0.py`、offline host contractを実装した。
- canonical V620 `gfx1030`とR9700 `gfx1201`の両方でVMM reserve/create/map/access、contiguous kernel、
  middle-page unmap/remap、event完了後cleanup、CPU byte oracleをPASSした。fallbackは使用していない。
- 両targetともminimum granularityは4 KiB、recommended granularityは2 MiBだった。Qwen3.5-4Bの
  full-attention K/V 16 region、4096 token capacityでは128 MiBのVAをreserveし、初回1024 token分として
  2 MiB x 16 = 32 MiBだけをphysical commitした。VA reserveのphysical deltaは0 byteだった。
- 5 warmup + 101 measured iterationにおける16 region一括activation p50/p95は、V620が
  508.199/582.841 us、R9700が452.418/496.488 usだった。deactivation p50/p95はV620が
  474.338/546.910 us、R9700が769.224/846.735 usだった。create、map、set-access、unmap、releaseも
  個別にp50/p95を記録した。
- token境界1023/1024/1025と非整列37、pre/post ECC 0、process残留なし、physical memory復元を確認した。
  tracked binary/raw reportは作らず、local aggregate SHA-256
  `1cf7a93b6c3ca4cba976bf5eb08be372cfde50d4327b4e4d17946405fb345256`だけを履歴へ記録した。
- この結果はAMD上でvAttention型VMM primitiveを再現できることを示すが、production採用やdecode支配性を
  単独で決定しない。後続A1でPaged Attentionとの比較、commit償却方針、KV ownership contractを確定した。

1. canonical V620 `gfx1030`とR9700 `gfx1201`について、exact GPU identity、ROCm/runtime、kernel、
   VMM capability、`hipMemGetAllocationGranularity`のminimum/recommended granularityを記録する。
2. 最大KV capacity相当の連続virtual address rangeを先にreserveし、小さなphysical allocationを
   `hipMemCreate`、`hipMemMap`、`hipMemSetAccess`で必要時だけcommitするmodel-free executableを作る。
   予約rangeの後方拡張は保証されないため、PoCでは最大logical capacityを最初にreserveする。
3. pageを跨ぐwrite/read、unmap/remap、physical handle解放、reserved address解放を行い、CPU oracleと
   byte/numerical exactで比較する。CPU fallback、通常の`hipMalloc`へのsilent fallback、0件実行をPASSにしない。
4. token相当のlogical offsetがpage境界`B-1/B/B+1`と非整列位置を跨ぐcaseを作り、既存kernelが
   contiguous pointerのままmapped pageを読み書きできることを確認する。
5. primitive 1 regionに加え、固定Qwen3.5-4B configからKV region数と1 token当たりbytesだけを導出した
   model-free shapeでpage activationを行う。map/set-access/unmapのp50/p95、page activationを含むstep latency、
   reserved virtual bytes、committed physical bytes、process/device health、全resource解放を測る。
   長いPhase 5 model benchmarkは実行しない。
6. in-flight HIP work完了前にはunmapしないevent/lifetime規則を小さなasync caseで検証する。

受入条件:

- canonical各GPUでreserve/create/map/access/read-write/unmap/releaseがfail-closedに成功し、exact targetと
  numerical oracleが記録される。片方だけ成功した場合はtarget限定の事実として扱い、全AMD対応を主張しない。
- committed physical bytesがlogical capacity全体の事前確保ではなく、mapped page数に応じて増減する。
- `B-1/B/B+1`、非整列offset、再map、cleanup、in-flight lifetimeのcaseがあり、timeout/crash/fallbackを
  PASSにしない。
- latencyとgranularityを方式選定に使える単位で記録し、機能PASSだけから性能上の採用を決めない。
- PoC source、runner、小さなsummaryだけを追跡し、binary、raw trace/profile、生成artifactを追跡しない。

### A1: Paged Attention / vAttention選択とKV memory契約（2026-08-13完了）

実績:

- [KV memory方式のdecision record](../../../../../architecture/kv-memory.md)で、初期方式をcanonical
  V620 `gfx1030` / R9700 `gfx1201`限定のHIP VMM virtual-contiguous KV（vAttention型）に決定した。
  vAttentionはmemory management、FlashAttentionはkernelであり排他的でない。contiguous pointerを受ける
  FlashAttention系kernelはvAttention上で同じaddressingのまま利用する。
- local CK 7.14.0-3のcontiguous/paged-KV headerは確認できたが、exact shape/targetを選択する安定した
  prebuilt/generated dispatch経路を確認できなかった。このため比較値はFlashAttention-2またはCKそのものではなく、
  accessor以外を同じにした独立HIP `FA2-style proxy`として記録した。
- Q length 1/37、KV length 255/256/257/1023/1024/1025の36 mode-caseを両targetで実行し、NumPy
  float64 oracle、mode間数値一致、fallbackなし、pre/post health、cleanupをPASSした。Q=37/KV=1025の
  vAttention p50はpaged proxyよりV620で約17.0%、R9700で約31.3%短く、通常contiguous allocationとは
  概ね同等だった。厳密なupstream FA性能または通常allocation比の高速化は主張しない。
- FlashAttention-3/4はHopperまたはHopper/Blackwell向けの公開実装であり、AMD実測へ換算せずdesign comparison
  だけを記録した。将来AMD向けkernelがcontiguous pointerを受ければvAttention memory contractを再利用できる。
- public C ABI KV create/viewをversion 2へ更新し、token-major `[capacity, 4, 256]`、virtual-contiguous
  memory kind、physical page、mapped capacity、committed bytesをversioned metadataにした。上位Rust APIは
  opaque `KvState`/resourceだけを持ち、pointer、VMM handle、block table、backend page sizeを公開しない。
- actual public runtimeへreserve-only create、append前grow、event lifetime、unmap/releaseを接続した。
  host contractと両exact GPUのproduction probeで1023/1024/1025 token、2/2/4 MiB per-plane commitment、
  全要素BF16→FP16 oracle、未map readback拒否、idempotent cancel/cleanupをPASSした。
- comparison/productionを統合したlocal aggregate SHA-256は
  `453756b16f55ef81ff28dcb48cdebe69b9bdd83381b3a04202f94855af236021`である。
- Paged Attention production backend、prefix sharing、RadixAttention、continuous batching、KV量子化は実装していない。
  decision recordの再検討条件が成立した場合だけ同一contractでPaged Attentionを再比較する。

1. A0結果を、機能再現性、target間一貫性、allocation granularity、page activationの償却可能性、
   既存attention kernel変更量、将来のprefix sharing/block reuseの観点でdecision recordへまとめる。
   FlashAttention、vAttention、ROCm CK/AITER、pinned vLLM readerについて、比較に使ったrelease/完全commit、
   local package version、観測日を先に固定する。
2. `q_heads=16`、`kv_heads=4`、`head_dim=256`、現在のQ/K/V dtype/layoutについて、ROCm 7.14.0の
   CK/CK Tile contiguous FMHAとpaged-KV FMHAがexact `gfx1030`/`gfx1201`でcompile・dispatchできるか確認する。
   unsupported shape/target、0件選択、fallbackはbenchmark PASSにせず、利用可能範囲をtarget別に記録する。
3. CK/CK Tileで両layoutを比較できるtargetでは同一入力・出力contractで実測する。exact shapeの片方が
   利用できないtargetでは、同一のtiled online-softmax HIP proxyをcontiguous pointer accessorと
   block-table accessorだけ差し替えて実行し、これは`FA2-style proxy`であってFlashAttention-2実装の
   性能証拠ではないと明記する。current baselineは数値oracleとsanity timingにだけ使い、baseline改善を
   方式選定の主判定にしない。
4. model-free比較はprefillとdecodeを分け、少なくともquery length 1/37、KV length
   255/256/257、1023/1024/1025を含める。kernel p50/p95、KV append/grow、page/block metadata、
   committed/peak VRAMを記録する。contiguous通常allocation、vAttention mapped allocation、paged-KVの
   3経路を可能な範囲で同じkernel familyへ揃え、VMM mapping latencyはkernel latencyと分離する。
5. NumPy oracleに対する誤差、causal mask、GQA head mapping、page/block境界、非整列長を両canonical GPUで
   fail-closedに確認する。性能比較が厳密でなくても、異なる数値contractや失敗runを速度値として混ぜない。
6. FlashAttention-2/3/4について、連続KVとpaged KVの公開interface有無、kernel変更量、対象hardware、
   paging metadata cost、vAttentionでの無変更再利用可能性を表にする。FA3/4はV620/R9700で実測不能な
   NVIDIA-specific implementationとして`design/literature comparison`に限定し、AMD測定値へ換算しない。
7. 次をすべて満たす場合はvAttention型を初期方式に選ぶ。
   - Phase 6のcanonical AMD targetで必要なHIP VMM操作と数値caseがPASSする。
   - 最大logical capacityの事前VA予約と、必要時だけのphysical commitでrequest lifecycleを表現できる。
   - page activation latencyを複数tokenへ償却でき、測定上、通常decodeを支配する同期点にならない。
   - FlashAttention系のcontiguous kernelへdevice pointerを渡したまま、event完了までmapping lifetimeを
     保持でき、paged版に対する明白なcorrectness/portability上の劣位がない。
8. 必須VMM操作、lifetime、target一貫性のいずれかが成立しない、またはpage activationがdecodeを支配して
   実用的に償却できない場合はPaged Attentionを初期方式に選ぶ。PoC不成立を無理にvAttention採用へ読み替えない。
9. Rust scheduler/serviceにはlogical token rangeとopaqueな`KvAllocation`/lease/viewだけを公開し、
   pointer arithmetic、VMM handle、block table、backend page sizeを公開しない。
10. C ABI/backend descriptorは、vAttentionのvirtual contiguous viewまたはPaged Attentionのblock tableを
   versioned layout/capabilityとして表現できるようにし、allocate/grow/release/cancelを一つの所有権規則へ揃える。
11. 選択方式の最小production pathをrequest-local KVへ接続する。比較用proxyや非選択方式は
    production実装済みと主張せず、decision recordに再検討条件だけを残す。

受入条件:

- AMD実測、proxy、公開資料だけの比較を区別したFA2/3/4 comparison matrixを作り、同一数値contractで
  実行できた経路だけにp50/p95を載せる。少なくとも一つのcanonical targetでcontiguous/pagedの実測比較を得る。
- A0と上記比較の明示criteriaから一つの初期方式を選び、target限定、既知の制約、再検討条件を記録する。
- scheduler、generation service、HTTP層がcontiguous pointerまたはblock tableの内部表現に依存しない。
- allocate/grow/release/cancelがidempotent cleanupを持ち、capacity/page境界の`B-1/B/B+1`をhost contractと
  focused GPU numerical testで確認する。
- vAttention選択時は未map領域へのaccessを成功扱いにせず、Paged Attention選択時はmissing/stale block entryを
  成功扱いにしない。
- full prefix cache、RadixAttention、KV量子化、continuous batchingをA1へ混在させない。

### A2: profile drift、reader、dependency、provenance設計（2026-08-13完了）

結果:

- normative OpenAPI pinとcurrent commit `11854aef674352d3f9cd5c0a7038f079a7bbac06`を比較し、対象operationの
  request/response/stream/error shapeに差がないことを確認した。唯一のreachable差分は
  `ModelIdsShared` enumへの`gpt-5.5`追加で、profile v1 pinは変更していない。
- llama.cpp commit `f5919bf458ef190468b5c329bb293f8a54a1e69c`の7 unitをexact blob/SHA-256/licenseで固定した。
  必要なsampler部分と選択testだけをported/adapted候補とし、HTTP harnessはfacts-onlyとした。
- vLLM、SGLang、TensorRT-LLM、LMDeployのexact commit/pathをfacts-only readerへ固定し、8個のtechnical
  acceptance caseだけをA3〜A5へ渡した。
- `sllm-server` dependency closureを132 package、308 edgeとして固定し、Phase 6 A2 contract、schema、
  fail-closed validator、negative testをH0へ登録した。

1. pinned profileとcurrent official OpenAPIのrequest/response/stream/error差分表を作る。
2. llama.cpp direct reuse unitのexact blob/license/copyrightを確認し、import manifest案を作る。
3. 他engineのfacts-only reader結果を`docs/references/`へ記録し、implementationへ渡すのは
   technical factsとacceptance casesだけにする。
4. HTTP runtime dependencyのexact closure、feature、MSRV、license、offline cacheを固定する。

受入条件:

- normative pin、current drift、supported subset、rejected subsetが混同されない。
- direct reuseとfacts-onlyのpath、reuse mode、release時provenance作業が決まっている。
- dependency validatorが全package/edge/checksum/license/MSRVをofflineで再現する。

上記3条件を満たした。詳細は
[`OpenAI drift`](../../../../../references/openai-chat-completions-v1-drift.md)、
[`llama.cpp import計画`](../../../../../provenance/phase6-a2-llama-import-plan.md)、
[`serving reader`](../../../../../references/phase6-openai-serving-reader.md)を正とする。

### A3: transport非依存generation serviceとsampling（2026-08-14完了）

結果:

- CLIのrender/tokenize/prefill/decode/stop loopを`GenerationServiceV1`へ抽出し、request-localな
  `QwenExecutionRequest`とopaque KV ownerをtransportから分離した。CLIはこの共通serviceを使用する。
- samplingが必要な場合だけterminal BF16 logits rowをbackendのtransfer上限で分割してreadbackし、
  temperature 0では既存device argmaxをそのまま使ってlogits readbackとRNGを行わない。
- llama.cpp固定commitからtemperature、top-p、presence/frequency penaltyだけをRustへportedし、tiny-logit
  testをadaptedした。source、test、license、hash、変更内容はrepository-level noticeへ記録した。
- stop文字列をincremental UTF-8 matcherへ接続し、token/UTF-8境界を跨ぐpartial matchを保留してstop自身を
  visible textへ出さない。usage、`stop`/`length`、generated/visible token差を共通reportへ持たせた。
- cancellationまたはexecution/sampling/decode errorでrequest ownerだけを取消不能状態にし、model-resident
  ownerは維持する。public seedは追加せず、testだけが明示RNG seamを注入する。
- host unit/integration、独立NumPy fixture oracle、provenance/dependency validatorをPASSした。canonical V620
  `gfx1030`のQwen3.5-4B 1-token smokeではgreedyとtemperature 1の両方がtoken 11、HIP dispatchのみ、
  fallbackなし、usage 1/1/2、process残留なしでPASSした。これはA3 draft smokeであり性能値ではない。

1. [`crates/sllm-cli/src/model.rs`](../../../../../../crates/sllm-cli/src/model.rs)のgeneration loopを
   transport非依存serviceへ抽出し、CLIとserverが同じrender/tokenize/prefill/decode/stop pathを使う。
   request stateはA1のKV lease/viewを所有し、選択方式の内部handleを公開しない。
2. `QwenExecutionOutput`から全vocabulary logitsをbounded readbackできる経路、または等価なsampler input
   interfaceを追加する。greedy時は既存argmax pathとbit/token一致を維持する。
3. llama.cpp samplerのprofile v1に必要なtemperature、top-p、presence/frequency penalty部分だけを
   Rustへportし、不要なsamplerを持ち込まない。
4. public `seed`はprofile v1どおりrejectする一方、host test用には明示RNG injectionを内部APIだけに持つ。
5. `stop`文字列をincremental UTF-8 outputへ適用し、token境界を跨ぐpartial matchをstreamへ早出ししない。
6. prompt/completion/total token usage、finish reason `stop`/`length`、visible/generated token差を共通reportへ持つ。

受入条件:

- tiny logits oracleでtemperature、top-p、penalty、NaN/Inf、tie、0/1境界、非整列vocabを検証する。
- `temperature=0`はPhase 3 CLI/G3と同じtoken列を生成する。
- stop文字列がtoken境界・UTF-8境界を跨いでも漏れず、stop文字列自身をvisible outputへ含めない。
- cancellation/error後にrequest stateが再利用されず、model ownerは健全なまま残る。

上記4条件を満たしてA3を完了した。後続のA4/A5も下記の一括実装方針で完了している。

### A4: model registry、strict JSON、non-stream endpoint（2026-08-14完了）

実績:

- `sllm-server`へlock fingerprint付きmodel registry、duplicate memberも拒否するstrict DTO、field path付き
  validation、optional bearer auth、1 MiB body/Content-Type検証、標準error envelopeを実装した。
- `GET /v1/models`とnon-stream `POST /v1/chat/completions`を共通scheduler/backend eventの上に実装し、
  unsupported/unknown field、role/content、range、unknown model、queue fullをprofile status/codeへ写像した。
- OpenAI Python SDK 3.0.0を`base_url=http://127.0.0.1:18080/v1`でfixture serverへ接続し、
  `models.list`とnon-stream Chat Completions、model/text/finish/usageのparseをPASSした。

1. `GET /v1/models`をalias、`object=model`、created、owned_byから構築する。
2. profile v1のrequest DTOを`deny_unknown_fields`相当で実装し、field path付きvalidation errorを返す。
3. model alias、messages、role/content、n=1、generation range、stop、body size、Content-Typeを検証する。
4. non-stream response ID、created、model、choice、assistant message、finish_reason、usageを生成する。
5. malformed JSON、unsupported parameter、unknown model、body too large、optional bearer auth、queue fullを
   profile v1 error envelope/statusへ写像する。

受入条件:

- supported field全組合せと、tools、developer/tool role、multipart、response_format、logprobs、seed、
  n≠1、unknown fieldを含むnegative matrixがある。
- `GET /v1/models`のaliasが実際のlock fingerprintへ結合され、floating revisionを返さない。
- OpenAI Python clientを`base_url`指定で使うmodels/non-stream smokeがPASSする。

### A5: SSE、backpressure、disconnect cancellation（2026-08-14完了）

実績:

- 1 active + bounded FIFOとrequestごとのbounded generation event channelを実装した。consumer遅延は
  synchronous backend sinkへbackpressureし、queue fullは429、timeout/shutdown/disconnectは共通cancellationへ伝播する。
- A3 generation serviceへvisible output sinkを追加し、incremental UTF-8/stop matcherが安全と判定したnonempty deltaだけを
  transportへ公開する。CLI/non-streamは同じloopをno-op sinkで再利用する。
- SSEはassistant role、content delta、terminal finish/usage、exact `[DONE]`の順とした。header後のfailureは
  error envelopeを一つ送信して`[DONE]`なしでcloseするsLLM conventionとしてprofileへ記録した。
- raw TCP contractでstream/non-stream一致、slow consumer、empty delta、先頭stop、length、Unicode、mid-stream failure、
  disconnect、queue full、backend failure後のowner健全性を確認した。Python SDK 3.0.0のstream parseもPASSした。

1. generation eventをbounded channelでHTTP taskへ渡し、consumer遅延でunbounded memoryを使わない。
2. first chunkのassistant role、content delta、final finish_reason chunk、exact `[DONE]`をprofile順で送る。
3. header前errorはJSON envelope、header後errorはdocumented mid-stream terminal behaviorへ分離する。
4. client disconnect、server shutdown、request timeoutをgeneration cancellationへ伝播する。
5. 1 active + bounded FIFOをstressし、queue fullは429、active model owner failureは後続requestへ伝染させない。

受入条件:

- SSE bytesが`data: <JSON>\n\n`と最後の`data: [DONE]\n\n`に一致する。
- stream/non-streamのvisible text、token usage、finish reasonが同じ固定requestで一致する。
- disconnect後にgeneration backendが継続せず、request state、bounded channel、HTTP/worker taskがcleanupされる。
  pending HIP work、VRAM、GPU processまでのfull-model cleanup evidenceはA6 item 5/6で取得する。
- slow consumer、empty delta、stop at first token、max token、UTF-8分割、mid-stream failureをtestする。

### A6: compatibility、differential、GPU integration（2026-08-14完了）

実績:

- pinned OpenAPI profile subset fixture、schema/validator、raw HTTP/SSE testをH0/H1へ追加し、llama.cppの
  Chat Completions/stream/security testからprofile該当caseだけをprovenance付きでadaptした。
- exact pinned vLLM/SGLang serializerと実稼働llama.cpp HTTP/SSEを共通subsetで比較し、3 peerを仕様oracleに
  しないdifferentialとしてPASSした。llama.cpp独自`system_fingerprint`/`timings`はextraとして分離した。
- `QwenChatBackendV1`とproduction `sllm-server`を追加し、verified lock/cache、resident model、既存
  `GenerationServiceV1`、strict registry/schedulerへ接続した。profile v1はPhase 5と同じthinking-disabled
  render契約を使う。serverから`QwenResidentModel`までのworkspace closureは132 package、309 edgeである。
- Qwen3.5-4B、temperature 0のraw non-stream/SSE、stop、OpenAI Python client 2.44.0、HIP dispatch後disconnect、
  recoveryをUUID単独可視化したcanonical V620 `gfx1030`とR9700 `gfx1201`でPASSした。completed requestは
  exact HIP dispatchのみ、fallbackなしで、shutdown後tracked allocationとGPU processは0だった。
- logical capacity 1023/1024/1025、2 MiB physical page、8 full-attention layerのK/V committed合計32 MiB、
  24 linear-attention layerをfull-model service auditとして記録した。disconnectは936 submission後にcancelされ、
  request-local KV/linear/workspaceを0へ解放した。
- Phase 5 render matrix revision 1の`chat-hello`（13 input、17 output）をservice経由で再実行した。backend/HTTPは
  V620 17.043882049/17.044670008秒、R9700 7.656757501/7.657290528秒で、JSON/HTTP residualはそれぞれ
  0.787959/0.533027 msだった。別の1-token caseでJSON residual、SSE residual、concurrent queue waitを分離し、
  これらをengine性能やsteady-state throughputとは主張しない。
- target別binary SHA-256は`gfx1030=dd07bca6c1ca023365bc8800142302929ee50495993e431843aa35528b81723c`、
  `gfx1201=029fdf71f5899200915f1f8a5161316c6f9832f85dbb3ea9a7ddc188c677067b`、reportはV620
  `b8ad41a3f35c693b98fc6629e5997413726fb8e9ad8dc16de21a49c20a874d8f`、R9700
  `0648e41bb3a92ac60b82223a15b8ef2540ec9db7354da0ba29ecb5bf8c1f845f`である。

1. official pinned schemaからprofile subset fixtureを作り、raw HTTP contractをH1で実行する。
2. provenance付きでadaptしたllama.cpp testをsLLM profileへ限定し、llama独自fieldは持ち込まない。
3. vLLM/SGLang/llama.cppとのdifferential testは共通subsetの外形差発見に使い、いずれも仕様oracleにしない。
4. fake generation backendでAPI matrixをCPU CIへ置き、full model API smokeはcanonical GPUへ分離する。
5. Qwen3.5-4B、temperature 0のnon-stream/stream/stop/disconnectをcanonical `gfx1030`/`gfx1201`で確認する。
6. 選択したKV方式について、logical capacity、committed physical memory、page/block境界、disconnect後の
   解放を記録し、A0/A1のmodel-free evidenceとfull-model service evidenceを混同しない。
7. API実装前のdirect-engine baselineと同じcaseをservice経由で測り、JSON/SSE/queue overheadを別指標で記録する。

受入条件:

- H0/H1がschema、dependency、adapted provenance、HTTP/SSE negative matrixをPASSする。
- canonical GPU API smokeがexact model/target、HIP dispatch、fallbackなし、health/cleanup PASSとなる。
- official Python client smokeとraw SSE parserがPASSする。
- current profileで未定義のmid-stream failure behaviorを文書化してから互換性を主張する。
- disconnect後のpending HIP work、request-local KV/linear state、VRAM、GPU process cleanupが両canonical targetでPASSする。

上記5条件を満たしてA6とPhase 6を完了した。初期production serverは単一GPUをstable UUIDで単独可視化し、
論理device 0を使う。複数GPU可視processでglobal physical indexをworkerへ渡す構成とmulti-GPU servingは非対象である。

## 非対象

- Responses API、legacy Completions、Assistants、embeddings、audio、image、video。
- tools/function calling、structured output、logprobs、developer/tool message、multipart content。
- full dynamic/continuous batching、prefix cache/RadixAttention、複数model同時resident、multi-GPU、distributed serving。
- vAttention選択時のPaged Attention production backend、またはPaged Attention選択時のvAttention production backend。
  A1のmodel-free比較用kernel/proxyは非対象ではない。
- KV cache FP8/NVFP4と、選択方式以外を同時に実装するための共通最適化。
- TLS termination、OAuth、multi-tenant quota、distributed rate limiting、persistent request history。
- WebUI、production SLA、autoscaling、service mesh。

## 検証lane

- Draft: A0 model-free VMM PoC、A1のfocused KV host/GPU test、fake backend、tiny sampler oracle、
  raw HTTP/SSE contract、必要な単一GPU smoke。
- Integration: selected KV pathのaffected host/GPU suite、Python client、stream/non-stream/cancel matrix、canonical dual-GPU API smoke、
  1回のintegration review。
- Release/push: clean final identity、dependency/provenance closure、関連H/G matrix、累積review。
- Docs-only: profile、reader、provenance/link整合だけを確認し、GPU/API smokeを取り直さない。

## Rollbackと停止条件

- serverはadditiveなbinary/crateとして導入し、既存CLI generationをrollback baseとして維持する。
- A0 PoCはproduction KV pathへ暗黙に昇格させない。A1 decision recordとproduction ownership contractを経て接続する。
- vAttentionが不成立ならA0を失敗したPhase 6全体として停止せず、記録したcriteriaに従いPaged Attentionへ切り替える。
- sampling未完了時にtemperature/top_p/penaltyをsilent greedyへ落とさず、profile v1完成を主張しない。
- streaming failure時に部分responseを成功JSONへ変換しない。
- direct importを外す場合も過去distributed versionのnotice historyを消さない。
- 同じ作業単位が2回reject、review時間が実装時間を超過、1時間以上機能進捗なし、見積り1.5倍超過、
  またはprofile/gate変更時は追加review/testを止めてreplanする。

## 完了後

KV方式のdecision record、A0 PoC identity/summary、selected KV layout/capability、served profile version、
server/dependency identity、llama.cpp import record、host/API/GPU evidence、service overheadをmain-planとhistoryへ記録し、
本計画をarchiveする。Responses APIは将来profileとして別計画を作る。

[対応する履歴](../../../../../history/2026/08/11-20/openai-chat-completions-v1.md)

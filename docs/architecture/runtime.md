# ランタイムアーキテクチャ

## 設計原則

sLLM のランタイムは Rust workspace が主導し、GPU の操作だけを C++/HIP に閉じ込める。Rust はユーザー入力から実行計画までの意味論、型、安全な所有権を管理し、C++ は HIP runtime に近い resource と kernel 実行を管理する。両者の境界は HIP 専用の versioned C ABI とし、C++ ABI や Rust ABI を公開境界にしない。

初期 MVP は Qwen3.5-4B BF16 の単一 GPU 推論に限定する。ただし、後から backend や dtype、KV cache 表現を追加する際に上位層を作り直さないよう、op descriptor、capability query、KV layout abstraction は最初から設ける。

## ディレクトリと責務

```text
Cargo workspace
├── crates/sllm-core
├── crates/sllm-frontend
├── crates/sllm-hip-sys
├── crates/sllm-hip
├── crates/sllm-cli
├── crates/sllm-server
└── native/hip
    └── CMake project
```

| 領域 | 責務 |
| --- | --- |
| `sllm-core` | frontend、model/config、tokenizer、scheduler、sampling、backend 非依存の execution plan と tensor/op descriptor |
| `sllm-frontend` | typed chat renderer/tokenizer、transport非依存generation loop、sampling/stop/usage/cancellationの統合 |
| `sllm-hip-sys` | versioned HIP C ABI の宣言、check-in した generated bindings、`build.rs` による native build と link 情報の伝達 |
| `sllm-hip` | C ABI の安全な Rust wrapper、HIP backend 実装、resource ownership、非同期実行の lifetime と error 変換 |
| `sllm-cli` | CLI、設定の読み込み、runtime の組み立てと起動。GPU resource の詳細を直接扱わない |
| `sllm-server` | strict OpenAI profile DTO、model registry、bounded FIFO、HTTP/non-stream/SSE adapter、transport cancellation |
| `native/hip` | HIP context、allocator、queues、events、operator dispatch、backend 内 kernel registry、HIP kernels |

Rust 側は model graph を backend 非依存の op descriptor 列へ落とし、scheduler が request の実行順序と batch を決め、execution plan が tensor dependency と access mode を明示する。HIP 固有の queue 選択や kernel symbol は `sllm-core` に漏らさない。

## Backend 境界

`sllm-core` は上位の Rust `Backend` trait だけに依存する。trait は少なくとも次の概念を提供する。

- device の列挙と capability query
- buffer の確保と import/export 可能性の照会
- op descriptor の support query と coarse な command list/prepared plan の準備
- prepared plan の queue への非同期 submit と completion event
- backend status を Rust error へ変換する診断情報

`sllm-hip` がこの trait を実装し、`sllm-hip-sys` の C ABI を呼ぶ。上位 trait の version と HIP C ABI の version は別に管理する。将来 backend を追加しても、その backend に HIP C ABI を実装させる必要はない。

### HIP 専用 versioned C ABI

C ABI では context、buffer、queue、event、prepared op などを opaque handle として扱う。公開 struct には先頭付近に `struct_size` と `abi_version` を置き、追加 field は size を確認してから読む。予約 field はゼロを要求し、未知の必須 version は status error で拒否する。C++ class、STL container、reference、template、Rust layout に依存する型は ABI を越境させない。

すべての ABI 関数は整数の status code を返す。詳細な診断は呼び出し側が渡す error sink に書き込み、文字列の所有権と有効期間を ABI で定義する。非同期処理が caller の一時 buffer や callback context を保持しない設計を基本とする。保持が必要な API では retain/release 契約を明示する。

C++ exception と Rust panic は境界を越えてはならない。C++ entry point は内部例外を捕捉して status と error sink に変換する。Rust の `extern "C"` callback を設ける場合は panic を捕捉し、unwind させず failure status に変換する。destructor や release API は exception を送出しない。

### Handle の所有権

Rust の安全な wrapper では `ContextInner` を `Arc` で所有し、`Context`、`Buffer`、`Queue`、`Event`、`PreparedOp` の関係を次のように固定する。

- `ContextInner` は native context handle と shutdown state を所有し、公開 `Context` は `Arc<ContextInner>` を保持する。
- 公開 `Buffer`、`Queue`、`Event`、`PreparedOp` はそれぞれ対応する `Arc<BufferInner>`、`Arc<QueueInner>`、`Arc<EventInner>`、`Arc<PreparedOpInner>` を保持する。各 inner は自身の native handle と `Arc<ContextInner>` を所有し、公開 `Context` より長生きしても native context を早期破棄させない。
- `ContextInner` は子 handle の強参照一覧を持たない。親から子への強参照を作らず、循環参照と暗黙の resource 延命を防ぐ。
- `PreparedOpInner` は prepare 時に即時 copy された metadata と native prepared handle、および `Arc<ContextInner>` だけを保持する。個々の submission、caller の tensor descriptor、command buffer への pointer は保持しない。
- handle の clone/drop は Rust wrapper で定義した retain/release 契約に従う。native release は対応する Rust 所有者の最後の drop から行い、別種類の handle の破棄順序に依存させない。

`HipExecutor` は `Arc<ContextInner>` と実行中の `InFlightSubmission` を所有する。`ContextInner` から executor または submission への強参照は持たない。`InFlightSubmission` は completion までに必要な `Arc<BufferInner>`、`Arc<QueueInner>`、`Arc<PreparedOpInner>`、`Arc<EventInner>` を保持するため、利用者が公開 `Event` を先に drop しても resource は解放されない。`EventInner` は native event と terminal status/diagnostic を所有するが、実行中 resource の唯一の所有者にはならない。

shutdown は新規 prepare/submit を拒否する closing 状態への遷移、executor が所有する全 submission の drain、terminal status の回収、in-flight 参照の解放、executor 所有 queue の停止という順に行う。shutdown 中に完了を待たず context を破棄しない。外部に残る子 handle は shutdown 後の新規操作を拒否するが、安全に drop できるよう、その `Arc<ContextInner>` がなくなるまで native context の最終 release は行わない。

## Tensor、Buffer、非同期 lifetime

`Buffer` は device allocation の所有権を表す。Rust の安全な wrapper は buffer state を `Arc` で保持し、native opaque handle の最終 release は最後の所有者が離れ、かつ in-flight work がなくなった後に行う。

tensor と view は所有形態を曖昧にしない。owned tensor と、非同期 submit に使える owned view は `Arc<BufferInner>`、byte offset、shape、stride、物理 `DType`、device を保持する。同期的な descriptor 構築中だけ使う borrowed view は Rust lifetime で元 buffer に拘束し、submit 境界を越える API には渡せない。borrowed view を submit する場合は、検証後に明示的に owned view へ変換する。

view の作成や op への binding では bounds、alignment、shape/stride の整合を検証する。execution plan は tensor ごとに `read`、`write`、必要なら `read_write` access を宣言する。submit は全 owned view の `Arc<BufferInner>` を clone して `InFlightSubmission` へ移し、completion まで解放しない。書き込み先の再利用、読み書き競合、別 queue 間 dependency は access 情報と event dependency から決める。

同期 API を基礎にして非同期 API を装うのではなく、queue submission と event completion を基本契約とする。MVP が一つの compute queue しか使わない場合も lifetime 契約はこの形を維持する。

Phase 3のbackend-neutral transferでは、adapterが1回のH2D/D2Hに許す非zero `max_transfer_bytes`を広告する。上位層はsession、queue、buffer identityとchecked half-open `BufferRange`を検証し、上限を超えるrangeをbackendへsubmitしない。任意rangeのD2Hはsemantic opのoutput専用readbackと別のsingle-observer completion型にし、terminal success後かつsource rangeと同じcapacityのcaller-owned destinationにだけcopyする。HIP adapterは既存のversioned transfer ABIと`SLLM_HIP_MAX_TRANSFER_BYTES`へlowerし、新しいnative ABIや直接queue経路を作らない。

weight uploadは、model lockと全file hashを検証して保持した`VerifiedCache`、B5のcontent-bound load-plan digest、期待tensor名/dtype、tensorと同じ長さのdestination rangeを結合する。planの最大16 MiB chunkごとにverified FDからpositional readし、同時に保持するhost stagingを1 chunkへ限定して、backend-neutral `ExecutionSession::upload()`へ順次submitする。plan-global destination offsetはcallerが渡すtensor rangeへ相対変換し、packed allocationとtensor別allocationのどちらでもsource planを変えない。失敗時は部分upload済みbufferを有効なweightとして公開せず破棄する。常時D2H検証は行わず、evidence経路だけがgeneric buffer readbackでchunkごとのbyte exactを確認する。HIP専用weight wrapper、直接queue、shard/tensor全体のhost複製は作らない。

### Prepare、submit、非同期 error

C ABI の主要呼び出し単位は kernel 一個ごとの細粒度 call ではなく、複数 op を含められる coarse な command list または prepared plan とする。prepare は op/tensor layout など再利用可能な metadata を native 所有 storage へ即時 copy し、caller-owned array、文字列、一時 descriptor への pointer を返却後に保持しない。submit も resource binding、access mode、dependency event の metadata を call 中に即時 copy する。

submit の同期的な validation/enqueue failure は submit 自体の status と error sink で返す。enqueue 後に生じた非同期 error は completion event に保存し、`event_query` と `event_wait` の status および caller-owned error sink から取得する。`event_query` は pending、success、failure を区別し、failure の詳細が別 thread の一時 buffer に依存しないよう completion state が診断情報を所有する。event を drop しても executor は submission の完了または shutdown drain まで監視を続ける。

## Registry と dispatch

registry は責務の異なる三層に分ける。

| Registry | 所有者 | 役割 |
| --- | --- | --- |
| Backend registry | Rust runtime | 利用可能 backend の生成、device/capability による選択 |
| Op registry | `sllm-core` と backend adapter | 論理 op descriptor、必要 capability、backend support query、fallback 方針の対応づけ |
| Kernel registry | 各 backend。MVP では `native/hip` | op、物理 dtype、encoding、shape/layout、GPU target に応じた実装候補の選択 |

op descriptor は kernel 名ではなく、演算の意味、shape、layout、数値要件を記述する。backend は capability query で descriptor を受理できるか返し、prepare 時に具体的な kernel を選ぶ。実行開始後に不足 capability が判明する構成を避ける。

Phase 8のBF16 Matmul registryは、数値比較用baseline、M>1の16x16 tiled kernel、M=1のworkgroup
reductionを区別する。canonical `gfx1201`だけはM=1、K/Nとも1024以上のprevalidated shapeで
`hipblasGemmEx`を選び、`gfx1030`は同shapeでも実測で遅いためcustom reductionを選ぶ。library pathは
`[M,K] x [N,K]^T`をoperand transposeで表し、checkpoint weightを転置・複製しない。hipBLAS handleは
native context lifetimeで一度だけ作り、workspaceは0 bytes、provider error後のbaseline fallbackはない。
dispatch evidenceはprovider別kernel ID/symbol、shape、exact target、fallback flag、GPU event時間を保持する。

### Model-neutral prepared execution制御

model adapterはimmutableなnode列を`PreparedExecutionPlan<N>`へlowerし、requestごとのtoken数、開始position、
期待state長、binding generation、state generationを`PreparedTransition`として渡す。共通planがnode順序を所有し、
adapter callbackはmodel固有nodeからsemantic descriptor、binding、typed state submissionを構築する。共通moduleは
Qwen/Gemma等のmodel名、tensor名、head/vocabulary定数、kernel symbolを参照しない。

semantic prepareの再利用identityはhuman-readable labelを使わない。`SemanticOpDescriptor`、input/outputのbuffer ID、
`TensorView`、access mode、`PreparedDynamicIdentity`をexact keyとし、token/position/期待長、binding/state generationの
いずれかが変われば別entryにする。動的positionを内包するattention preprocess等は`Transient`として再利用しない。
cacheはrequest ownerに閉じ、resident model間で共有しないため、異なるmodel fingerprintやresident allocationのentryが
交差しない。

非同期submissionは型ごとのownerをadapterで待つのではなく、共通`ExecutionSegment`が同一ordered queue上で保持する。
adapterが`StatePublication`、`TerminalReadback`、`Cancellation`、`Error` boundaryを宣言し、boundaryのterminal event成功後に
segment内の先行completionをqueryしてdispatch evidenceを集約する。`PreparedExecutionAudit`はexact backend/target、
submission/kernel、fallback、segment/boundary countを成功済みrequestだけへ公開する。

request lifecycleは共通`ExecutionTransaction`がsingle in-flight、commit、drop/cancel/error時のpoisonを管理する。
adapterはtransaction開始前にmodel固有stateをadmitし、completion・readback・state length検証の後だけcommitして公開する。
pending、timeout、query failure、partial mutation、guard dropではoutput/stateを公開せず、同じrequest ownerの再利用を拒否する。
Qwen3.5 adapterはgraph lowering、attention preprocess、GDN/KV descriptor、Argmax/logits解釈だけを所有し、独自のprepared
cache、pending submission enum、flush loop、completion wait policyを持たない。

Gemma 4 adapterも同じplan/transition/segment/transactionを使い、48 layer・958 nodeのtensor/buffer layoutだけを所有する。
immutable weight/constant/ordered queueは`Gemma4ResidentModel`が保持し、request ownerはtoken/position、workspace、連続BF16
K/Vだけを持つ。prefill allocationをrequest capacity ownerとし、decode viewが収まる同名・同backing workspaceを再bindする。
prepared semanticもrequest内でexact descriptor/buffer/view/dynamic identityが一致する場合だけ再利用し、position/state依存の
attentionとKV appendはtransientを維持する。decode tailはpublished attention prefixと同じbufferのchecked offsetへappendし、
state-publicationとterminal-readbackの両boundary完了後だけ長さを公開する。greedyではArgmaxだけを返し、sampling時だけ最終
BF16 logits rowをbounded chunkでreadbackしてからtransactionをcommitする。

## DType と量子化 encoding

`DType` は BF16、FP16、FP8 など、要素の物理 scalar format を表す。量子化の scale、grouping、packing、codebook、tensor ごとの付加 metadata は `DType` に詰め込まず、独立した quantization encoding descriptor として表す。これにより、同じ低精度 storage dtype に複数の量子化方式を対応づけたり、weight、activation、KV cache で異なる encoding 制約を表現できる。

Qwen production graphはBF16、FP8、weight-only NVFP4のlinear bindingを同じ構造へ差し替える。NVFP4 v1はU8 packed
E2M1 value、K-axis block 16のOCP E4M3FN scale、tensorごとのFP32 scaleを別resident rangeとして所有し、
descriptor/cache identityはencoding、scale layout、provider、exact target、sidecar fingerprintを含む。`packed-dequant`は
requestごとにunpackせず、packed weightからBF16 activationとの積をFP32 accumulateしてBF16 outputを返す。
exact `gfx1030`/`gfx1201`以外、scale欠落、provider未指定、runtime failureではBF16/FP8へfallbackしない。

Phase 15Oではformatを変更せず、FP8 activation量子化をwave reduction/native pair conversionへ更新した。FP8は
activation量子化とhipBLASLtの2 dispatch、NVFP4は1 dispatchを維持する。NVFP4はM=1のdecode provider ID 8と、
K=256のweight tileを最大8 M rowで共有するM>1 prefill provider ID 9へ分離する。prefill展開はworkgroup内LDSだけで、
resident BF16 weightを作らない。prepare/runtime failure時の別dtype/provider fallbackは引き続き禁止する。

## KV cache layout

KV cache は通常の tensor descriptor に加え、layer、K/V の分離または interleave、token/block addressing、head grouping、stride、dtype、quantization encoding を表せる layout descriptor を持つ。Phase 6の初期方式はHIP VMMのvirtual-contiguous FP16 KVで、storage layoutはtoken-major `[capacity, kv_heads, head_dim]`である。create時に最大logical capacityのVAをreserveし、append前に必要なK/V physical pageだけをcommitする。model weight/activation の BF16 と KV cache の FP16 を同一 dtype として扱わない。

schedulerとgeneration serviceはopaqueなKV state/resource、logical token range、versioned view metadataだけを扱い、内部pointer arithmetic、VMM handle、block table、backend page sizeを所有しない。contiguous pointerはnative backend内だけでattention kernelへ渡す。このためvAttention上でもcontiguous-KV FlashAttention系kernelを利用でき、上位APIを変更せず将来paged/block layoutへ切り替えられる。Paged Attention production backendと量子化layoutは未実装である。詳細は[KV memory decision](kv-memory.md)を正とする。

Phase 11でVMM非対応が想定されるMI300X `gfx942`向けに、同じopaque resource、token-major FP16 layout、
contiguous attention pointerを保つ`contiguous-resident` providerを追加した。logical capacity分を通常のdevice
allocationで確保する。Phase 12のHot Aisle MI300X VFはVMM capability=trueだったが、開始時に固定した比較条件を
維持するためexact `gfx942`はcreate時にこのproviderを明示選択する。他targetはcapability-selectedのままで、
VMM capability=trueなら既存virtual-contiguous providerを維持する。これはPaged Attentionへの方針変更でも
実行時error後のfallbackでもない。必要byte、capacity、selected provider、resident allocationはdiagnostic/auditへ残す。

Phase 8のproduction causal attentionは、Qwen3.5のhead dim 256を一workgroupで協調reductionし、scoreの
再計算とthread-0 softmaxを一pass online softmaxへ置き換えたFA2-style pathである。opaque KV owner、
virtual-contiguous FP16 K/V pointer、token-major layout、GQA mappingは変更しない。これはupstream
FlashAttention-2そのもの、Paged Attention、RDNA4向けFA3-likeをclaimしない。FA3-likeは別の将来taskである。

## Generation service境界

Phase 6ではrender/tokenize/prefill/decode/sampling/stop/usageを`GenerationServiceV1`へ集約する。CLIとHTTP
adapterは入力DTOとtransportだけを担当し、token loopを複製しない。各呼び出しはrequest-localなexecution、
sampling履歴、stop matcher、cancellation flag、opaque KV stateを所有し、model weightとresident ownerだけを
request間で共有する。

temperature 0はQwen executionが返すdevice argmaxをそのまま使い、full-vocabulary logitsをhostへ読まない。
samplingが必要な場合だけterminal BF16 logits rowをbackendのbounded transfer単位へ分割してreadbackし、
temperature、top-p、presence/frequency penaltyを適用する。public requestにseedは持たせず、deterministic testは
明示的なrandom-source seamを内部serviceへ注入する。

stop文字列matcherはdecoded UTF-8の末尾がstop prefixである間はpublicationを保留し、token境界またはUTF-8
byte-fallback境界を跨いだ完全一致でもstop自身をvisible outputへ含めない。共通resultはprompt/completion/total
usage、`stop`/`length`、generated/visible/decode-input token列を保持する。cancelまたは途中errorはrequest ownerを
再利用不能にするが、resident model ownerを破棄しない。A5のHTTP disconnect/shutdown/timeoutはこのcancellation
境界へ接続する。

### OpenAI serverのadmissionとstreaming境界

Phase 6 A4/A5の初期serverは一つのworkerだけがgeneration backendを呼び、bounded FIFOを超えるrequestを
HTTP 429で拒否する。served aliasはmodel-lockのSHA-256 fingerprintとmodel-resident backend ownerへ結合し、
request-local errorやdisconnectでregistry entryを破棄しない。HTTP taskとworker間のgeneration eventもbounded
channelとし、slow consumerに対してbackend側を同期的にbackpressureする。

non-streamとSSEは同じdelta、finish reason、usage eventから構築する。SSEはassistant role chunk、0個以上の
nonempty content delta、terminal finish chunk、exact `[DONE]`の順に送る。response header送信後のgeneration errorは
standard error envelopeを一つのSSE data eventとして送り、finish chunkと`[DONE]`なしでcloseする。receiver dropは
request cancellationを発火し、scheduler timeoutとgraceful shutdownも同じflagへ伝播する。backendはbounded sink
へのpublish前後とlong-running operationの境界でcancellationを観測し、request-local stateを解放する。

A6以降のproduction backendはreviewed model lock kindからQwen/Gemmaを選び、verified tokenizer/templateまたは明示的な
Gemma raw-text transcript、weight plan、exact HIP sessionを一度loadする。`QwenResidentModel`または
`Gemma4ResidentModel`をworkerへ接続し、token loopは複製せず`GenerationServiceV1`を呼ぶ。各requestの
監査値はlogical KV capacity、mapped token capacity、physical page bytes、K/V committed bytes、full/linear
layer数、HIP submission、fallback、request/workspace allocationとcleanupを含む。成功responseはexact targetの
HIP dispatchのみ、fallbackなし、整合したphysical metadataを満たさなければfail-closedにする。

初期serverは単一GPU runtimeである。mixed-GPU hostでは`ROCR_VISIBLE_DEVICES=<stable GPU UUID>`で対象を1台だけ
可視化し、serverには論理device 0を渡す。HIP current deviceはthread-localなので、複数GPUを可視化したまま
global physical indexをworkerへ渡す構成は初期対応外である。

## Build integration

Cargo を top-level build entry point とする。`sllm-hip-sys/build.rs` が CMake を使って `native/hip` を configure/build し、Cargo に native link search path、library、必要な rerun 条件を伝える。CMake の configure/build/install output は Cargo が割り当てた `OUT_DIR` 以下だけに生成し、source tree や共有 build directory へ生成物を書かない。

`build.rs` は検出済みの `ROCM_PATH`、`CMAKE_HIP_ARCHITECTURES`、`SLLM_HIP_CODEGEN_FEATURES` を明示的に CMake へ渡す。CMake は別の ROCm や host GPU target を独自に再発見せず、同じ tree の `amdclang++` と渡された target/features を使う。release build で target/features が不足する場合は configure error とし、生成した native artifact の metadata に実際の入力を記録する。

C ABI から生成した Rust bindings は repository に check-in する。通常ビルドは bindgen の実行を必須にせず、明示的な再生成操作だけが bindings を更新する。生成元 header、ABI version、生成 tool/version、生成 option を固定し、header を変更した commit では対応する generated bindings も同時に更新する。安全性、所有権、`Send`/`Sync` の判断は generated code に持たせず `sllm-hip` の手書き wrapper に閉じ込める。

### Phase 1 host stub

Phase 1では、CargoからCMake static libraryをbuild・linkする経路とversioned C ABIを実際に成立させる一方、HIP compiler、ROCm library、GPU処理は導入しない。`native/hip`はC++17 host stubとしてbuildされ、ABI versionとlibrary versionを返し、HIP backend/context probeには`SLLM_STATUS_HIP_UNAVAILABLE`を返す。成功、CPU fallback、GPU対応として扱わない。

公開headerは`include/sllm/hip.h`、check-inしたbindingsは`crates/sllm-hip-sys/src/bindings.rs`、Cargo/CMake統合は`crates/sllm-hip-sys/build.rs`を正とする。Phase 2のH3でHIP languageとtarget別codegenを追加するまで、stub artifactをHIP compile evidenceまたはGPU evidenceに使用しない。

Phase 1のbackend-independent contractは、read/write access mode、opaque queue/buffer/event handle、completionまでresourceを強参照するownership tokenを含む。これは非実行のlifetime contractであり、非同期実行や完了を偽装しない。C/Rust ABIのsize、alignment、field offset、constantはbuild時のnative layout probeでcheck-in bindingsと機械的に照合し、C/C++両方で公開headerをcompileする。

### Phase 2 model-free evidence path

Phase 2の最小GPU経路はpublic inference ABIへ未成熟なopを追加せず、private evidence ABIと専用`sllm-hip-evidence` binaryへ分離する。Rustはinputをnative submit中にcopyさせ、opaque integer handleを一度だけwait/destroyできる所有型として保持する。native completionはHIP stream、event、device buffer、pinned host bufferをcompletionまで所有し、各caseで2 device allocation、2 HIP transfer、1 diagnostic kernel dispatchを行う。

caseは1、3、17、255、256、257 byteとし、Rust側の独立XOR oracleへbyte exactで照合する。host stubは明示`HIP unavailable`を返し、CPU fallback、model、semantic numerical opはこの経路に存在しない。HIP buildはbare exact `gfx1030`または`gfx1201`を要求し、runtimeのraw `gcnArchName`、embedded Code Object V6/target/ELF flags/wave32/kernel symbol、実際にloadしたHIP/ROCr library pathが契約と一致しない場合は実行evidenceにしない。

この経路はcommit `f393d688a051d2b73c8773d8a930a711592609bc`でcanonical `gfx1030`/`gfx1201`のG1をPASSした。これはmodel-free診断経路のarchitecture evidenceに限り、public semantic op、数値正しさ、model推論、性能または一般GPU対応の証拠ではない。

timeoutまたは早期drop後のresourceは完了を証明せずにfreeしない。background reaperへ所有権を移し、回収枠を固定上限へ制限する。同期・解放を証明できなければcircuit breakerを開いて新規submitを拒否し、専用processの終了とtrusted local runnerによる子process/GPU process残留確認を最終cleanup境界とする。このprivate pathは将来のsemantic command list ABIや一般的なGPU対応を確定しない。

### Phase 3最初のpublic semantic op

最初のpublic semantic opはRMSNormとする。private diagnostic G1を昇格または流用せず、`SemanticOpDescriptor -> Backend -> sllm-hip -> versioned public C ABI -> native op registry -> HIP kernel registry`を通す。private diagnostic G1はallocation、transfer、lifetime、loaderの回帰evidenceとして並行して残し、semantic RMSNorm G1は独立したschema、runner、aggregateで数値正しさを記録する。

baseline RMSNormはBF16 activation、BF16 raw scale weight、BF16 output、FP32 accumulation、row-majorで連続した最終次元に限定する。Qwen3.5 HF checkpointでは実効scaleをFP32で`1 + raw_weight`として適用し、raw weightを通常scaleとして直接乗算しない。disk上のweightを事前変換せず、descriptorのversioned scale modeでoffset-one semanticsを明示する。epsilonはlocked model configまたは明示的なsynthetic caseから取得し、暗黙の既定値へfallbackしない。初期実装はin-placeを許可せず、input/output alias、unsupported dtype/encoding/scale mode/stride/alignment/shape、zero-lengthをcapability queryまたはprepareで明示的に拒否する。

Rustはsemantic descriptor、owned tensor view、buffer access、completionまでの強参照を所有し、native側はopaque resource、metadataの即時copy、dispatch、kernel選択、非同期completionを所有する。exact target、layoutまたはkernelが適合しない場合は別backend、generic kernel、CPUへfallbackしない。公開ABIを拡張するときは既存v1を壊さず、additiveに表せない変更だけversionを上げる。

synthetic caseによるsemantic G1の後、固定model lockから抽出した実RMSNorm weightと独立生成activationをG2で検証する。短いP0は同じpublic RMSNorm pathのkernel latencyとdispatch境界だけを観測し、full model性能または最適化済みであることを意味しない。

## MVP の対象外

次は初期 MVP に含めない。

- dynamic backend/plugin loading
- runtime JIT compilation
- 複数 compute stream を使う scheduling と overlap 最適化
- multi-GPU、Infinity Fabric、その他 RDMA transport
- backend 外へ公開する kernel plugin ABI

これらのための実装や不完全な ABI は先行追加しない。一方で、opaque handle、capability query、非同期 event、op/KV descriptor により、将来の追加で上位の model/scheduler API を破壊しない境界を保つ。

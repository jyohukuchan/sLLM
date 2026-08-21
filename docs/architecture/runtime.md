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

server contextはmodel artifactが宣言するnative/推奨値と、operator/stateが実際に使用する実行容量を分離する。
前者は省略時の既定値と起動時advisory warningにだけ使い、graph、KV、RoPE、attentionのhard gateには使わない。
後者は`--context-length`で指定し、requestのprompt+最大outputをadmissionする上限となる。ABIの固定幅位置表現、
1 dispatchのshape上限、memory確保失敗は独立した実装制約であり、model品質上の推奨値へ偽装しない。

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

Phase 34は同じregistryと`hipblasGemmEx`実装をexact `gfx1030`の長行projectionへ限定して再利用する。主要5 internal shapeは
`M>=128`、Full Attention K/Vの`K=2560,N=1024`は`M>=1024`でHipBlasを返す。`N=32`、未知shape、
all-logits vocabulary shape、短Mは従来providerを維持し、gfx1201/gfx942 ruleは変えない。gfx1030 contextへhipBLAS handleを
一つ追加するがhipBLASLtは作らず、graph、tensor layout、weight、public ABI、retry fallbackを複製しない。

MTP target verifyは実際のdecode block rowをすべてterminal outputとして保持する。long-prefill graph capacityを理由に
speculative verify blockをlast-rowへ圧縮しない。通常prefill、通常decode、partial replayのterminal-row compactionは維持する。

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

Phase 21では、通常opをeventlessにして非空segmentを一つのuntimed queue fenceで閉じ、同一queueのownerをfence成功後に
個別queryなしでfinalizeできるadditive ABI/core primitiveを実装した。standalone completionとprofile/evidenceの
`PROFILED` timing contractは維持される。17 ownerを1 eventへ集約する構造削減はhostで成立したが、final dual-GPU
counterbalanced laneではV620/R9700ともwall中央値が0.14%/0.18%遅くnoise内だったため、Qwen/Gemma productionは
`ExecutionSegment::profiled`を選び続ける。deferred primitiveは実験基盤であり、現在のproduction defaultや性能claimではない。

request lifecycleは共通`ExecutionTransaction`がsingle in-flight、commit、drop/cancel/error時のpoisonを管理する。
adapterはtransaction開始前にmodel固有stateをadmitし、completion・readback・state length検証の後だけcommitして公開する。
pending、timeout、query failure、partial mutation、guard dropではoutput/stateを公開せず、同じrequest ownerの再利用を拒否する。
Qwen3.5 adapterはgraph lowering、attention preprocess、GDN/KV descriptor、Argmax/logits解釈だけを所有し、独自のprepared
cache、pending submission enum、flush loop、completion wait policyを持たない。

Phase 31ではQwen text prefillをrequest-localな連続chunkへ分割できるようにした。device total VRAMが16 GiB以下では512、
16 GiB超では16K/8K/4K/2K/512を大きい順に、model-resident bytes、全request終了時のKV/GDN state、candidate行数の
workspace high-water、`max(total VRAMの5%, 1 GiB)` reserveからdispatch前に一度だけ選ぶ。promptがcandidateより短ければ
actual prompt行数を使い、allocation失敗後の小bucket retryは行わない。absolute position、full-attention KV、GDN stateは
chunk間で継続し、中間chunkではLM head/Argmax/visible outputを省略する。中間chunk末尾は同一queueのterminal fenceで
submissionを完了させてから次chunkへstateとworkspace slotを渡し、最終chunkだけ通常decodeへ接続する。

Qwen dynamic intermediateはtensor別allocationの総和ではなく、graph use intervalをsubmission completion boundaryまで延長した
liveness slotへ配置する。同時liveまたは同じcompletion segmentのtensorは別slotとし、backendがbuffer handle単位で保持する
in-flight leaseをaliasで破らない。selected capacityに対するslot群は全chunkで再利用し、model weight、KV/GDN state、terminal
outputはarenaへ入れない。10,001行では従来39,950,821,120 byte相当の個別workspaceに対してhigh-water
5,278,049,280 byte、16,385行では65,448,547,584 byteに対して8,646,688,768 byteとなった。

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

Phase 15Qでは同じbindingをGemma 4のMLP gate/up/downへ接続した。`Gemma4ResidentModel::new_nvfp4`はverified Gemma source lockと
sidecar fingerprintをresident identityへ含め、sidecarに存在するweightだけをpacked NVFP4 allocation/uploadへ置換し、残りを
exact BF16 cacheからloadする。primary comparisonは144 MLP tensorを要求する。1〜143 tensorのpartial sidecarはlayer単独・累積
感度診断専用で、decode rebindでも同じsidecar fingerprintを必須とする。いずれもrequestごとのBF16全weight展開や別providerへの
fallbackを行わない。

Phase 15Oではformatを変更せず、FP8 activation量子化をwave reduction/native pair conversionへ更新した。FP8は
activation量子化とhipBLASLtの2 dispatch、NVFP4は1 dispatchを維持する。NVFP4はM=1のdecode provider ID 8と、
K=256のweight tileを最大8 M rowで共有するM>1 prefill provider ID 9へ分離する。prefill展開はworkgroup内LDSだけで、
resident BF16 weightを作らない。prepare/runtime failure時の別dtype/provider fallbackは引き続き禁止する。

## 量子化modelの選択と内部状態

公開runtimeの最終interfaceはmodel artifactを指定する同一操作をBF16、FP8、NVFP4、MXFP4へ使う。GGUF metadataとexact targetから
encoding、mixed-precision recipe、providerをloaderが自動解決し、低bit modelだけに追加の許可flag、確認prompt、通常警告を要求しない。
量子化済みartifactの選択をユーザーの明示選択とみなす。provider名やscale layoutは`doctor`、明示的なdiagnostic、benchmark reportで
確認可能にするが、通常のgenerate/server応答へ品質警告を注入しない。

Phase 20完了後の公開CLI/serverは`--gguf PATH --derived-lock PATH`だけをmodel入力として受け付ける。
旧`--lock`/`--cache`、量子化sidecar、provider overrideは公開parserから削除した。safetensorsとsidecarのreaderはconverter・
開発adapterに限定し、GGUFのhash/schema不一致、未対応encoding、memory/shape contract不成立はfallbackせずerrorにする。
Qwen dense GGUFはtextだけでなく、同じverified descriptorからvision tensorとMTP componentをscope付きでlowerする。
multimodal requestはrank-4物理tableから復元したlogical shapeでvision graphを構築し、text-only greedy `gfx1201`は同じGGUF内の
MTP planをidentity/digest検証して内部providerへ渡す。

内部状態は一つの序列へ潰さず、少なくとも次を独立に記録する。

- runtime成熟度: `supported / experimental / unsupported`。loader/provider実装の安全性と完全性を表す。
- provider選択: exact targetごとの自動優先順位と、実際に選択したexecution pathを表す。
- converter品質: sLLM製PTQ recipeが対応BF16 sourceから許容範囲内の品質を維持するかを表す。
- model evidence: 提供元PTQ/QAT、native low-bit、sLLM変換の別と、reference runtime、task、GPU targetの検証scopeを表す。

BF16 sourceがあるsLLM製PTQにはBF16 KLD budgetを適用できる。提供元PTQ/QATは同じquantized checkpointのreference実行とtask評価、
BF16を正本として公開しないnative low-bit modelはartifact fidelity、reference実行、task評価で判定する。converter不採用をencodingまたは
runtime providerの不支持へ転用せず、逆に正しいdecodeだけでmodel task品質を証明したとも扱わない。

## KV cache layout

KV cache は通常の tensor descriptor に加え、layer、K/V の分離または interleave、token/block addressing、head grouping、stride、dtype、quantization encoding を表せる layout descriptor を持つ。Phase 6の初期FP16方式に加え、Phase 16は`kv-fp8-v1`（token/headごとのE4M3FN valueと独立FP32 scale）と`kv-nvfp4-v1`（low-nibble-first E2M1、block-16 E4M3FN scale、token/headごとのFP32 outer scale）を追加した。いずれもlogical shapeはtoken-major `[capacity, kv_heads, head_dim]`で、opaque stateがK/Vのvalue/scale planeを所有する。FP8のNaNはE4M3FN NaN、Infは最大有限値へ写す。NaN/Inf codeを持たないNVFP4はNaNをcanonical zero、Infを有限値由来のrow scaleで表現可能な上限へ飽和する。HIP VMM providerはcreate時に最大logical capacityのVAをreserveし、append前に新規token範囲へ必要なphysical pageだけをcommitする。VMM非対応targetでは同じencoding contractをcontiguous-resident allocationで実装する。model weight/activationのdtypeとKV cacheのdtype/encodingは独立に選ぶ。

schedulerとgeneration serviceはopaqueなKV state/resource、logical token range、versioned view metadataだけを扱い、value/scale pointer、内部pointer arithmetic、VMM handle、block table、backend page sizeを所有しない。appendは新規BF16 K/Vだけを一度量子化し、K/Vと全scale planeの完了後にlogical lengthをatomicに公開する。causal attentionはFP16、packed FP8、packed NVFP4をstateから直接読み、request全体のFP16/BF16 mirrorを作らない。legacy create/readback ABIはFP16のまま維持し、additive create v2で低bit recipeを指定する。旧evidence readbackへ低bit stateを渡した場合はpacked bytesをFP16と誤認せず`unsupported encoding`でfail-closedにする。Paged Attention production backendは未実装である。詳細は[KV memory decision](kv-memory.md)を正とする。

Phase 31はBF16 weight graphとKV encodingの選択を分離し、Qwen CLI/serverへ`fp16`、dynamic `fp8`、
`fp8-static`、`nvfp4`の明示設定を通した。省略時は引き続きFP16である。static FP8は明示選択時の固定K/V scale 1.0を
descriptorへ入れ、zero scaleをfail-closedする。low-bit選択時は未検証のMTP/multimodal/MoE組合せを拒否し、別encodingへの
fallbackを行わない。この公開選択はlow-bit KVの長context検証を可能にするが、全modelのdefault昇格や品質保証を意味しない。

Phase 32はFP8 appendのsemantic kernel、256-thread workgroup、grid、scale recipe、value/scale plane、publicationを維持し、
最終F32→OCP E4M3FN encodeだけをexact `gfx1201` device compileでnative scalar conversionへlowerする。NaN、Inf、signed zero、
448 saturationはsoftware contractへ明示補正する。exact `gfx1030`は同じsourceからsoftware helperを生成し、FP16/NVFP4、
public ABI、KV format、default FP16は変えない。native packed pair/128-thread候補はproductionへ採用しない。

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

Phase 30は同じsemantic op、opaque KV layout、256-thread launch ABIの内側にexact gfx1201 providerを追加した。
`query_count=1`または`query_count>=32`ではwave32 shuffleで8個のQK partialを固定tree合成し、key当たりのblock同期を
約11回から3回へ減らす。FP8/NVFP4 scaleのE4M3FN readはgfx1201 native scalar conversionを使う。`query_count=2..31`、
gfx1030、その他targetは従来providerを維持する。targetとquery-count境界はruntime dispatch metadataとactual kernel launchで
同じ判定を使い、prompt/token/model固有値をrouting keyにしない。public API、KV encoding、state publication、softmax式、
FP32 accumulator、BF16 RNE outputは変えない。これはmatrix instructionを使うFlashAttention providerのclaimではない。

Phase 33はexact gfx1030/gfx1201、head dim 256、GQA ratio 4のprefill `query_count>=64`へ、1 query rowと同じKV headを
使う4 query headを一workgroupで処理するGQA4共有providerを追加する。K/V elementは既存token-major planeから一度だけ
direct decodeし、4 headのQK、online maximum/denominator、weighted Vを独立に更新する。gridは`M × kv_heads`、
global scratchと追加dispatchは0である。`M<=63`、別head shape/ratio/targetはPhase 30またはbaseline providerを選ぶ。
runtime error後のfallbackは行わない。gfx1201は既存wave reduction順を維持し、gfx1030は同じ8段boundの固定wave treeを使う。
採用tileは4 rowしかなく16×16×16 WMMAへ同じlayoutのまま写せないため、Phase 33ではmatrix providerを追加しない。

同Phaseの`M=1`、KV長1,024以上を8 waveの連続KV区間へ分けるdecode providerは、scratch 0で大幅に高速化する一方、
QK reductionの依存深さが概ね8段から12段へ増えるN2である。独立oracle、token一致、両targetのdevice短縮を確認し、
2026-08-20のユーザー承認によりN2分類を維持したままproductionへ限定採用した。KV長1,023以下とscope外target/head shapeは
Phase 30またはbaseline providerを選ぶ。

Phase 35は同じopaque KV stateとsemantic opの内側で、exact gfx1030/gfx1201の`query_count>=128`を
4 query row × 1 KV head/workgroupへrouteする。8 waveが16 logical `(row,GQA head)`を分担し、各K/V elementを一度だけ
decodeして共有する。causal key集合、logical queryごとのonline softmax、FP32 accumulator、BF16 RNE outputは維持し、
global scratch、追加dispatch、KV layout変更はない。`query_count<=127`、decode、別shape/targetはPhase 33以前のproviderを選ぶ。

同PhaseのGDN long-prefill providerはtoken count 128以上でQ/K normalizationとbeta/decay、column-owned recurrent state、
output RMSNorm/z gateを4 dispatch familyへ分ける。recurrent gridはQwen shapeで1,024 workgroup、各waveが1 state columnの
128行をlane当たり4 FP32 registerへ保持してtoken順に更新する。既存のtarget別物理state index、previous/next transaction、
conv state、short/decode providerを維持し、10,001 tokenで追加するrequest scratchはbeta/decayの2 FP32 planeだけである。
診断用baseline overrideはpublic APIや実行失敗後fallbackではない。

## Generation service境界

Phase 6ではrender/tokenize/prefill/decode/sampling/stop/usageを`GenerationServiceV1`へ集約する。CLIとHTTP
adapterは入力DTOとtransportだけを担当し、token loopを複製しない。各呼び出しはrequest-localなexecution、
sampling履歴、stop matcher、cancellation flag、opaque KV stateを所有し、model weightとresident ownerだけを
request間で共有する。

temperature 0はQwen executionが返すdevice argmaxをそのまま使い、full-vocabulary logitsをhostへ読まない。
samplingが必要な場合だけterminal BF16 logits rowをbackendのbounded transfer単位へ分割してreadbackし、
temperature、top-p、presence/frequency penaltyを適用する。OpenAI requestの任意`seed`とCLIの`--seed`は同じ
sampling RNGを初期化し、同一model artifact、runtime、target、prompt、generation parameterの反復を再現可能にする。
異なるsoftware/GPU tuple間のbitwise再現性や`system_fingerprint`互換は主張しない。temperature 0のgreedy pathは
seedの有無にかかわらずrandom sourceを読まない。deterministic testは引き続き明示的random-source seamも利用できる。

stop文字列matcherはdecoded UTF-8の末尾がstop prefixである間はpublicationを保留し、token境界またはUTF-8
byte-fallback境界を跨いだ完全一致でもstop自身をvisible outputへ含めない。共通resultはprompt/completion/total
usage、`stop`/`length`、generated/visible/decode-input token列を保持する。cancelまたは途中errorはrequest ownerを
再利用不能にするが、resident model ownerを破棄しない。A5のHTTP disconnect/shutdown/timeoutはこのcancellation
境界へ接続する。

### Phase 40 sampler・grammar・selected-only generation

Phase 40はこのrequest-local境界にversioned `SamplerChainV1`、bounded grammar、確率metadata、choice stateを接続する。
chainの順序はraw logitsのfinite検証、biasとhistory penalty、grammar/EOS mask、temperature、candidate filter、terminal selector、
logprob metadataで固定する。legacy-v1で追加fieldを省略した場合は既存device Argmaxとtoken/tie/RNG/stop semanticsを維持する。
追加chainはtop-k、min-p、typical、repeat、dynamic temperature、DRY、XTC、Mirostat、ignore-EOSを含み、NaN/Inf、zero mass、
all-masked、上限超過はerrorへlowerする。

grammarはRust frontendのraw token bytes、bounded token trie、partial UTF-8 stateで受理可能tokenのU8 maskを作る。GBNFとJSON object、
JSON Schemaの明示subset（object/array/string/number/integer/boolean/null、enum/const、required、`additionalProperties:false`、
`anyOf`、local `$defs`/`$ref`）だけをcompileし、unsupported keyword、remote/recursive reference、state explosionを受付時に拒否する。
generic `json_object`はdepth 1、containerあたり最大4 members/items、string/number 64、whitespace 16のbounded grammarであり、JSON Schemaは
global property/state limitsを別に適用する。
選択後もgrammar stateをchoice ownerへ保持し、SSE/non-streamのlogprobはpost-mask/post-filter分布から同じtoken IDのtext/raw bytesへ写像する。

`n=1..=8`はchoice index、derived seed/RNG、sampler history、grammar、stop matcher、usage、KV/generation ownerを分離する。
Qwen/GemmaのGPU selector対応subsetでは、terminal projectionと同じqueue上でadditive F32とvalid-token U8 maskをuploadし、
`TokenSelect`をsubmitする。completion後に固定16-byte selected recordだけをD2Hし、full-vocabulary logits D2Hは0にする。selectorが
対応しないsamplerはhost full-logits pathを選択し、GPU route開始後のCPU silent fallbackはしない。MTP block selectorはこのM=1 contractの対象外である。

Phase 40のHIP selectorはadditive ABIとして既存Argmax ABIと分離し、exact target/capability、status、reserved、token範囲、finite
logprobをfail-closedに監査する。gfx1030/gfx1201のselector contract matrixはvocabulary・counter・CPU oracle・selected-only D2HをPASSした。
gfx942はwave64 feature-pinned compile/routeのみPASSで、MI300X real correctness/performanceはVM再確保後へdeferredする。直接llama.cpp
source reuseはなく、provenance lockは変更しない。

### Phase 41 prefix・context・checkpoint state

Phase 41はrequest-local ownerを壊さず、公開済みquiescent stateだけをcross-request boundaryへ出す。prefix keyはmodel-lock、
derived artifact/plan、adapter、renderer/tokenizer、exact tokens、KV encoding/layout、target semantics、context policyを含む。
bounded longest-prefix indexはimmutable entry、lease、checked reader/LRU/accountingを持ち、workspace、queue、prepared plan、
terminal selectionをcache ownerへ移さない。

opaque state forkはKV encodingごとのvalue/scale/outer-scale planeとlinear/GDN active stateを一つのtransactionとして扱う。
VMM pathはcomplete pageをread-only共有し、append対象tailをCOWする。contiguous pathとGemma sliding K/Vはsame-device D2D cloneを使う。
post-COW queryでchild owned bytesをrequestへ追加予約し、cache quotaはshared physical bytesを一度だけ数える。fork/import/exportは
additive C ABIであり、partial import、異session raw image、identity/layout/encoding mismatchをpublication前に拒否する。

`keep-prefix-recent-v1`はretained logical rangesとabsolute positionsを分離する。Qwen/Gemma executorはcapacity到達前にretained
tokensからfresh ownerを作り、RoPE/attentionへexplicit absolute positionを渡し、成功後だけownerとcompact historyを交換する。
Qwen GDN/linear recurrenceはcompact logical sequenceから再計算する。unsupported model/encoding、prefix/draft/multimodal、
device-selectorとの組合せはGPU work前にfail closedとする。

checkpoint envelopeはlittle-endian `sllm-session-checkpoint-v1`で、全section digest、全file digest、bounded length/countを検証する。
Qwen/Gemma productionはstateless prompt checkpointだけを提供する。loadはbackend open時にstrict検証したimmutable snapshotを保持し、
request tokensがcheckpoint historyをprefixとして持つnonempty suffix continuationだけを許す。saveはfresh prompt prefill後かつ最初の
visible delta前にatomic replaceする。mid-generation resume、暗黙のglobal conversation、client間state共有はこのcontractに含めない。
sampler/RNGとgrammarは将来の明示session ownerが利用できるversioned bounded snapshotを持つが、prompt checkpointのfrontend state
sectionはcanonical emptyである。

assistant prefillはrender/tokenize後のprepared prompt stateとしてdecoder、grammar、stop matcherをprimeし、visible completionへ
再公開しない。MTP、external、ngramはmodel-neutralなbounded proposal/verification/publication/accountingを共有し、target samplerだけが
visible tokenとRNGを所有する。external executorがprovisionされていないproduction configは実行可能providerへfallbackせず拒否する。

### Phase 44 template・reasoning・interactive boundary

Generic templateはreviewed rendererの代替ではなく、callerがsource bytesとlowercase SHA-256を提示した明示opt-in providerである。
MiniJinja `2.24.0`をexact pinし、rendererへ渡す値はJSON-onlyのmessages、tools、special-token strings、generation/thinking flags、
reasoning effort、bounded kwargsに限定する。filesystem、environment、network、process、host object/method callback、credential/secret、
dynamic loader、include/import/extends、private attributeへ到達する経路はない。source、rendered output、messages、kwargs、recursion、fuelの
上限はcompile/render/tokenizeとscheduler/GPU admissionより前に適用する。

Generic adapterは`GenericTemplateMessagesInputV1`を`TokenizerUtilityServiceV1`へlowerし、rendered bytesを一度tokenizeする。result identityは
profile/template digest/source size、kwargs digest、rendered digest/sizeを含み、checkpoint/prefix keyへexact renderer/template identityを
結合する。raw textとGemma raw-textはgeneric inputへ暗黙変換せず、capabilityなしbackendはGPU work前にrejectする。

Reasoningは別generation loopを作らず、既存selector、grammar、stop、sampling、cancellationのcandidate maskと同じownerで制御する。
budgetは1〜4,096 generated reasoning token、closing marker列を含むmax-output admission、early close、forced close、grammar/stop conflictを
checked stateへlowerし、forced tokenもusage/generated historyへ通常tokenとして記録する。CLI/Chat/Responsesは同じfrontend controllerを使い、
transport wireのreasoning/visible splitだけをadapterが表現する。

Interactive `chat`はprompt/message/prompt-file/interactive stdinのsourceを混同しないclosed matrix、bounded regular-file reader、typed
conversation、reverse-prompt boundary、JSONL eventsを所有する。turn commitはgeneration成功後だけで、save/resume時のPhase 41
checkpoint bytes/KV/GDN stateはopaque runtime ownerへ渡し、CLI state machineはGPU planeを解釈・複製しない。Persistent chatのsuccessful turnは、
reviewed Qwen history semanticsでhidden reasoning、selected stop token、matched reverse markerを除外したcanonical history prefixへrebaseし、
fresh resident ownerへre-prefillしてからopaque checkpointをcaptureする。これによりnext turnとfresh resumeのprompt token prefixが一致する。
checkpoint loadはmodel、renderer、tokenizer、target、weight-plan、KV encoding/descriptorのexact identityをtransactionalに検証し、失敗時は
current ownerを公開しない。conversation bytesとKV pending/currentは同じcommit boundaryでpromoteし、generation/capture/save/commit失敗やcancelは
pendingを破棄してcurrent conversation+KVへrollbackする。CLI production adapterはprompt/source/reasoning/limitsをpreflightしてからmodel/backendをopenし、
prompt fileは一度だけ読む。SIGINTは専用listenerからcurrent turnの`GenerationCancellationV1`だけをcancelし、次turnへtokenやpending stateを持ち越さない。
mid-generation/wire session resume、WebUI、tool/MCP executionはこのboundaryの外である。MI300X real executionはVM再確保までdeferredであり、
gfx942 compile/host evidenceをruntime capabilityやGPU PASSへ昇格しない。

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

Phase 39はこの単一worker境界を維持したままoperability stateを追加する。server lifecycleは
`loading/ready/draining/failed/shutdown`をatomicに共有し、readinessは`ready`かつscheduler acceptingの場合だけ成功する。
schedulerは非ゼロ単調slot IDとboundedな`queued/active/cancelled` registryを持つ。公開snapshotはID、model alias、stateだけで、
prompt、token、credential、backend内部stateを所有しない。queued cancelはdequeue時にterminal cancellationとなり、active cancelは
既存のgeneration cancellation tokenへ伝播する。

通常SSEは従来どおりreceiver dropでgenerationをcancelする。明示opt-inのresumable SSEだけはgeneration receiverをbounded
process-local replay producerへ移し、transport disconnectとgeneration lifetimeを分離する。replay event IDはsession内で単調、
session/event数は起動時上限、active sessionはcapacity evictionしない。cursorが保持windowより古い場合は再開をfail closedにする。

metrics registryは起動時の最大16 model aliasとcompile-time固定enumだけからseriesを作る。production backendのmemory観測は
既存`ExecutionSession::memory_snapshot()`を`try_lock()`で読み、model-resident、request/KV、workspace/arena、totalの
current/high-waterだけへlowerする。scrapeはgeneration mutexを待たず、busy/shutdown/poison時はzero snapshotとする。
これはsLLM allocatorのdevice accountingであり、driver全体のVRAM telemetryではない。

listenerはTLS無効時もTLS有効時も同じrouterとgraceful drainを使う。certificate/keyのpairとPEM、private keyのfile type・権限は
model backendをopenする前に検証する。CORSは完全一致originのbounded allowlistだけをrouter layerへ渡し、disabled defaultでは
layer自体を追加しない。credential storeとrole境界は[credentials](../security/credentials.md)を正とする。

Phase 26ではwaiting/decode-ready、compatibility class、checked row map、round-robin、bounded prefill挿入、backpressureを
model/device非依存に検証するhost plannerを追加したが、production schedulerへは接続していない。現行Qwen ownerの
`committed_length`、KV state、linear/GDN stateはrequestごとのscalar contractであり、既存`M>1` decodeは一request内の
speculative token blockである。独立requestをこの形へ束ねるとcausal stateを共有するため、GPU `B>1`として使用しない。
production continuous request batchingには、per-row positionと独立KV/GDN binding、row-local transactional publicationを
core/native ABI/kernelまでadditiveに通す別work unitが必要である。Phase 26はこの境界でcandidate棄却となり、初期serverの
FIFO、whole-generation backend mutex、production defaultは維持している。

A6以降のproduction backendはreviewed model lock kindからQwen/Gemmaを選び、verified tokenizer/templateまたは明示的な
Gemma raw-text transcript、weight plan、exact HIP sessionを一度loadする。`QwenResidentModel`または
`Gemma4ResidentModel`をworkerへ接続し、token loopは複製せず`GenerationServiceV1`を呼ぶ。各requestの
監査値はlogical KV capacity、mapped token capacity、physical page bytes、K/V committed bytes、full/linear
layer数、HIP submission、fallback、request/workspace allocationとcleanupを含む。成功responseはexact targetの
HIP dispatchのみ、fallbackなし、整合したphysical metadataを満たさなければfail-closedにする。

Phase 31以降のQwen監査値はこれにselected prefill chunk capacity/count、device total/available/required bytes、
model-resident/request-state/safety-reserve bytes、workspace arena high-waterと旧個別allocation合計も加える。chunkはwire上の
request、usage、SSE eventへ露出せず、OpenAI usageのprompt token数はrender後の全入力、completion token数は生成分だけを数える。
CLIの初回placementはfull layout requiredを未確保のavailable bytesへ比較する。serverはmodel residentを起動時に確保済みなので、
request admissionでは`full required - graph model resident`をcurrent availableへ比較し、full/incremental requiredの両方を監査する。

初期serverは単一GPU runtimeである。mixed-GPU hostでは`ROCR_VISIBLE_DEVICES=<stable GPU UUID>`で対象を1台だけ
可視化し、serverには論理device 0を渡す。HIP current deviceはthread-localなので、複数GPUを可視化したまま
global physical indexをworkerへ渡す構成は初期対応外である。

### First-class low-bit model input

Phase 16Fではsource containerを`QuantizedTensorEncoding`、logical tensor role、value/scale plane、source range、
mixed recipeへlowerする。Unsloth compressed-tensors importerと将来のGGUF readerはこの同じ境界を生成し、executorは
containerを見ない。primary Gemma recipeはMLP W4A4、attention W8A8、static FP8 KV、BF16/ignoreを既存Gemma graphへ
bindする。W4A16、FP16 KV、requestごとのBF16 weight展開、別provider fallbackを正常系に置かない。

CLI/serverはBF16とprovider low-bit artifactを同じmodel-directory引数から自動判別し、低bit専用mode、許可flag、確認、
通常警告を設けない。runtime/model evidenceの`experimental`分類は内部audit/documentationに留める。artifact source identity、
recipe digest、topology plan、exact targetをresident identityに含め、異なるrecipe間でresident cacheを共有しない。

### Qwen3.5 sparse MoE

Phase 19はQwen3.5-35B-A3Bのcontainer inputをstrict artifact inventory、config、tensor index、support-file lockから
container-neutralな40個のimmutable layer blobへlowerする。各layerはBF16 routerでstable top-8を選び、選択された
OCP MXFP4 routed expertだけをdecodeでは8 pair、prefillではexpert別にgroup化したactive pairとしてHIP実行する。
shared expertとsigmoid gate、weighted combineも同じordered queueへ接続し、host routing、256 expert全件実行、
requestごとのexpert upload、CPU fallbackを正常経路にしない。

artifact検証は使用時のidentityまで拘束する。全weight shardはhash確認に使ったopen descriptorをverified ownerが保持し、
execution uploadはそのdescriptorからpositional readする。config/index/support fileもopen後にbounded readし、descriptorとpathの
device/inode/size/mtime/ctimeを読み込み前後で照合するため、検証後のpath置換や同一inode mutationを有効なmodel inputとして扱わない。

`SparseMoe`はmodel-neutral semantic opであり、Qwen adapterは40層の3 GDN + 1 full-attention schedule、GQA 16/2、
hidden 2048とmodel固有weight mappingだけを所有する。resident ownerが22,009,574,016 byteのmodel allocationを一度構築し、
request ownerはroute metadata、state、workspaceだけを持つ。auditはlayerごとのSparseMoe submissionとactive pairを数え、
3-token prefillの40/960、1-token decodeの40/320からの逸脱をfail closedにする。CLI/serverはmodel directoryから自動検出し、
MoE用flag、low-bit opt-in、通常警告を追加しない。vision/MTP、batching、expert/tensor parallel、CPU offloadはこの経路の範囲外である。

### Qwen3.5 MTPとvision

Phase 17のMTPはQwen固有の15 tensor manifest/graphをmodel-neutralなspeculative decisionとopaque transactionへ接続する。
Phase 18ではtarget candidate列をM=2..8のserial-equivalent blockへlowerし、各rowのlinear reduction/roundingを通常M=1と同じにした。
draftは公開stateを直接更新せず、最初のrejectまでのaccepted prefixとreplacement tokenだけを既存one-token generation loopへ渡す。
KVとlinear-attention stateはblock単位のopaque rewind後にaccepted input prefixだけをtarget pathでreplayし、上位層はencoding別rollbackを持たない。
数値target blockはM=8まで保持するが、generation transactionのdraft widthはrecurrent stateが保持する一世代のrewind範囲に合わせて1/2、
通常auto-selectionは性能確認済みのwidth 1だけとする。

通常runtimeのprovider選択はwire modeではない。fixed Qwen3.5-4B BF16 text-only greedyのexact `gfx1201`だけ、反復性能tableに基づき
draft width 1を内部選択する。`gfx1030`、量子化target、vision、sampled request、未計測tupleは同じCLI/API操作のままtarget-onlyを選ぶ。
sampled requestはpublic target sampler/RNGを唯一の選択経路として保持し、draft用RNGやresidual samplerを導入しない。

visionはtext residentと別のlazy `QwenVisionResidentModel`を持つ。text-only requestはvision 297 tensorをdeviceへloadしない。
画像requestはbounded decoder/processorを一度実行し、patch projectionと24 vision block、merger/projectorのdense演算を既存HIP
semantic Matmulへ下ろす。画像埋め込みはtyped multimodal promptでimage-pad runだけを置換し、3-axis mRoPE positionをtext graphへ渡す。
decodeでは通常token embeddingを使い、vision encodeを再実行しない。初期実装はBF16 text artifactだけをvisionと組み合わせ、
vision weightのlow-bit化とcross-request image cacheは後続範囲とする。

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

## Phase 42 transport-independent inference modes

Phase 42のpublic endpoint adapterは、既存のChat Completions runtimeや
Phase 40のchoice/sampler stateを複製しない。HTTPとCLIは同じ
transport-independent frontend serviceへlowerし、そのserviceがverified
tokenizer、renderer、model-lock identityを一度だけ解決する。`/v1/tokenize`、
`/v1/detokenize`、`/v1/apply-template`、`/v1/input-tokens`はmodel
execution/GPU executionを起動せず、model-default special-token policyを
使う。templateを使う場合はrenderer version・digest・sizeを結果identityへ
含める。未検証template、任意Jinja、custom kwargsは
`unsupported_parameter`で拒否する。

`/v1/completions`と`/v1/infill`のwire requestは共通のbounded generation subsetへlower
する。stop、usage、streaming、`n` choiceは既存の共通state machineを使い、
endpointごとに別のsamplerやchoice accountを作らない。infillだけが追加で
model-lock capabilityを要求し、verified FIM prefix/suffix/middle token IDs、
template digest、context limitが揃わなければ生成前に拒否する。productionで
未確認モデルをgeneric completionへfallbackしない。

Embedding executionはfinal hidden rowsをhost-sideの固定profileへ渡す。
Phase 42 v1のpoolingは算術平均、accumulatorはF64、出力は有限F32、
normalizationはL2のみであり、dimensionはmodel lockのhidden sizeに結合する。
multimodal rows、client-selected pooling/normalization、non-finite値、
dimension mismatchはprofile boundaryでrejectする。Rerankは同じL2-normalized
vectorのdot productをscoreとし、高いscoreを先に、完全tieは元のdocument
index順に並べる。`top_n`は範囲外をclampせずrejectし、空文書・dimension mismatch・
non-finite scoreを生成経路へ渡さない。

The fixed public-reference pins are OpenAI OpenAPI `2.3.0`, commit
`117ce5680e4269f6656a4fd70d28f9755630d938` and llama.cpp `b10453` commit
`3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`. They constrain adapter tests and
semantic comparison only; llama.cpp aliases and implementation-specific
options do not become sLLM runtime contracts. Exact MI300X `gfx942` execution
is deferred until a fresh runtime is available. Host tests or compile-only
evidence never promote a production GPU capability.

## Phase 43 protocol adapters and tool boundary

Responses and Anthropic Messages lower into the same bounded scheduler and
`ChatCompletionRequestV1`; they do not create provider-specific generation
loops. Tool definitions, choice, ordered message/call/result items and parallel
policy live in `sllm-frontend`. A fixed Qwen renderer escapes the complete
untrusted history and definitions, emits a thinking-disabled assistant prefix,
and couples the request to a Phase 40 JSON-Schema grammar for the canonical
message-or-tool-call envelope. Grammar compilation and all count/byte/capability
checks happen before scheduler/GPU admission. The Qwen production backend
advertises the capability explicitly; an unadvertised backend such as the
current Gemma path rejects tool requests without fallback.

The server adapter owns only wire parsing, profile-specific IDs/events/errors,
usage and stop mapping. Responses and Anthropic serializers are separate closed
state machines, while cancellation, timeout, single-owner model execution and
replay storage remain the Phase 39 runtime. Assistant prefill is accepted only
by the Responses no-tool subset and primes grammar/stop state without
republishing the prefix. Anthropic prefill and thinking blocks are rejected.
Generated deltas are UTF-8-safe 16 KiB fragments. A resumable request is
admitted only through 40 output tokens, derived from the 128-byte token-piece
bound and worst-case JSON escaping, and preflights
the complete named-event batch against the configured event count and Phase 39
64 KiB/event and 256 KiB/session limits before publishing success events.

No executor is reachable from this path. A generated call is serialized to the
client, and a later client request may return its result as untrusted history.
The server does not resolve tool names or touch process, network, filesystem,
environment, credentials, MCP, hosted tools, workers or sandboxes. Any such
execution remains the separately approval-required Phase 47 boundary.

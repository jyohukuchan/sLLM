# Phase 6: KV memory方式選定とOpenAI-compatible Chat Completions profile v1履歴

## 2026-08-11: 計画作成

- Phase 6のprofile v1実装をHTTP外形だけでなく、共有generation service、sampling、stop文字列、usage、SSE、
  disconnect cancellationまで含む作業として計画した。
- llama.cpp固定commitからsamplerとprofile該当testをprovenance付きでport/adaptする候補を定めた。
- vLLM、SGLang、TensorRT-LLM、LMDeployはno-copyのfacts-only readerとした。
- Rust HTTP runtimeはaxum/tokio系列の2026-08-11時点のMSRV互換versionを候補とし、実装前に
  exact dependency closureをoffline policyへ固定することとした。
- API server、dependency追加、direct import、GPU testはまだ開始していない。

## 2026-08-13: AMD vAttention PoCをPhase 6の最優先へ変更

- ユーザー指示により、Paged Attention/vAttentionの選択を後続API・scheduler作業より先に行うよう
  Phase 6を再計画した。
- A0として、PyTorch、full model、model weightを使わないstandalone C++/HIP PoCを最初に置いた。
  VMM capability/granularity、最大logical capacityのVA予約、physical pageの必要時commit、page境界の
  write/read、unmap/remap、in-flight lifetime、latency、resource cleanupをcanonical V620 `gfx1030`と
  R9700 `gfx1201`で確認する。
- ROCm 7.14.0の直接queryでは両canonical GPUとも
  `hipDeviceAttributeVirtualMemoryManagementSupported=1`だったが、この結果だけでは採用せず、
  実際のreserve/map/access/unmapと数値・latency evidenceをA0で取得することとした。
- A1で機能再現性、target間一貫性、granularity、page activationの償却、既存kernel変更量、将来拡張を
  比較して初期方式を一つ選ぶ。vAttention不成立時はPhase 6全体を停止せずPaged Attentionへ切り替える。
- scheduler/serviceにはopaqueなKV allocation/lease/viewだけを公開し、VMM handle、device pointer arithmetic、
  block tableを上位層へ漏らさない。full continuous batching、prefix cache、KV量子化は引き続きPhase 6の非対象とした。
- Microsoft vAttentionはCUDA/A100中心の公開研究実装でありAMDでの証明ではないため、非llama sourceとして
  no-copyの技術参考に限定する。
- この時点では計画変更だけであり、PoC source、production KV変更、API server実装はまだ開始していない。

## 2026-08-13: A0 AMD vAttention再現性PoC完了

- MITのstandalone C++/HIP source `ci/tools/vattention_a0_probe.hip.cpp`とfail-closed runner
  `ci/tools/run_vattention_a0.py`を実装した。runnerはAMD-SMIのHIP UUID、BDF、physical HIP indexを照合し、
  target別Code Object V6/wave32 binaryを一時directoryへbuildして直列実行する。binaryとraw reportは追跡しない。
- host contract 10件はcanonical probe、target/BDF substitution、fallback、境界欠落、non-sparse commitment、
  cleanup shortfall、AMD-SMI mapping、ECC異常を検査してPASSした。suite/path matrixへoffline H0として登録した。
- ROCm 7.14.0、kernel `6.17.0-35-generic`のcanonical V620 `gfx1030`
  (`0000:03:00.0` / `GPU-76a08c022586fed6`)とR9700 `gfx1201`
  (`0000:07:00.0` / `GPU-a8e9ddefa2d60f55`)で、次を両方PASSした。
  - `hipMemAddressReserve`、`hipMemCreate`、`hipMemMap`、`hipMemSetAccess`、kernel read/write、
    event synchronize、middle-page unmap/remap、handle/address release。
  - 3 physical pageを一つのcontiguous pointerとして扱うbyte-exact CPU oracle。CPU/`hipMalloc` fallbackなし。
  - minimum 4 KiB、recommended/selected 2 MiB。Qwen3.5-4B full-attention K/V 16 region、4096 tokenの
    logical reserve 128 MiBに対し、最初の1024 token pageを各regionへcommitしたphysical bytesは32 MiB。
    virtual reserveだけのphysical deltaは0 byteで、cleanup後は計測前free bytesへ復元した。
  - token境界1023/1024/1025と非整列37、pre/postのECC uncorrectable/deferred 0、対象process残留なし。
- 16 region activationの5 warmup + 101 measured結果は、V620がp50 508.199 us / p95 582.841 us、
  R9700がp50 452.418 us / p95 496.488 usだった。deactivationはV620がp50 474.338 us /
  p95 546.910 us、R9700がp50 769.224 us / p95 846.735 usだった。個別操作のp50/p95は、
  V620でcreate 243.344/306.476 us、map 71.411/81.972 us、set-access 190.243/204.134 us、
  unmap 325.685/395.217 us、release 137.723/159.402 us、R9700でcreate 244.424/282.515 us、
  map 72.861/78.212 us、set-access 131.452/169.493 us、unmap 606.401/674.102 us、
  release 159.473/197.574 usだった。
- pre/post ECCとprocessは健全だった。R9700のpost snapshotは11 W、edge 32℃、hotspot 33℃、ECC 0、
  processなしでもlegacy `throttle_status=THROTTLED`を返した。これはsoftware compatibilityに記録済みの
  無負荷でも交互に変わる非actionable fieldであり、A0のVMM correctness判定には使わない。
- probe source SHA-256は`27bed6379e82e9caf5ee711f29012fbf4671aa757544727db412b5371553a9af`、
  target binary SHA-256はV620 `1f05fa692e5c2ed064da5a04f45bfb7373b13d256442aea836be0b87b230f29f`、
  R9700 `217f02e468dfffaf261c6b9a25546dcd63de6f81149894c9d893dcaacf181279`である。local aggregate
  SHA-256は`1cf7a93b6c3ca4cba976bf5eb08be372cfde50d4327b4e4d17946405fb345256`だった。
- A0の受入条件を満たしたためA0を完了する。これはAMD上のvAttention型primitiveの再現可能性を証明する
  draft evidenceであり、production方式の採用決定、full model correctness、continuous batching、別target、
  長時間性能を主張しない。次はA1でPaged Attentionとの方式選択とKV memory契約を確定する。
- 2026-08-13のユーザー方針により、A1は現行baselineに対する改善判定よりも、vAttention上の連続KV
  FlashAttention系kernelとPaged Attention上のblock-table kernelの比較を優先するよう再計画した。
  local ROCm 7.14.0の`amdrocm-ck7.14`にはCK Tileのcontiguous/paged-KV FMHA headerが存在するが、
  Qwen3.5の16 Q head、4 KV head、head dim 256とexact `gfx1030`/`gfx1201`でのinstance可用性は未検証である。
  A1では先にcompile/dispatch feasibilityを確認し、利用不能なtargetでは同一tiled online-softmax実装の
  accessorだけを変える`FA2-style proxy`でlayout差を測る。upstream FlashAttention-3/4はNVIDIA固有実装のため
  AMD実測値を作らず、公開algorithm/interfaceに基づく設計比較として明示的に分離する。

## 2026-08-13: A1 vAttention選択と最小production path完了

- vAttentionはKV memory management、FlashAttentionはattention kernelであり排他的ではないと整理した。
  virtual-contiguous KVはkernelへ通常の連続pointerを渡すため、contiguous-KV FlashAttention系kernelを
  block-table対応へ変更せず利用できる。初期方式はcanonical V620 `gfx1030`とR9700 `gfx1201`に限定した
  HIP VMM virtual-contiguous方式（vAttention型）に決定した。
- facts-only identityをFlashAttention HEAD `145b1010051dbfd4bdc41a0ae55d495b08d7a458`、v2.8.3
  `060c9188beec3a8b62b33a3bfa6d5d2d44975fab`、Microsoft vAttention HEAD
  `ef3fff25dbe4e10f5897da8648718c53df6a20ea`、ROCm AITER HEAD
  `ef7dd32ca159e86b24f51447dbc9868d0aad7d1b`、v0.1.13
  `cdcfa833bdf554ca75594c90dde4316ea9b50199`、local vLLM
  `568afb3a13806beb53bb2e6bd518269357b237c0`へ固定した。いずれもno-copyである。
- local `amdrocm-ck7.14` 7.14.0-3のcontiguous/paged-KV FMHA headerは確認したが、exact
  Q heads 16、KV heads 4、head dimension 256、`gfx1030`/`gfx1201`を選択するprebuilt/generated
  instanceと安定したdispatch経路を確認できなかった。従って実測は独立実装の同一tiled online-softmax
  `FA2-style proxy`を使い、upstream FA2/CKの性能値とは主張しない。
- comparison probeはBF16 Q/output、FP16 token-major K/V、Q length 1/37、KV length
  255/256/257/1023/1024/1025、contiguous/vAttention/pagedの36 caseを各targetでwarmup 3 + measured 9実行した。
  NumPy float64 oracleのabsolute tolerance 0.016、mode間0.004を全caseが満たし、最大mode間誤差は
  V620 0.000732421875、R9700 0.000946044921875だった。fallbackなし、pre/post ECC 0、process残留なしだった。
- 代表Q=37/KV=1025のkernel p50/p95はV620でcontiguous 7700.648/7761.488 μs、vAttention
  7697.567/7765.328 μs、paged 9271.974/9529.896 μs、R9700でcontiguous
  1330.646/1344.446 μs、vAttention 1303.727/1336.647 μs、paged 1897.409/1972.049 μsだった。
  vAttention p50はpaged proxyよりV620で約17.0%、R9700で約31.3%短い。通常allocation比の高速化や
  upstream kernel性能は判定せず、同じcontiguous addressingを維持できることを採用根拠にした。
- 16 MiB logical K/Vに対するvAttention commitmentはKV 255〜1024で4 MiB、1025で8 MiBだった。
  page growは約74〜221 μsで1024 token境界へ償却できる。A0の32 MiB/16 regionというmemory換算は
  token-major配置を前提にしたmodel-free extrapolationであり、当時のhead-major production配置とは一致して
  いなかった。A0のVMM primitive evidence自体は有効で、A1でproduction storageをtoken-majorへ移行したことで
  このintended layoutと一致した。
- public C ABI KV create/viewをversion 2へ更新し、virtual-contiguous memory kind、token-major layout、
  physical page bytes、mapped token capacity、K/V committed bytesを明示した。native stateがVA、physical handle、
  mapping、event lifetimeを所有し、Rust scheduler/serviceはopaque state/resourceだけを扱う。
- actual production runtimeへreserve-only create、append前grow、cancel時非publication、release時
  unmap/handle release/VA freeを接続した。host contractは1023→1024→1025 append、2/2/4 MiB per-plane、
  idempotent cancel、cleanupを確認した。両exact GPUのproduction probeも同じ境界、全1025×4×256要素の
  BF16→FP16 oracle、未map readback拒否、fallbackなし、cleanupをPASSした。
- comparison source SHA-256は`9dbd91d2bf3c30bad505506ace62f95b324618c7afa9b02f2dec586e00c8bd9e`、
  production probeは`3b973081b9d27acf1d6baccc0c2073af4846c814732257effb8aec3363505862`、
  production KV kernelは`85a4ac25ff7ee1489e0765c1b090e71485ab21b6372a2409e7eff43c3c720047`、
  local aggregateは`453756b16f55ef81ff28dcb48cdebe69b9bdd83381b3a04202f94855af236021`だった。
- FlashAttention-3/4はそれぞれHopper、Hopper/Blackwell向け公開実装のため、AMD実測ではなく設計比較に限定した。
  prefix sharing、continuous batching、VMM非対応target、mapping latency支配、VA/driver制約、paged-only backendの
  明確な総合優位を再検討条件とした。Paged Attention production backendは実装していない。
- A1を完了し、次のPhase 6作業をA2のprofile drift、reader、dependency、provenance設計へ進める。

## 2026-08-13: A2 profile drift、reader、dependency、provenance設計完了

- normative OpenAPI commit `117ce5680e4269f6656a4fd70d28f9755630d938`とcurrent main
  `11854aef674352d3f9cd5c0a7038f079a7bbac06`を再帰的なschema closureで比較した。
  `POST /v1/chat/completions`は両側71 schema、`GET /v1/models`は両側2 schemaで、request、response、
  SSE stream、error shapeに差はなかった。唯一の差は`ModelIdsShared` enumへの`gpt-5.5`追加であり、
  sLLMのserved alias方式へ影響しないためprofile pinは更新しなかった。
- llama.cpp `f5919bf458ef190468b5c329bb293f8a54a1e69c`について、sampler source/header、sampling test、
  Chat Completions/stream/security test、server test utilityの7 unitをexact blobとSHA-256へ固定した。
  profileに必要なsampler部分はported、選択testはadapted、HTTP harness utilityはfacts-onlyとした。
  actual importはまだなく、最初のimport時にsource header、notice、local SHA-256、変更、import commitを記録する。
- vLLM `568afb3a...`、SGLang `fdebc938...`、TensorRT-LLM `376f7e1b...`、LMDeploy `f4b8140b...`の
  exact pathをfacts-only readerへ固定した。implementationへ渡すのはvalidation、共通generation result、
  SSE順序、usage、disconnect、error時点、backpressureに関する技術的事実と8 acceptance caseだけである。
- `crates/sllm-server`を追加し、axum 0.8.9、tokio 1.53.1、tower-http 0.7.0、tokio-stream 0.1.19、
  futures-util 0.3.33、serde_path_to_error 0.1.20のexact direct dependency/featureを固定した。
  Cargo closureは132 package、126 registry package、6 workspace package、308 edgeで、Rust 1.85.0 Linux上の
  locked/offline metadata/checkとpackage checksum/license/MSRV/feature policyで再現する。
- `ci/contracts/phase6-a2-v1.json`、schema、validator、negative testを追加し、H0 suite/path matrixへ登録した。
  local llama.cpp source hash確認、A2 negative test、Rust dependency policy、JSON/schema/matrix validatorをPASSした。
- A2の受入条件を満たしたため完了する。次はA3のtransport非依存generation serviceとsamplingへ進む。

## 2026-08-14: A3 transport非依存generation serviceとsampling完了

- `sllm-frontend`へ`GenerationServiceV1`を追加し、CLIのrender/tokenize/prefill/decode/sampling/stop/usageを
  transport非依存の単一loopへ移した。requestごとの`QwenExecutionRequest`がA1のopaque KV ownerを保持し、
  CLI/server層へVMM pointer、handle、block tableを公開しない。
- Qwen executionへoptional terminal logits readbackを追加した。sampling時だけBF16の最終vocabulary rowを
  backendの`max_transfer_bytes`以下に分割してf32へ変換し、temperature 0は既存device argmaxだけを使用する。
- llama.cpp `f5919bf458ef190468b5c329bb293f8a54a1e69c`からtemperature、top-p、presence/frequency
  penaltyをRustへportedし、tiny-logit/boundary testをadaptedした。`THIRD_PARTY_NOTICES.md`、source/test header、
  exact blob/SHA-256、保持MIT licenseを追加した。import commitは開発中のpending markerでありrelease時に解決する。
- incremental UTF-8 stop matcherはtoken/UTF-8境界を跨ぐpartial matchを保留し、stop文字列をvisible outputへ
  漏らさない。共通resultへprompt/completion/total usage、`stop`/`length`、generated/visible token差を追加した。
- public seedはprofile外として追加せず、host testだけがexplicit RNGを注入する。cancellation、sampling error、
  decode errorはrequest ownerだけを再利用不能にし、resident ownerを維持する。
- sampler unit 6件、generation unit 7件、CLI unit 19件/process 3件、shared fixture Rust/NumPy oracle、format、
  provenance validatorをPASSした。Rust dependency closureは新規test targetをmanifestへ同期したが、package 132、
  edge 308、direct dependency集合は不変である。
- V620 `gfx1030`、ROCm 7.14.0、Qwen3.5-4B lock
  `sha256:f143d7b504170d071c77818105f7a07dc0297c6bea0c61a5404b071fed0c1fae`でprompt `Hello`、
  max 1のfocused smokeを実行した。greedyとtemperature 1/top-p 1はいずれもtoken 11、HIP dispatchのみ、
  fallbackなし、usage 1/1/2、request cleanupとGPU process残留なしでPASSした。性能比較は行っていない。
- A3の受入条件を満たしたため完了する。次はA4のmodel registry、strict JSON、non-stream endpointへ進む。

## 2026-08-14: A4/A5一括実装、A6分離を決定

- A0〜A3が想定より早く完了したことを受けてA4〜A6の作業境界を再評価した。
- A4 non-streamとA5 SSEは同じstrict DTO、model registry、bounded scheduler、generation event、
  error mapping、usage/finish resultを共有するため、一つの実装バッチとして進めることにした。
  ただし受入条件はA4/A5ごとに維持し、一方の未達を一括完了へ埋没させない。
- A6は外部compatibility fixture/differential、official client、canonical GPU full-model smoke、
  VMM cleanup、service overheadという実行時間とevidence identityの異なる統合工程なので分離する。
  A4/A5のhost contract完了後に開始し、A4/A5の通常の実装反復でdual-GPU evidenceを取り直さない。

## 2026-08-14: A4/A5 host API実装完了

- `sllm-server`へserved aliasと`sha256:<64hex>` model-lock fingerprintを結合するregistry、duplicate JSON memberも
  fail-closedにするstrict request parser、field path付きvalidation、optional bearer auth、1 MiB body limit、
  Content-Type検証、OpenAI型error envelopeを実装した。
- `GET /v1/models`、non-stream Chat Completions、assistant role/content/final usageを送るSSE、exact `[DONE]`を
  共通generation eventから構築した。header後failureはerror envelopeを一つのSSE data eventとして送り、
  finish chunkと`[DONE]`なしでcloseする明示的なsLLM conventionとした。
- schedulerは1 active + bounded FIFO、bounded per-request event channel、429 admission、request timeout、shutdown、
  receiver drop cancellationを持つ。slow consumerはbackend sinkをblockし、disconnectまたはsink failureは
  request ownerをcancelするがmodel registry ownerを破棄しない。
- A3 generation serviceへvisible output sinkを追加した。incremental UTF-8/stop matcherが確定したnonempty textだけを
  publishし、sink failureではexecution requestをcancelする。既存CLI/non-stream semanticsはno-op sinkで維持した。
- server unit 10件、raw TCP integration 7件、frontend unit 34件を含むworkspace Rust test、format、clippy、MSRV
  dependency closure、JSON/matrix、markdown linkをPASSした。negative matrixはtools、developer/tool/function role、
  multipart、response_format、logprobs、seed、n≠1、unknown/duplicate field、range両側を含む。
- OpenAI Python SDK 3.0.0をfixture serverの`base_url`へ接続し、models、non-stream、streamをPASSした。
  streamはassistant role、`fixture response`、stop、3/2/5 usageを4 chunkとしてparseした。
- ここで完了したA5 evidenceはhost cancellation/task/channel cleanupである。pending HIP work、request-local
  KV/linear state、VRAM、GPU processのfull-model cleanupは、決定したbatch境界どおりA6の両canonical GPU
  integrationでfail-closedに確認する。

## 2026-08-14: A6 compatibility、differential、GPU integration完了

- official OpenAPI固定commit `117ce5680e4269f6656a4fd70d28f9755630d938`からprofile subset fixtureと
  schema/validatorを作り、positive non-stream/SSEとmalformed JSON、tools、n、role、multipart、unknown modelの
  negative caseをH0/H1へ登録した。fixtureを直接使うraw HTTP/SSE contractもPASSした。
- llama.cpp `f5919bf458ef190468b5c329bb293f8a54a1e69c`のChat Completions、stream、security testから
  profile該当caseだけをadaptした。exact blob、source/local SHA-256、MIT license、変更内容は
  `THIRD_PARTY_NOTICES.md`の独立noticeに記録した。HTTP harness構造とllama.cpp独自fieldは移植していない。
- vLLM `568afb3a13806beb53bb2e6bd518269357b237c0`とSGLang
  `fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1`は実serializer、llama.cppは固定binaryとGGUFの実HTTP/SSEで
  共通subsetを比較した。3 peerともPASSしたが、仕様oracleはpinned OpenAI profileだけである。llama.cppの
  `system_fingerprint`と`timings`は非互換ではなくpeer固有extraとして記録した。
- production `QwenChatBackendV1`はverified lock/cache、tokenizer/template、weight plan、exact HIP session、
  `QwenResidentModel`を一度loadし、既存`GenerationServiceV1`をstrict registry/schedulerへ接続する。API用Qwen
  templateはthinking-disabledを明示し、Phase 5 render契約と一致させた。workspace closureは132 package、
  309 edgeとなり、package集合を変えずserver→`sllm-hip`のproduction edgeを追加した。
- 初期serverは単一GPU・単一model-resident instance・1 active requestである。HIP current deviceはthread-localのため、
  mixed-GPU hostでは対象UUIDだけを`ROCR_VISIBLE_DEVICES`へ指定して論理device 0を使う。複数GPU可視のまま
  global physical indexをworkerへ渡す構成は初期対応外とした。
- canonical V620はUUID `GPU-76a08c022586fed6`、BDF `0000:03:00.0`、exact `gfx1030`、R9700は
  `GPU-a8e9ddefa2d60f55`、`0000:07:00.0`、exact `gfx1201`として単独可視化した。Qwen3.5-4B lock
  `sha256:f143d7b504170d071c77818105f7a07dc0297c6bea0c61a5404b071fed0c1fae`、plan digest
  `sha256:0474ed893fbc043c3ace0197515f8d99e27fe3d28a4844fbdd9781bb9d30c7fa`で、raw non-stream/SSE、
  stop、OpenAI Python client 2.44.0、HIP dispatch後disconnectとrecoveryを両targetでPASSした。
- full-model serviceでlogical capacity 1023/1024/1025を確認した。physical pageは2 MiB、8 full-attention layerの
  K/V committed合計は一active pageで32 MiB、linear-attention layerは24だった。disconnect probeは936 HIP
  submission後にcancelされ、request-local KV/linear/workspaceを0へ解放した。completed requestは全てHIP-only、
  fallbackなしで、shutdown後tracked allocation 0、pre/post GPU process 0、ECC/health正常だった。
- Phase 5 render matrix revision 1の`chat-hello`（message `Hello`、thinking disabled、13 input、17 output）を
  service経由で再実行した。V620はbackend 17.043882049秒 / HTTP 17.044670008秒 / JSON residual 0.787959 ms、
  R9700は7.656757501秒 / 7.657290528秒 / 0.533027 msだった。1-token caseのnon-stream JSON residualは
  V620 20.044258 ms、R9700 20.272576 ms、SSE residualは1.001370/0.839829 ms、同時2 requestのqueue wait
  residualは7.605346579/2.880756600秒だった。queue値は1-active FIFO待ちを含み、steady-state性能ではない。
- final binary SHA-256はV620 `dd07bca6c1ca023365bc8800142302929ee50495993e431843aa35528b81723c`、
  R9700 `029fdf71f5899200915f1f8a5161316c6f9832f85dbb3ea9a7ddc188c677067b`、local report SHA-256は
  V620 `b8ad41a3f35c693b98fc6629e5997413726fb8e9ad8dc16de21a49c20a874d8f`、R9700
  `0648e41bb3a92ac60b82223a15b8ef2540ec9db7354da0ba29ecb5bf8c1f845f`である。binary/report/raw healthは
  repositoryへ追跡せず、digestと要約だけを履歴へ保持する。
- H0/H1 schema/dependency/provenance/API matrix、official client/raw SSE、両canonical GPUのexact HIP/no-fallback、
  disconnect cleanup、service overheadというA6受入条件を全て満たしたため、A6とPhase 6を完了した。

## 2026-08-14: Phase 6 completion audit

- `cargo test --workspace --locked --offline`、workspace clippy `-D warnings`、Rust/C++ format、OpenAI fixture、
  license/provenance、Rust dependency、JSON/schema/matrix、Markdown linkをPASSした。
- registered H1は364/364 selectedをPASSした。registered H0の修正後runは495/496をPASSし、唯一の失敗は
  C++整形後に`hip-runtime-compile-v1`のsource inventory hashが古いことだった。source hashとcanonical set
  digestを同期し、失敗したJSON/schema/manifests commandだけをfocused rerunしてPASSした。変更していない
  495 caseを再々実行せず、integration findingのfocused re-review方針に従って累積496/496と判定した。
- C++整形後に両targetのrelease binaryを同じtoolchain/build inputで再buildし、SHA-256がGPU evidence時の
  `dd07bca6...` / `029fdf71...`から変わらないことを確認した。従って既存reportが検証したartifact mappingは
  維持される。
- A6 item 1〜7と受入条件5件を個別に再監査し、未達、silent fallback、zero selection、未文書化claim、
  active Phase 6 planの残留がないことを確認した。

[対応する計画](../../../../plans/archive/2026/08/11-20/openai-chat-completions-v1.md)
[KV memory decision](../../../../architecture/kv-memory.md)

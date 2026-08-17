# Phase X: Qwen3.5系GDNのllama.cpp AMD性能調査・修正・sLLM還元

> 状態: completed (`fixed`)
> 作成日: 2026-08-17
> 完了日: 2026-08-17
> Phase割当: 数値roadmapから独立した調査・修正phase。Phase 20のGGUF統一と並行または前後どちらでも実行できる。

## Phase割当の理由

2026-08-17のlocal検証で、Qwen3.5 architectureを使うQwen3.8-27B GGUFはllama.cppのHIP backendだけが
同じGPUのVulkan backendより大幅に遅く、特に長いprefillでprompt長に応じた崩れを示した。これはQwen3.8だけの
model追加作業ではなく、Qwen3.5系Gated DeltaNet（GDN）、HIPの小GEMM/dispatch、MTP、長context memoryという
共通実装境界の問題である。

sLLMはPhase 9でllama.cppのGDN recurrent-state layoutを直接adapt済みであり、今後もllama.cppからの直接reuseを
許可している。したがって本件を単なる外部benchmark調査にせず、upstreamの原因と修正候補を固定し、llama.cppへの
patch、sLLMへのimport/port、sLLM固有実装のどれを採るかまでを一つの独立Phaseとして扱う。

`Phase X`は製品形式を扱うPhase 20の範囲を広げない。`X`は数値roadmapから独立した横断調査・修正を表し、
Phase 20の完了、skip、順序変更、またはGGUF実装への依存を意味しない。

## 目的

1. Qwen3.8-27B/Qwen3.5 architectureのHIP性能低下を、GDN prefill、GDN decode、MTP、KV/context memory、
   harness/API overheadへ定量分解する。
2. 最新llama.cppで再現するHIP固有の原因をprofileと切替実験で特定し、最小の修正候補を実装・評価する。
3. upstream issue/PRの適用可能範囲を確認し、既存patchの取込み、upstream向け修正、sLLMへの直接reuse/port、
   sLLM固有実装のいずれかを根拠付きで決定する。
4. sLLMのQwen3.5 dense/MoE/MTPに共通するGDN operator、provider routing、prepared executionへ有効な変更だけを
   provenance付きで還元し、exact `gfx1030`/`gfx1201`のHIP pathを改善する。

## 開始baseline

### 固定artifactと実行条件

- model: `/home/homelab1/datapool/ai_models/gguf/qwen3.8-27b-UD/Qwen3.8-27B-UD-Q5_K_XL.gguf`
- size: `20,218,178,624` bytes。A0でSHA-256、GGUF tensor inventory、metadata digestを追加固定する。
- architecture: `qwen35`、main block 64 + MTP block 1、native context `262,144`、full-attention interval 4。
- GDN: state size 128、inner size 6,144、group count 16、time-step rank 48、conv kernel 4。
- attention: 24 query heads、4 KV heads、key/value head dimension 256。
- llama.cpp開始source: commit `4df29be4f4c3673f428170fda944a5b19f743bb8`、build 901。A0で
  upstream `master`を再取得し、実際に開始する完全commitを新しいsource lockとして固定する。
- primary request: 約9.4k prompt tokenの実Python code作成。random token列はMTP acceptanceと実生成decodeの
  性能を代表しないためprimary性能caseに使わない。
- runtime: context 262,144、batch 1、parallel 1、model/draft KV `Q5_1`、MTP draft幅 3、Flash Attention on。
- entrypoint: DeepSeek Harness経由のResponses-compatible requestをprimaryとする。llama.cpp direct CLI/serverを
  controlとして併用し、Responses API自体のsLLM実装を本Phaseへ含めない。

### 2026-08-17の観測値

全rowは同じQ5_K_XL modelと実code-generation promptを使用した。生成はEOS前に中断しており、MTP acceptanceの
最終集計値ではない。値は原因調査の開始baselineであり、修正candidateの最終性能証拠には再利用しない。

| GPU/backend | prefill | decode | 観測範囲 |
| --- | ---: | ---: | --- |
| spare V620 `gfx1030` / HIP | 59.6 tok/s | 約5.2 tok/s | 約2,721 generated tokenで中断 |
| R9700 `gfx1201` / HIP | 68.95 tok/s | 約12.0 tok/s | 約830 generated tokenで中断 |
| spare V620 / Vulkan | 203.69 tok/s | 33.41 tok/s | 約3,168 generated tokenで中断 |
| R9700 / Vulkan | 718.08 tok/s | 48.16 tok/s | 約3,806 generated tokenで中断 |

R9700 HIPのcumulative prefillは2,048 tokenで251.74、4,096で145.87、6,144で103.66、9,377で
68.95 tok/sへ低下した。単なるmodel bandwidth上限より、token/chunk数に比例する大量の小dispatchまたは
host-side solution lookupを第一仮説とする。

### upstream調査の開始snapshot

状態は2026-08-17確認時点であり、A0で再確認する。

| 種別 | upstream | 開始時の判断 |
| --- | --- | --- |
| historical issue | [#18823](https://github.com/ggml-org/llama.cpp/issues/18823) | Qwen3 Next HIP prompt低下。約15kの小GEMM dispatchとhipBLASLt solution lookupが報告され、build 8627で解消したとしてclosed。ただし現Qwen3.8/MTPで再現するため回帰または別pathを疑う |
| duplicate evidence | [#20218](https://github.com/ggml-org/llama.cpp/issues/20218)、[#20292](https://github.com/ggml-org/llama.cpp/issues/20292) | Qwen3.5 ROCm/HIPがVulkanより約10倍遅い事例と、pp128で約14,977 dispatch・wall timeの約99%がdispatch overheadというprofile |
| merged GDN | [#19504](https://github.com/ggml-org/llama.cpp/pull/19504)、[#20366](https://github.com/ggml-org/llama.cpp/pull/20366) | fused GDNとHIP shared-memory path。current sourceに入っている前提をcommit/path単位で検査する |
| merged Vulkan | [#20334](https://github.com/ggml-org/llama.cpp/pull/20334)、[#20662](https://github.com/ggml-org/llama.cpp/pull/20662) | dedicated GDN shaderとsubgroup sharding。Vulkanが速い理由のalgorithm/dispatch比較資料であり、sLLMへVulkanを追加する根拠ではない |
| open CUDA prefill | [#26001](https://github.com/ggml-org/llama.cpp/pull/26001) | chunked GDN prefill。Ampere以降のBF16 Tensor Core向けで、そのままHIPへ適用しないがchunk分割と数値検証を候補にする |
| open Vulkan chunking | [#20377](https://github.com/ggml-org/llama.cpp/pull/20377) | draft/競合中。Vulkan chunked parallel GDNの設計比較だけに使う |
| open MTP | [#26038](https://github.com/ggml-org/llama.cpp/issues/26038)、[#26750](https://github.com/ggml-org/llama.cpp/issues/26750)、[#26432](https://github.com/ggml-org/llama.cpp/issues/26432) | ROCm draft context buffer、backend別acceptance、GTT fallbackを独立原因として切り分ける |
| open KV/context | [#27109](https://github.com/ggml-org/llama.cpp/issues/27109)、[#21831](https://github.com/ggml-org/llama.cpp/issues/21831) | quantized KVとrecurrent/SWA stateの長prompt再処理・prefill低下をcontrolにする |

## 原因仮説と判定方法

### H1: GDN chunked prefillの小GEMM・launch爆発

- `src/models/delta-net-base.cpp`のfused/autoregressive/chunked routingと、実行graphのGDN関連node数を
  prompt長別に数える。
- `rocprofv3`のHIP API/kernel traceとCPU `perf`を同時に取り、GDN layerあたりのGEMM/elementwise/graph launch、
  hipBLASLt heuristic/solution lookup、GPU idle gapを集計する。
- 2,048/4,096/6,144/9,377 tokenのdispatch数とwall timeへ線形または段階的に説明できるか確認する。
- chunk size、`ubatch`、HIP Graph on/off、GDN fused/chunked routingを一変数ずつ切り替え、原因寄与を測る。

### H2: autoregressive fused GDNのHIP decode kernel

- `ggml/src/ggml-cuda/gated_delta_net.cu`をHIPとしてcompileしたpathのkernel時間、occupancy、register/LDS、
  memory transaction、launch数をexact `gfx1030`/`gfx1201`で分ける。
- Vulkan dedicated shaderと同じstate size 128、group/head layout、one-token semanticsでoperator microを比較する。
- CUDA/HIP共有templateのNVIDIA前提、wave32、shared-memory size、row/state mapping、KDA branchを確認し、
  target別guardなしの遅いspecializationを特定する。

### H3: MTP draft/verify/rollbackとacceptance

- MTP off、draft幅1、draft幅3を同じ実prompt・seed・output budgetで比較する。
- drafted/accepted/rejected token、target forward数、visible token/s、draft context buffer、rollback/replay時間を記録する。
- MTP offでも遅ければGDN/backendを優先し、MTP onだけが遅ければdraft graph、acceptance、state rollbackを独立laneへ送る。
- sLLMへの還元ではPhase 18のtarget-only数値同一、逐次承認、accepted-prefix publicationを維持し、llama.cppの
  speculative controlを一括移植しない。

### H4: Q5_1 KV、262k context、VRAM/GTT

- context 32k/262,144、KV F16/Q8_0/Q5_1、MTP off/onを変更し、model resident、KV reserve/commit、compute buffer、
  GTT、device-local VRAMを起動時とprefill中に記録する。
- physical memoryがdevice-localからGTTへ移ったrun、OOM/timeout、silent CPU/backend fallbackは性能PASSにしない。
- Q5_1固有で再現する場合はllama.cpp benchmark issueとして分離し、sLLMの対応KV形式や一般INT量子化方針を
  この調査だけで変更しない。

### H5: Harness/API/tokenization overhead

- 同一token列をllama direct CLI、llama server native endpoint、DeepSeek Harness経由で比較する。
- tokenizer、prompt rendering、HTTP serialization、stream consumer時間をGPU prefill/decodeから分離する。
- direct pathでも同じ低下ならGPU/backend原因、Harnessだけならwrapper修正とし、GDN kernel変更へ誤帰属しない。

## 対象と非対象

### 対象

- 最新llama.cppのQwen3.5/Qwen3.8 dense GDN、MTP、HIP backend、Vulkan比較control。
- exact spare V620 `gfx1030`、canonical R9700 `gfx1201`。必要時にcanonical V620をguardとして追加する。
- ROCm 7.14.0と現在のlocal Vulkan runtimeの完全tuple、llama.cpp source/build option、model/harness lock。
- llama.cppの原因修正candidate、upstream report/PRに必要な最小reproducer、benchmark summary。
- sLLMの`linear_attention.gdn.v1`、Qwen graph/provider、prepared execution、MTP state contractへの還元判断。
- llama.cppから直接reuseする場合のnew provenance event、notice、source header、source-lock/reader記録。

### 非対象

- Vulkan backendをsLLMのsupported/experimental backendへ追加すること。Vulkanは比較controlだけに使う。
- Q5_K_XL、Q5_1 KV、一般的なllama.cpp INT4/INT8+scale形式をsLLM製品supportへ追加すること。
- Responses APIのsLLM実装、DeepSeek Harness一般機能、local subagentのprompt品質改善。
- Qwen3.8の正式sLLM model lock/loader/full integration、Phase 20 GGUF converter/readerの完成。
- CUDA、Metal、OpenCL、別model family、multi-GPU、request batching、長時間安定性の一般claim。
- vLLM/SGLang等からのsource copy/adapt/port。これらは技術的事実と評価方法だけのno-copy referenceとする。

## 固定する実行契約

- benchmark rowはmodel SHA-256、llama.cpp commit、build type/CMake cache、backend、GPU UUID/BDF/exact target、
  driver/runtime/library、context、batch/ubatch、KV type、MTP幅、prompt digest/token count、seed、output limitを記録する。
- backendごとに一台だけをstable UUIDで可視化し、別GPUへのoffload、CPU compute fallback、mixed backendを拒否する。
- primary promptは実Python code作成を固定し、MTPの主性能判定へrandom token列や無意味な反復tokenを使わない。
- prefillとdecodeを別計測し、decodeはvisible output token/sに加えてtarget evaluation、draft/acceptance、
  cumulative/interval token/sを記録する。
- raw trace、binary、model、生成全文はrepository外に置く。Gitにはbounded JSON/Markdown summary、command template、
  digest、採否理由だけを残す。
- latest upstream比較とcandidate patchは別branch/worktree/build directoryに置き、古いbuild artifactを再利用しない。
- sLLMへ変更を入れる場合、semantic op descriptor、opaque request state、transaction、fallbackなし、exact target auditを維持する。

## 受入条件

### correctness/security blocker

1. model、source、build、software/GPU tuple、prompt、実行parameterを再現可能に固定し、row間で意図しない差を残さない。
2. HIP runはexact target、GPU-only offload、loaded ROCm root、kernel/backend auditを記録し、CPU/Vulkan fallback、timeout、
   crash、test未収集をHIP PASSにしない。
3. GDN operator変更はstate size 32/64/128、token `1/2/3/17`、chunk境界`B-1/B/B+1`、非整列head/groupを含む
   独立float64/FP32 oracleへ照合し、outputと次stateの両方を検証する。
4. prefill chunkingはchunkなし/別chunk sizeとの最終state・outputを明示tolerance内で一致させ、state publication順序、
   sequence境界、cancel/error時のpartial stateを公開しない。
5. MTP変更をsLLMへ還元する場合、Phase 18の通常逐次target-onlyに対するtoken/logit/KV/sampling同値と
   accepted-prefixだけのcommitを維持する。
6. VRAM/GTT、compute/draft/KV buffer、workspace、cleanupを計測し、速度改善をsilent host memory fallback、
   全weight展開、capacity縮小、MTP無効化で作らない。
7. llama.cpp source expressionをreuseする場合はreleaseまでに完全commit/path/blob/hash、reuse mode、変更点、notice、
   import commitを記録する。既存`llama-cpp-phase9-gdn-layout-001`を上書きせず、新しいimportを新しいprovenance eventにする。

### 調査完了条件

8. H1〜H5をそれぞれ採用、棄却、残留不確実性のいずれかへ分類し、prefill/decodeのwall timeを主要componentへ分解する。
9. HIP/Vulkan差について、少なくとも一つの切替実験で主要原因の寄与を再現し、profile上のdispatch/CPU/GPU/memory指標と
   end-to-end token/sの変化を対応付ける。
10. current upstream issue/PRをexact commitと状態で再監査し、直接適用可、algorithmだけ参照、無関係、競合/未完成を分類する。
11. llama.cpp修正、sLLM import/port、sLLM固有実装、upstream待ち、Vulkanをlocal subagentの暫定backendにする判断を、
    `gfx1030`/`gfx1201`、prefill/decode、MTP on/offごとにdecision tableへ固定する。

### 修正candidateの採用条件

12. candidateは同一rowのfresh baselineに対し、prefillまたはdecodeの主要指標が測定noiseを越えて改善し、profile上の
    想定componentも同方向に減少した場合だけ採用する。単一の最良run、backend条件変更、output短縮だけで採用しない。
13. 変更対象外のprefill/decode、MTP off/on、32k/262k、F16/Q5_1 controlで説明不能なnoise超過退行を残さない。
14. full 9.4k promptの最終candidateはbaseline/candidateの順序をcounterbalanceし、最低1 warmup + 5 measuredを行う。
    分散が判定を覆す場合だけ最大3 warmup + 10 measuredへ増やし、全candidateへ長いmatrixを繰り返さない。
15. Phaseを「fixed」として完了するには少なくとも一方のHIP targetで採用candidateを得る。採用candidateがない場合も、
    bounded candidate集合を理由付きで棄却し、再現可能なupstream reportとsLLM dispositionを残せば
    `investigated-no-adopt`としてcloseできるが、性能修正済みとは表記しない。

### nonblocking target

- HIPを同一GPUのVulkanに近づけ、少なくとも歴史的Qwen3.5 issueで解消済みとされたdispatch pathologyを現Qwen3.8で
  再発させないことを目標にする。HIP/Vulkan比や絶対token/sの一律hard thresholdは、hardware/backendの実装差と
  現時点のsample不足から本Phaseでは設定しない。
- local Qwen subagentのbackend切替はPhaseの製品acceptanceではない。修正後HIPがVulkanより実用上遅い場合は、
  user-facing wrapperをVulkanへ切り替える別の運用判断を明示して行う。

## 実装・検証順序

### PX-A0: source/model/tuple lockと再現harness

- current llama.cpp `master`、対象issue/PR head、ROCm/Vulkan loader、GPU identityを再取得し、完全commitとbuild optionを固定する。
- GGUF metadata/tensor inventory、SHA-256、chat template、MTP block、native contextをbounded manifestへ記録する。
- DeepSeek Harnessのrequest bodyとdirect llama requestを同じrender済みprompt tokenへ固定し、prompt digestとtoken countを照合する。
- prefill/decode、MTP draft/accepted、VRAM/GTT、backend/dispatchをmachine-readableに出すrunnerをrepository外artifact対応で用意する。
- 最新buildだけをcanonical baselineにし、旧llama.cpp buildは比較・runtime wrapperから外す。削除が必要なartifactは
  exact pathを確認し、recoverableな範囲で整理する。

### PX-A1: fresh four-way baselineと短いablation

- spare V620 HIP/Vulkan、R9700 HIP/Vulkanを同じpromptで再取得する。backend別にmodel loadを分け、foreign GPU workloadを拒否する。
- prompt 2,048/4,096/6,144/約9.4k、context 32k/262k、MTP off/1/3、KV F16/Q5_1を一変数ずつscreeningする。
- direct CLI/server/Harnessを比較し、wrapper overheadを分離する。primary performanceは実code promptのまま維持する。
- 数分のscreeningでgenerated 128 token以上かつdecodeが数tok/s域に留まり、baseline比の改善方向もないcandidateは中断し、
  full outputを待たない。中断runもelapsed、token count、cumulative rate、理由を記録する。

### PX-A2: HIP profileとroot-cause attribution

- R9700をprimaryに`rocprofv3` HIP API/kernel trace、CPU `perf`、GPU利用率/VRAM/GTTを同期取得する。
- GDN layer、chunk、GEMM shape、solution lookup、kernel launch、graph launch、host idle/GPU idleのcount/timeを集約する。
- spare V620でwave32/target差を確認し、R9700固有と共通HIP pathologyを分ける。
- Vulkan traceは同じ意味のGDN dispatch数とshader時間をbounded summaryにし、API固有event名を同一kernelと誤対応させない。
- H1〜H5の寄与tableを作り、最大寄与のlaneだけを最初の修正対象にする。

### PX-A3: upstream deltaとpatch候補の選別

- #18823の修正前後、current `delta-net-base.cpp`、`gated_delta_net.cu`、HIP build optionをdiffし、closed issueの
  修正がQwen3.8/MTP pathへ適用されているか確認する。
- merged #19504/#20366/#20334/#20662のcurrent sourceへの到達をcommit/pathで確認する。
- open #26001/#20377等はisolated worktreeで適用可能性を確認する。CUDA Tensor CoreやVulkan shaderをHIP対応済みと
  読み替えず、algorithm/graph changeとbackend kernelを分離する。
- upstreamで未報告の再現なら、model metadata、commit、command、profile summary、HIP/Vulkan比較、MTP/KV ablationを
  含む最小issueを準備する。外部投稿やPR作成はユーザーの明示依頼がある場合だけ行う。

### PX-H1: llama.cpp HIP prefill candidate

- `build_delta_net` routingをprompt/chunk/target capabilityで監査し、GDNを多数の小GEMMへ展開する条件を最小化する。
- chunk sizeを増やすだけのworkaround、chunked parallel recurrence、dedicated fused GDNの三候補を分ける。
- hipBLASLt solution lookupが支配する場合、problem descriptor/shape/layout/device/library versionをkeyにprepared-timeで
  solutionをcacheし、token/chunk loop内のheuristic探索を除く。unsupported solution時に別backendへfallbackしない。
- HIP Graph on/offを比較し、graphが小dispatchを隠せない、またはcapture/replay overheadを増やす場合は
  target/path guardで選択する。単純に全buildで無効化しない。
- candidateごとにoperator/state oracle、2k→9.4k scaling、full prompt prefill、decode guardを通す。

### PX-H2: llama.cpp HIP decode candidate

- fused `GGML_OP_GATED_DELTA_NET`のstate size 128 specializationをexact `gfx1030`/`gfx1201`でmicro-profileする。
- one-token decodeではstate row mapping、wave-coalesced load、KDA、register保持、shared memoryをtarget別candidateとして比較する。
- Vulkan shaderのwork decompositionはalgorithm参考に限定し、SPIR-V/GLSL source expressionをsLLM HIPへ移植しない。
- decode candidateはMTP offで先に選び、MTP width 3の改善を同じものと仮定しない。

### PX-M: MTP・memory candidate

- #26038/#26750/#26432の該当条件をcurrent HIP/Vulkanで確認し、draft context compute buffer、KV、GTT、acceptanceを分離する。
- MTP width 3がtarget evaluationを減らしてもdraft/rollback overheadでvisible token/sを落とす場合、幅1/3の
  target別選択またはMTP offをllama.cpp/local wrapperの候補にする。これはsLLM Phase 18の内部auto-selectionと別判断である。
- Q5_1だけのprefill collapseはKV dequant/attention laneとしてissue化し、GDN修正へ混ぜない。
- sLLMへMTP制御を還元する場合はdevice-resident draft/verify、accepted-prefix commit等の技術要点だけを採り、
  llama.cpp control flowの一括移植を行わない。

### PX-S0: sLLM reuse/port decision

- sLLM current `native/hip/src/linear_attention_kernel.hip.cpp`、`linear_attention.gdn.v1`、Qwen graph、
  Phase 9 profileとllama candidateを同じoperator semanticsで比較する。
- llama.cpp candidateがsLLMにも有効なら、exact/adapted/portedの区分、upstream commit/path/blob、変更点、local destinationを
  import前に固定する。既存Phase 9 noticeとは別entryにする。
- source expressionを使わずalgorithmだけ採る場合はreader noteへtechnical facts、shape、数値順序、resource、benchmarkを
  固定し、その記録からsLLM固有HIP providerを実装する。
- sLLM providerはbaseline `linear_attention.gdn.v1`を残し、candidateをnew provider ID/symbolとしてregistryへ追加する。
  prepare-time capability/shape guardで選び、runtime failure後のsilent baseline/CPU fallbackを行わない。
- GDN state layout、MTP rewind、model-neutral prepared execution、transaction publication、Phase 19 MoEの共通Qwen GDNを回帰する。

### PX-S1: sLLM focused GPU evidence

- model-free GDN operatorはtoken/chunk/state境界をNumPy/float64 oracleへ照合し、exact target、provider、fallback、cleanupを確認する。
- Qwen3.5最小dense modelをprimary full-model guardとし、fixed/Unicode/stop、prefill/decode、MTP off/onをaffected範囲だけ実行する。
- Phase 19 MoEはGDN共通pathを変更した場合だけ代表real-weight sliceまたは短いfull-model rowを追加する。35B全matrixを
  candidate screeningごとに繰り返さない。
- Qwen3.8 Q5_K_XLはllama.cpp診断artifactであり、sLLM correctness PASSには使わない。Phase 20完了後にQwen3.8 GGUFを
  正式対応する場合は別model lock/integration phaseで扱う。

### PX-I0: integration、upstream handoff、closeout

- adopted candidateだけをlatest sourceへrebaseし、fresh baseline/candidateをcounterbalanceして最終測定する。
- llama.cpp側とsLLM側の結果を混ぜず、target/backend/model/encoding/MTPごとのdecision tableへまとめる。
- code変更がある場合はaffected host/compile/GPU checksと1回のintegration review、findingだけのfocused re-reviewを行う。
- GPU/software compatibility、runtime、provenance、source lock/reference、main plan、historyを変更内容に応じて同期する。
- resultを`fixed`または`investigated-no-adopt`として明示し、本planをarchiveする。Phase 20の状態は独立に維持する。

## 計測matrix

| lane | target/backend | primary変数 | 指標 |
| --- | --- | --- | --- |
| end-to-end baseline | spare V620/R9700、HIP/Vulkan | 9.4k実code prompt、MTP 3、Q5_1 KV | prefill tok/s、TTFT、visible decode tok/s、acceptance、VRAM/GTT |
| scaling | R9700 HIP primary、V620 HIP guard | prompt 2,048/4,096/6,144/9,377 | dispatch/token、CPU lookup、GPU idle、prefill slope |
| GDN prefill | `gfx1201`/`gfx1030` HIP | chunk、fused routing、graphs、solution cache | operator time、launch/GEMM count、state error、workspace |
| GDN decode | `gfx1201`/`gfx1030` HIP | state mapping、wave、KDA、LDS/register | kernel time、occupancy、bandwidth、decode tok/s |
| MTP | HIP/Vulkan | off/width 1/width 3 | drafted/accepted、target eval、rollback、visible tok/s |
| memory | HIP/Vulkan | context 32k/262k、KV F16/Q8_0/Q5_1 | device VRAM、GTT、compute/draft/KV bytes、prefill |
| sLLM operator | `gfx1201`/`gfx1030` HIP | baseline/candidate、token/chunk境界 | output/state oracle、provider audit、kernel time |
| sLLM model guard | canonical local target | Qwen3.5 dense、必要時MoE/MTP | token/logit/KV contract、prefill/decode、fallback、cleanup |

O0はcorrectnessと単発profile、O1は一candidate数分のscreening、O2はsurvivorだけの1 warmup + 5 measuredを基本とする。
O2でnoise envelopeが重なる場合に限り最大3 warmup + 10 measuredへ増やす。slow candidateのEOSまでの生成、全PRの
full-model比較、Vulkan全matrixを完了条件にしない。

## 成果物

- source/model/software/GPU/promptを固定した再現manifestとcommand template。
- HIP/Vulkan、prompt scaling、MTP/KV ablationのbounded benchmark summary。
- GDN dispatch/solution lookup/GPU idle、decode kernel、VRAM/GTTを分けたprofile attribution。
- upstream issue/PR applicability tableと、必要なら外部投稿可能な最小reproducer草案。
- llama.cpp patch candidateまたは理由付き棄却記録。
- sLLM reuse/port/native decision、採用時のprovider implementationとprovenance event。
- exact `gfx1030`/`gfx1201`のfocused correctness/performance evidenceとcloseout history。

## 完了結果

### 固定identityと根因

- latest検証sourceはllama.cpp build 901、commit
  `4df29be4f4c3673f428170fda944a5b19f743bb8`、model SHA-256は
  `176a6a3f034e9cdc447c10cd00329fc9b31002e6589b9295f2ad4f1eefe0f6ab`、primary promptは9,435 token、
  SHA-256 `40225ad16c22a83f4065c7e666b54c3b5ea48e5aa9959c2f3d5da30a8b46977f`で固定した。
- 原因はGDNのhistorical small-GEMM展開ではなく、HIP baseline buildで`GGML_CUDA_FA_ALL_QUANTS=OFF`だったため、
  Q5_1 K/Vが`ggml_cuda_fattn_kv_type_supported`でFlash Attention対象外となったことである。Vulkanは同じQ5_1 caseを
  Flash Attentionで処理できたため、同じGPUでも大差になった。
- R9700の2,306-token MTP-off profileでは48,113 kernel dispatchのうちfused GDNは1,152 dispatch、kernel timeの
  6.16%だった。current fused `GGML_OP_GATED_DELTA_NET`は動作しており、過去issueの約15k small-GEMM GDN pathologyは
  再発していない。candidate profileではquantized-KV Flash-Attention kernelを確認し、GDN比率は5.90%のまま、
  trace下prefillは216.43から833.58 tok/sへ改善した。

### 性能とcorrectness

full 9,435-token Python code prompt、context 262,144、Q5_1 model/draft KV、MTP幅3のfresh baselineと、
`GGML_CUDA_FA_ALL_QUANTS=ON` candidateは次のとおりである。candidateは1 warmup + 5 measuredの中央値を示す。

| target/backend | baseline prefill/decode | candidate prefill/decode | 改善倍率 |
| --- | ---: | ---: | ---: |
| spare V620 `gfx1030` / HIP | 60.99 / 6.81 tok/s | 340.80 / 33.42 tok/s | 5.59x / 4.91x |
| R9700 `gfx1201` / HIP | 69.50 / 12.50 tok/s | 779.06 / 41.93 tok/s | 11.21x / 3.35x |
| spare V620 / Vulkan control | 207.18 / 36.56 tok/s | — | — |
| R9700 / Vulkan control | 726.48 / 51.91 tok/s | — | — |

- candidateのmeasured meanはV620が341.336/33.426、R9700が781.056/41.962 prefill/decode tok/sだった。
  peak VRAMは約30.1/30.2 GB、GTTは起動時近傍のままで、host-memory spillやCPU/backend fallbackはない。
- userが数分でdecode数tok/sなら中断してよいと明示したため、slow baselineは各一回で停止した。したがって受入条件14の
  baseline/candidate counterbalanceとbaseline 5 measuredだけはユーザー指示を上位authorityとして適用しない。
  survivor candidateは規定どおり両targetで1 warmup + 5 measuredを完了し、全run値を集計へ保持した。
- upstream既存Q5_1 Flash-Attention testはV620で336/336 PASSした。さらにQwenのhead dimension 256、GQA比6、
  KV長113/512/1024、query batch 1/3/17を覆うlocal testを追加し、CPU backend numerical oracleに対して
  `gfx1030`と`gfx1201`で各18/18 PASSした。GDN operator sourceは変更していないためGDN oracleの再実行条件は発生しない。
- 固定manifest、全反復値、profile attribution、raw summary digest、scope limitは
  [Phase X bounded summary](../../../../../../ci/matrix/phase-x-qwen38-amd-summary-v1.json)を正とする。

### 仮説、upstream、sLLM disposition

post-closeoutのmulti-GPU selection controlでは、独立V620 server 2基、V620×2 layer/tensor、
R9700+V620×2 layer/tensorを同時2要求で比較した。最大aggregate throughputは独立V620 2基、V620だけで
524,288 context/slotを必要とする場合はexperimental tensor、R9700を明示的に空けられるsingle-process候補は
layer split `5,2,2`となった。rowはupstream deprecatedのため除外し、3基tensorとV620×2 layerを棄却した。
これは通常起動・Phase X受入条件・Phase 20を変更しないpost-closeout evidenceであり、詳細は
[multi-GPU selection summary](../../../../../../ci/matrix/phase-x-qwen38-multi-gpu-selection-v1.json)を正とする。

2026-08-17の後続ユーザー決定は上記の「通常起動を変更しない」判断だけを上書きし、V620×2 tensor、parallel 2を
491,520 context/slotへ縮小してlocal Qwenの通常運用へ昇格した。Phase Xの受入結果とPhase 20の範囲は変更しない。
現行契約は[Local Qwen3.8 subagent](../../../../../development/local-qwen-subagent.md)を正とする。

| 項目 | 完了判断 |
| --- | --- |
| H1 GDN launch爆発 | 棄却。current fused GDNが使用され、historical dispatch pathologyなし |
| H2 GDN decode kernel | 主要因として棄却。GDN sourceを変えずにquantized-KV FA buildだけで改善 |
| H3 MTP | 棄却。off/幅1/幅3で機能し、短いcontrolでは幅3がvisible decode最速 |
| H4 KV/context | 採用。Q5_1のHIP Flash-Attention build coverageが根因。262k reserve自体は32k controlとの差を説明しない |
| H5 Harness/API | 棄却。direct server profileでも再現し、修正後Responses-compatible DeepSeek Harnessで正常動作 |

- #18823/#20292はclosed、#20366/#20334はmerged、#26001/#20377はopenのまま確認した。current sourceにはfused HIP/Vulkan
  GDN変更が到達済みである。Q5_1 HIP Flash Attentionがdefault-disabled all-quant buildから外れるexact defectに一致する
  issue/PRは見つからず、外部投稿は行っていない。local exact-shape testはupstream候補として未投稿で保持する。
- local Qwen subagentはspare V620 `gfx1030`専用HIP candidateへ切り替え、Responses-compatible endpoint、
  DeepSeek Harness、context 262,144、Q5_1 model/draft KV、MTP幅3で実taskを完走した。長い累積slotではdecodeが低下するが、
  初回taskのprefill/decodeは340.28/35.44 tok/sで、旧buildの崩れは解消した。
- sLLMは現状FP16 KVを使用し、このdefectのQ5_1 build branchを通らない。`linear_attention.gdn.v1`も原因ではないため、
  sLLM source/provider/model lockは変更しない。新しいllama.cpp source expressionのimport/adapt/portはなく、
  provenance eventやnoticeの追加も不要と判断した。

以上により主要原因を切替実験とprofileで説明し、exact `gfx1030`/`gfx1201`の採用candidateを得たため、Phase Xを
`fixed`として完了する。Phase 20の状態と完了条件は変更しない。

## 再計画・停止条件

- 一candidateが数分のO1で数tok/s域に留まり、profile指標もbaselineより改善しない場合はEOSを待たず中断する。
- profileでGDN以外が支配要因ならGDN tuningを止め、H3/H4/H5の該当laneへ移す。
- upstream PRがCUDA Tensor Core、Vulkan shader、別state sizeだけに依存する場合、HIPへ無理に適用せずtechnical referenceに留める。
- model format、Q5_K_XL support、Responses API、sLLM Vulkan backend、public ABI非互換が必要なら本Phaseへ追加せず別計画にする。
- 同じwork unitの2回reject、review時間が実装時間超、1時間以上の機能進捗停止、検証/docs 30%超、見積り1.5倍超、
  gate/受入条件変更時は追加candidate/review/verificationを止め、ユーザーへ報告して再計画する。
- timeout、crash、CPU fallback、mixed backend、GTTへのsilent fallback、zero test selectionをGPU PASSにしない。

[対応する履歴](../../../../../history/2026/08/11-20/phase-x-qwen35-gdn-amd-performance.md)

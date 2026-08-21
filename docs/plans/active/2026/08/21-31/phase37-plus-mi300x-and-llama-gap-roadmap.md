# Phase 37以降: MI300X最適化とllama.cpp機能差解消計画

## 目的

Phase 36で確認したexact `gfx942`の大きな性能差を、まず支配的なGDNとFull Attentionから解消する。
その後、2026-08-21に固定llama.cpp `b10453`と比較して棚卸しした、モデルアーキテクチャ・hardware・parallel以外の
未実装機能を、共通基盤から公開surfaceへ依存順に実装する。

この計画は2026-08-21のユーザー指示によりPhase番号と順序を割り当てる。Phase 36以前の完了条件を遡及変更せず、
角括弧で将来項目だったResponses APIとWebUIも後続Phaseへ割り当てる。各Phaseのcorrectness/security条件は必須とする。
性能値は採用判断と再計画に用いる目標であり、数値未達を隠すために比較条件やモデルを変更しない。

## 正本とbaseline

- 製品要件: repository外の`sLLM.md`。
- 全体計画: [main plan](../../../../main-plan.md)。
- API: [OpenAI compatibility profile](../../../../../api/openai-compatibility.md)。ResponsesやAnthropic Messagesは、
  実装前に別のversioned profileと外部仕様pinを追加する。
- runtime: [runtime architecture](../../../../../architecture/runtime.md)。transportからgeneration、token selection、
  model state、HIP providerを分離する。
- model identity: [model lock](../../../../../models/model-lock.md)。cache、adapter、checkpoint、dynamic model lifecycleで
  verified identityを弱めない。
- MI300X baseline: [Phase 36 archive](../../../../archive/2026/08/11-20/phase36-mi300x-current-main-validation.md)と
  [Session D summary](../../../../../../ci/matrix/phase36-mi300x-session-d-summary-v1.json)。Qwen3.5-4B BF16、FP16 KV、
  input ID `23066`を10,001個、greedy 2 output、3 warmup＋10 measuredで、sLLM E2E中央値は
  `22.556130816`秒、fixed llama.cppは`0.8512540725`秒、E1比は`26.4975x`だった。
- Session Dのrocprofv3 device shareはGDN `73.95%`、Full Attention `25.12%`、projection `0.70%`、other
  `0.23%`。Phase 37はこの二familyを最優先する。
- Phase 36のVMは削除済みである。新しいexact gfx942実機を確保するまではcompile、selector、host oracle準備だけを
  draftとして進め、compile成功や過去のSession D証拠を新candidateのGPU PASSへ読み替えない。
- 比較は同じupstream revision、token列、dtype、KV、GPU、warmup/反復、timing boundaryを固定する。
  GGUF bytes/tensor setが異なる間はE1 system-equivalentとし、strict-identicalと表記しない。

## 全体順序

| Phase | 状態 | 主範囲 | 主要依存 |
| --- | --- | --- | --- |
| 37 | ready-host-prep | gfx942 GDN・Full Attention provider parity | Phase 36 profile。実機baseline/perfはVM再確保待ち |
| 38 | planned | MI300X residual、FNUZ/GEMM、execution replay、最終peer比較 | Phase 37 fresh profile |
| 39 | complete | service operability・認証・observability基盤 | 現行profile v1 |
| 40 | planned | sampler chain、GPU sampling、logprobs、grammar/structured generation | 現行generation loop |
| 41 | planned | prefix/KV reuse、session checkpoint、context shift、speculation | opaque KVとPhase 40 token selection |
| 42 | planned | Completions・Embeddings・Rerank・token utility・infill endpoint | transport-independent modes |
| 43 | planned | Responses・Anthropic Messages・function/tool protocol | Phase 40・42 |
| 44 | planned | generic template、reasoning control、interactive CLI | Phase 41・43 message/state |
| 45 | planned | LoRA/control vector、dynamic model lifecycle/router cache | model lock・Phase 39 ops |
| 46 | planned | conversion、quantization、benchmark、quality/debug tools | stable GGUF/model identities |
| 47 | approval-required | 組込みtool/MCP実行 | Phase 39 security・Phase 43 tool protocol |
| 48 | planned | minimal WebUI/server UI | Phase 39・42〜45 public APIs |

Phase 37–38はMI300X性能lane、Phase 39–48は機能laneである。MI300X VMを利用できない期間も、Phase 39以降の
host-only設計・実装は独立branch/work unitとして進められ、Phase 37/38のGPU完了を開始・merge gateにしない。ただし同じ
runtime領域を変更するcandidateは、統合時点のsourceでaffected host regressionを実行し、次のMI300X実機sessionでは
fresh baselineから測定する。

複数surfaceへ現れる機能の所有権は一つに固定する。Phase 39はresumable transport/replay、Phase 40はsamplerと`n` choice
state、Phase 41はassistant-prefill/state semantics、Phase 42はFIM/infill execution modeを所有する。Phase 42〜44の後続記述は、
所有済み機能を各wire profile、renderer、CLI/UIへ接続するadapter範囲であり、別実装や別state machineを作らない。

## 共通実施規則

1. 各Phase開始時に、対象surface、非目標、仕様pin、受入case、source/build/model identityを固定する。
2. llama.cppから直接reuseする場合はMIT provenanceをfile単位で記録する。llama.cpp以外はno-copy referenceを維持する。
3. host unitは非整列値と境界の両側を含める。GPU PASSはexact target、数値oracle、HIP dispatch、fallbackなし、
   cleanupを必要とし、timeout、crash、0 caseをPASSにしない。
4. draftはaffected test、integrationは影響matrixと一回のintegration review、release/pushはclean candidateの最終gateを使う。
   各checkpointのfresh reviewや全GPU rerunを要求しない。
5. public APIは各Phaseの最初に外部schema/profile pin、rejection matrix、security/provenance境界を固定する。
   未知fieldを黙って無視せず、versioned schemaに従い4xxでfail closedにする。prompt、token、secretをmetric/logへ出さない。
6. 性能candidateは同一sourceでbaseline/candidateをcounterbalanceし、median、MAD、全反復値、kernel family、VRAM、
   process/healthを記録する。局所改善だけでwall改善を主張しない。
7. 既存gfx1030/gfx1201 routeを変更する共通sourceは該当targetのfocused regressionを行う。gfx942固有sourceだけなら
   RDNA GPU rerunを常に要求せず、compile/dispatch selector testで非選択を証明する。

## Phase 37: gfx942 GDN・Full Attention provider parity

### Scope

- `gfx942:sramecc+:xnack-`、wave64、ROCm 7.14、Qwen3.5-4B BF16/FNUZ FP8、4 KV encodingを対象とする。
- Phase 35のcolumn-state GDNとQ_TILE=4/GQA共有attentionがgfx1030/gfx1201にだけ選択され、gfx942でbaselineへ
  戻る現状を解消する。
- 現行GDN selectorはtoken 128以上でもgfx942をcolumn candidateへ送らず、baselineはvalue head 32 workgroupで
  128 state要素をtoken loop内で走査する。Full Attention selectorもgfx942をcommon tiled providerへ送らず、
  1 query×KV headごとにkey方向のreduction/barrierを反復する。Phase 37はこのroute差を明示baselineにする。
- state layout、accepted-state transaction、opaque KV、`contiguous-resident`、public API、model/GGUF形式は維持する。

### Work units

1. P37-A0 baselineをfresh sourceで再取得する。これはexact gfx942 VMを再確保するまでdeferredとする。Phase 36の
   10,001/2、operator境界、device family shareを再現し、
   provider ID、kernel symbol、wavefront、dispatch/resource、binary bundleをsummaryへ結合する。
2. P37-A1でwave64 column-state GDNを実装する。head×state-column ownership、norm/recurrent update/output projection、
   state publicationをwave64向けに再配置し、token `1/3/17/127/128/129/255/256/257/2047/2048/2049/10001`、
   width/column tail、nonzero start、zero/nonzero state、MTP rewind/replay、chunk境界を扱う。Phase 36でgfx942へ固定した
   128項index順FP32 norm和を最初のN0 candidateでは維持する。
3. P37-A2でgfx942向けtiled Full Attentionを実装する。複数query rowとGQA headでK/V tileを共有し、vector QK/PV、
   online softmax、causal/chunk境界、FP16/dynamic FP8/static FP8/NVFP4 KVのunpack/scale共有を独立candidateにする。
   `63/64/65`、`127/128/129`、`255/256/257`、`511/512/513`、`2047/2048/2049`、`4095/4096/4097`、
   `8191/8192/8193`、`10001`とnonzero startをselector境界へ含め、Phase 35のthresholdを無検証でコピーしない。
4. A1/A2は別route・別採否にし、一方の不成立で他方を捨てない。FP16/FP8 accumulatorやsoftmax順序変更はN0〜N3へ
   分類し、N2以上はユーザー承認なしに性能だけで採用しない。
5. P37-A3はP37-A0と同じくexact gfx942 VM再確保までdeferredとし、sLLM 4B BF16/FNUZ FP8のshort、decode-long、
   prefill-long、10,001/2を3＋10反復する。fixed llama.cppとの
   peer比較は、既存peer artifactとstrictに合わせられるBF16 weight＋FP16 KV行だけを同じVM・protocolで再取得する。
   FNUZ FP8はsLLM内のBF16対照とし、対応peer artifactなしにllama.cpp比を作らない。

### Acceptance

- operator oracleはGDN state digest、causal/GQA/future-key poison、NaN/Inf/subnormal、非整列/tailを含め全PASSし、
  production rowはexact gfx942/HIP-only、fallback/partial offloadなし、cleanup 0、終了後process/HBM/GTT/ECCが
  baselineへ戻る。
- selectorはgfx942の承認shapeだけを新providerへ送り、gfx1030/gfx1201とshort shapeの既存routeを変えない。
- 採用candidateは同じfamilyのdevice中央値がbaselineよりMADを越えて短縮し、E2Eを有意に悪化させない。
- Phase終端でGDN、Full Attention、projection、other、host-visible E2Eを再分解する。peer未達でも残差の所在を固定して
  Phase 38へ渡し、根拠なしに別kernelを追加しない。

### Non-goals

- multi-GPU、別CDNA SKU、一般的なFlashAttention 4製品claim、KV format変更、model architecture追加。

## Phase 38: MI300X residual closureとpeer比較

### Scope and selection

Phase 37後のfresh profileだけを根拠に、E2E差へ寄与するcandidateをAmdahl上限順に選ぶ。候補は以下だが、shareが
小さいものを実装gateにしない。

1. wave64 MMVF/BF16 projectionとhipBLAS/hipBLASLt solution比較。ただしPhase 36でprojectionはdevice totalの
   `0.70%`だけなので、Phase 37後もAmdahl上限が小さければ実装しない。
2. FNUZ FP8の有限workspace、複数solution、shape/target別algorithm cache、queue別handle。
3. 共通activationを消費するQ/K/V、gate/upのdynamic quantization共有、producer直結、quantize+matmul fusion。
4. parameter update可能なcommand listまたはreusable HIP Graph、event/completion pool、registry lock、token/position H2Dと
   token D2H boundaryの集約。requestごとのgraph instantiateは再導入しない。
5. VMM=trueの実機で`contiguous-resident`とvirtual-contiguous/incremental commitを、同じopaque KV契約で比較する。
6. loader/profileがcold-start差を支配する場合だけmmap、並列hash、double-buffered H2Dを別cold metricとして扱う。

### Acceptance

- 各candidateはcorrectness、resource、target selectorを独立評価し、採用/棄却理由とAmdahl上限を記録する。
- canonical 10,001/2 direct laneでsLLM BF16/FNUZ FP8を3＋10反復する。fixed llama.cpp E1比較はBF16 weight＋FP16 KVに
  限定し、FNUZ FP8はsLLM BF16とのdtype内比較として別表示する。
- 最終目標はsLLMの単一request E2E中央値を同条件fixed llama.cpp以下にすることとする。`<=1.0x`は性能目標であり、
  correctnessを緩めて達成しない。届かない場合は残差上位family、必要な新scope、推定上限を示して同一work unitを
  反復せず再計画する。
- MI300X結果をMI300A/MI325X/別partitionへ一般化しない。

## Phase 39: service operability・認証・observability

> 状態: complete（2026-08-21、host-only。GPU PASS claimなし）

### Scope

- `/healthz`はprocess liveness、`/readyz`はmodel resident・backend受付可否を分離する。
- opt-in Prometheus metricsはqueue、request、token、TTFT、E2E、failure、cancel、VRAM/arena、model identity labelを
  bounded cardinalityで公開し、prompt/token/credentialを含めない。
- read-only props/slots、admin用slot cancel、resumable SSEのevent IDとbounded replay bufferを追加する。
- CORS allowlist、TLS certificate/key、key fileと複数API key、constant-time照合、権限分離したadmin credentialを実装する。

### Acceptance

- startup/loading/ready/draining/failed/shutdownの状態遷移、slow client、disconnect、replay範囲外、queue full、key rotation、
  malformed configをhost integrationで検証する。
- health endpointはGPU処理を起動せず、readinessはfallbackで成功扱いにしない。metric scrapeはgenerationをblockせず、
  label数とmemory上限をtestで固定する。
- TLS/CORS/auth無効時の既存local profileを維持し、有効化時だけ対応surfaceを公開する。

### Closeout

atomic lifecycle、bounded/redacted slot registry、admin cancel、digest-only user/admin key storeとatomic reload、exact CORS、
Rustls TLS、opt-in metrics、nonblocking runtime allocator memory snapshot、明示opt-in resumable SSEを実装した。
既存HTTP contract 10件を含むserver all-target 62件をhostでPASSし、clippy warning 0を確認した。詳細は
[archived Phase plan](../../../../archive/2026/08/21-31/phase39-service-operability.md)と
[history](../../../../../history/2026/08/21-31/phase39-service-operability.md)を正とする。

## Phase 40: token selection・grammar・structured generation

### Work units

1. samplerをbackend非依存のordered chainへ型付けし、greedy、temperature、top-p、presence/frequency penalty、seedを
   既存互換のadapterとして移す。
2. top-k、min-p、typical、Mirostat、DRY、XTC、adaptive/dynamic temperature、ignore-EOSを追加し、順序とdefaultを
   versioned request schemaへ固定する。
3. logit bias、選択token logprob、top-logprobsを実装する。NaN/Inf、tie、all-masked、large vocabularyをfail closedにする。
4. GPU samplerはpenalty、mask、partial selection、RNG、selected-token D2Hを一つのprepared pathへまとめる。CPU referenceを
   oracleとして残し、GPUを使えないhost testでfull-model性能を主張しない。
5. GBNFをbounded automatonへcompileし、UTF-8/byte fallbackとtoken trieでvalid-token maskを作る。JSON Schema subsetは
   明示support表へlowerし、unsupported keywordを拒否する。
6. `response_format`、structured output、`n>1` choicesをtransport-independent generationへ接続する。choiceごとのseed、
   KV/sampler/stop stateを分離する。

### Acceptance

- samplerごとにfixed logits、tie、境界値、deterministic seedをreferenceへ一致させ、既存requestのtoken列を維持する。
- grammarは受理文字列だけを生成し、無効schema、状態爆発上限、全token禁止を明示errorにする。
- logprobsは実際にsamplingへ使ったpost-bias/post-mask distributionと一致する。
- GPU pathはexact target、fallbackなし、selected token以外の不要なfull-vocabulary D2Hを行わない。

## Phase 41: prefix/KV・session state・speculation

### Work units

1. prefix cache keyをmodel-lock fingerprint、adapter identity、renderer/template digest、exact token列、KV encoding、target
   semanticsへ結合し、最長一致とbounded evictionを実装する。
2. vAttention pageをrequest間でread-only共有し、continuation時にcopy-on-writeする。GDN、RoPE position、sampler/stop stateを
   prefix ownerと分離する。
3. context shiftは保持token範囲、absolute/logical position、RoPE scaling、attention maskをversioned policyへ固定する。
4. session/slot checkpointはheader、model/adapter/template identity、token history、KV/GDN/state checksumを持ち、atomic write、
   size/quota、corruption・version mismatch拒否を実装する。KV＋会話＋model SHA-256の簡易永続化をここで満たす。
5. assistant prefillをchat/Responses/Completions共通generation inputへ追加する。
6. external draft modelとngram speculationは同じpropose/verify/accept contractへ接続し、MTPとは別providerとして扱う。
   reject時のCOW rollback、accepted-prefixだけのpublish、通常逐次生成とのtoken一致を維持する。

### Acceptance

- cache hit/miss/partial hit、eviction、concurrent readers、cancel、restart、corrupt/truncated checkpoint、wrong model/adapter/KVを
  検証する。異identityのsilent reuseは不可。
- reused resultはfresh prefillとtoken/visible outputが一致し、cache accounting、cleanup、quotaが閉じる。
- speculation disabled時の既存経路を変えず、有効時はaccepted/rejected/proposed accountingと逐次同一性を示す。

## Phase 42: inference modeと基本public endpoint

### Scope

- OpenAI Completions、Embeddingsを対応するversioned schema pinへ実装する。
- RerankはOpenAI互換を名乗らず、別のsLLM endpoint/profileとしてquery/document、score意味論、上限を固定する。
- tokenize、detokenize、apply-template、input-token-countを追加し、CLIとHTTPが同じfrontend serviceを使う。
- FIM/infillをmodel capabilityとverified templateへ接続し、unsupported modelは拒否する。

### Acceptance

- tokenizer special token、Unicode、byte fallback、empty/large input、normalization、template digestをCLI/HTTPで一致させる。
- Embeddingsはpooling、normalization、dtype、dimension、usageを明示し、internal embedding gatherをHTTP supportへ正しく接続する。
- Rerank scoreはfixed oracleと順序/tieを満たす。completion/infillはstop、usage、streamingを共有し、`n` choicesは
  Phase 40が所有するchoice stateをwire responseへ写像するだけとする。
- endpoint追加でChat Completions profile v1のreject/response/SSE semanticsを暗黙変更しない。

## Phase 43: Responses・Anthropic Messages・function/tool protocol

### Work units

1. official Responses schemaを実装開始時のfull commit/versionへpinし、request item、output item、usage、error、stream eventの
   closed state machineを定義する。Chat Completionsのaliasにはしない。
2. Anthropic Messagesはversion header、content block、stop reason、usage、SSE eventを別compatibility profileへ固定する。
3. function/tool definition、tool choice、tool result message、parallel tool call、structured argumentsを共通internal itemへlowerする。
4. Phase 40のJSON Schema grammarをtool argumentsへ適用し、生成後parseだけで正しさを主張しない。
5. reasoning content、assistant prefill、multi-choice、cancel、mid-stream error、resumable eventを各transport adapterへ接続する。
   assistant prefillはPhase 41、multi-choiceはPhase 40、resumable replayはPhase 39のstate machineを再利用する。

### Acceptance

- official-client fixtureとraw HTTPでnon-stream/SSE、tool call/result round trip、structured output、reasoning、cancel、invalid item、
  unsupported multimodal typeを検証する。
- transport間で同じinternal generation requestとvisible token順序を使い、API固有eventへ変換する。
- このPhaseはtool callを生成・受理するが、任意のtool/MCPをserver process内で実行しない。実行はPhase 47まで明示的に無効。

## Phase 44: template・reasoning control・interactive UX

### Scope

- arbitrary Jinja互換templateとbounded kwargsをsandboxed rendererへ追加する。filesystem、environment、network、process、
  unrestricted object accessは公開しない。
- model lockでreviewed templateをdefaultに保ち、custom templateはdigest付きopt-inとする。
- reasoning budget/mode、生成中のreasoning controlを実装し、Phase 41のassistant-prefill semanticsとPhase 42のFIM/infill
  execution modeをtemplate/CLIへ接続する。Phase 44で別のprefill/FIM stateを作らない。
- interactive CLI、conversation history、reverse prompt、prompt file、save/resume sessionをPhase 41 checkpoint上に実装する。

### Acceptance

- llama.cpp互換fixtureとsLLM canonical templateでrole、special token、tool/reasoning block、Unicode、kwargs、malformed templateを検証する。
- template resource上限、recursion、oversized output、unknown filter/functionをfail closedにする。
- interactive/non-interactive、resume/freshで同じtoken列を生成し、terminal inputとprompt fileを混同しない。

## Phase 45: adapter・control vector・dynamic model lifecycle

### Work units

1. preloaded LoRAをverified base model/target tensor/shapeへ結合し、requestごとのadapter setとscaleを指定可能にする。
2. control vectorをlayer/range/dtype/scale付きderived artifactとしてlockし、request stateへ適用する。
3. model registryを複数alias、lazy load、preload、unload、LRU/cache quota、offline-onlyへ拡張する。
4. routerはrequest aliasをimmutable model+adapter identityへ解決し、load中/draining/cancel/failureをPhase 39 readiness/metricへ反映する。
5. load/unload中のGPU allocation、in-flight request、shared tokenizer/template、failed model quarantineを所有権contractへ固定する。

### Acceptance

- wrong base、missing tensor、shape/dtype mismatch、duplicate adapter、scale boundary、adapter orderを拒否する。
- adapter/control disabled時はbase logits/tokenを維持し、有効時はbounded slice oracleとfull-model smokeをPASSする。
- unloadはin-flight ownerを早期解放せず、新規requestを止め、最後のowner後にVRAM/file handleをbaselineへ戻す。

## Phase 46: conversion・quantization・benchmark・品質評価tool

### Scope

- general HF-to-GGUFをmodel plugin/capability方式へ拡張し、supported architecture/dtypeだけを受理する。
- GGUF split/merge、LoRA conversion、execution-ready layout/repackをmodel-lock/derived-lockへ結合する。
- quantize/imatrixはsLLMが採用するBF16/FP8/NVFP4/MXFP4等だけを対象とし、一般Q8_0/Q4_K対応を導入しない。
- `sllm-bench`、perplexity、KLD、task eval、token/logit/debug dumpを共通dataset/result schemaと再現可能なseedへ固定する。

### Acceptance

- converterはtensor catalog、metadata、recipe、source/output digest、tool commit/args/environmentをfail closedに検証する。
- split→mergeはbyte/semantic identity、LoRA conversionはbase+adapter oracle、quantizeはtop1/KLDとbounded slice誤差を記録する。
- benchmarkはwarmup/measurement、E2E/TTFT/TPOT/prefill、model lifecycle、GPU identityを明示し、raw trace/modelを追跡しない。
- debug dumpはopt-in、size上限、secret/prompt方針を持ち、production defaultで無効にする。

## Phase 47: 組込みtool・MCP実行

このPhaseは新しい外部実行security boundaryを作るため、開始時にユーザーがdeployment trust model、許可tool、credential、
network/filesystem、confirmation、audit保持を明示承認するまで`approval-required`とする。Phase番号への割当は実装許可を
先取りしない。

### Proposed scope

- Phase 43のtool callを、server本体から分離したworkerへ渡す。tool allowlist、schema validation、timeout、CPU/memory/output、
  concurrency、cancellationを必須にする。
- MCP client/server connectionはendpointとcapabilityをdeployment設定へpinし、credentialをmodel/promptから分離する。
- network deny-by-default、workspace外filesystem deny、shell/process denyをdefaultとし、capability単位で明示許可する。
- tool resultはuntrusted contentとしてmessageへ戻し、system/developer instructionへ昇格させない。全call/result digest、duration、
  dispositionをsecret-free auditへ残す。

### Acceptance

- schema逸脱、prompt injection、oversized output、timeout、worker crash、disconnect、credential漏洩、path escape、network deny、
  cancel/retryをadversarial testで検証する。
- tool未設定時はPhase 43のtool-call生成だけが利用でき、任意codeを暗黙実行しない。

## Phase 48: minimal WebUI/server UI

### Scope

- sLLM HTTP APIだけを利用する薄いUIとして、model選択、chat/stream、reasoning/tool表示、sampling/structured controls、
  session save/resume、adapter選択、health/metrics要約を提供する。
- admin面はmodel load/unload、slot cancel、key/credentialの値を表示しないstatus、log downloadを権限分離する。
- UI資産はserver binaryへ埋め込むかversioned static artifactとして配布し、CDN依存と外部telemetryをdefaultで持たない。

### Acceptance

- keyboard操作、stream cancel/reconnect、large conversation、tool/reasoning block、upload制限、XSS/CSRF/CORS、auth分離を検証する。
- UIだけのhidden APIやmodel filesystem accessを作らず、APIで拒否される操作をUIが迂回しない。
- WebUIのrichnessは完了条件にせず、CLI/APIで利用できる機能の管理surfaceに限定する。

## Intentional exclusions and deferred items

- Vulkan、一般的なllama.cpp INT4/INT8+scale形式は明示的な製品方針どおり対象外。
- model family/architecture追加、RDNA3等の新hardware、CPU/NVIDIA、parallel/continuous batching、multi-GPU、Infinity Fabric、
  RCCL/RDMAは今回のllama.cpp機能差計画へ含めない。
- LMCache、RadixAttention、Paged Attention、TurboQuant、残るKV形式、MXの将来形式は今回のPhaseへ自動追加しない。
  Phase 41のcache/state ABIは後からproviderを追加できる形にする。
- README整備、人間による発表、release packagingは別作業であり、Phase 37–48の機能受入をblockしない。
- fixed llama.cppに存在する機能でも、外部仕様がないrerank、Anthropic、MCP、server extensionは「llama互換」を名乗らず、
  sLLM固有または別仕様pinとして公開する。

## Phase closeout

各Phaseは、採用source、棄却candidate、test/evidence、既知制約、次Phaseへの入力をmatching historyへ記録する。完了または
放棄時にこのplanをarchiveへ移し、main planのroadmap/current stateを更新する。Phase 37–48を一括commitにせず、独立して
review・rollback可能なPhase/work unit単位で公開する。

[全体計画](../../../../main-plan.md) / [対応する履歴](../../../../../history/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md)

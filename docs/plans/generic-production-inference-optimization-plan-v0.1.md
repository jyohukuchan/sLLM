# Generic production inference optimization plan v0.1

Status: active; P0, P1-A, P1-B1, and P1-B2 CPU RMS normalization/RoPE/typed trace completed; causal GQA, state transaction, and adapter fixtures next

## 前回の要点

- Qwen3.5-9B AQ4 resident workerは、モデルを常駐化し、短いcontextでは約70 tok/sのdecodeを実現した。
- しかしprefillは32層を1 tokenずつ通すdecode-like経路で、1011-token promptは63.638 tok/s、1339-token promptは約59.5 tok/sだった。
- 8個のself-attention層は各tokenで現在のKV cache全体を走査するため、1339-token contextからのdecodeは約42.64 tok/sまで低下した。
- 既存のbatched AQ4 projection、causal/cached-prefix attention、mixed-request runnerはcomponentまたはdiagnostic経路にあり、OpenWebUIが使うresident sessionへ接続されていない。
- 過去の数千tok/s値はllama.cppのbatched prefill、またはuLLMの単一component測定であり、uLLM full-model product pathの値ではなかった。

## 今回の変更点

- 最適化をQwen3.5/AQ4専用loopとして追加せず、モデル構成、状態、batch計画、backend演算を分離する共通実行基盤として計画する。
- model adapter、typed model graph、state schema、batch planner、backend operation registry、generic model executorの境界を定義する。
- component性能ではなく、resident worker、OpenAI gateway、OpenWebUIまで同じexecutorが使われたことを機械的に証明してからpromotionする。
- AQ4/Qwen3.5を最初の縦切りとするが、SQ8/Qwen3と追加モデルadapterでも同じexecutorを使えることを完成条件にする。

## 次の行動

P1-B2でcausal GQA core、state transaction契約とQwen3.5/Qwen3 adapterのgraph生成fixtureを段階的に追加する。AQ4 kernelの変更から先に始めない。

## 1. 目的

実際の推論経路がbatched/chunked prefill、context-scalable attention、量子化別batched projectionの恩恵を受ける状態を作る。

同時に、新しいモデルを追加するときに次を再実装しない構造にする。

- prefill chunk planning
- token/request batch layout
- buffer lifetimeとworkspace planning
- paged KV allocationとcommit
- recurrent/conv state ownership
- backend/kernel selection
- cancel/reset/rollback
- performance/correctness evidence
- resident worker、gateway、OpenWebUI integration

モデル固有実装は、tensor binding、層構成、演算属性、状態schemaの宣言へ限定する。

## 2. 設計原則

### 2.1 分離する五つの軸

次の軸を一つのmodel名分岐へまとめない。

1. model topology
   - 層順、演算種類、tensor binding、head数、hidden幅、RoPE、activation、residual構造
2. execution phase
   - cold prefill、cached-prefix prefill、decode、将来のverify/speculative phase
3. state kind
   - paged KV、sliding-window KV、recurrent state、conv history、将来のcross-attention cache
4. numerical format
   - F32/BF16/FP16、AQ4_0、SQ8_0、将来の量子化形式
5. backend capability
   - CPU、HIP、GPU architecture、具体的GPU、対応shape、workspace上限

### 2.2 model adapterに許可するもの

- package/HF tensor名からlogical weightへの対応付け
- layer topologyと繰返しblockの構築
- operator属性とshapeの指定
- state schemaの指定
- architecture-specific semanticsを表す明示的なoperator属性
- model identity、tokenizer、EOS、context等の契約

### 2.3 model adapterに置かないもの

- GPU API呼出し
- token-by-token/prefill/decode loop
- buffer allocation方針
- chunk size決定
- kernel IDの直指定
- schedulerの進捗更新
- HTTP/SSE/OpenWebUI処理

### 2.4 specialized kernelの扱い

モデル固有kernelは禁止しない。ただし次を満たす。

- 標準operator semanticsに対する最適化としてregistryへ登録する。
- model architectureはoptionalな選択条件とし、generic fallbackを必須にする。
- generic executorへ`if model == ...`を追加しない。
- kernelは対応shape、format、state layout、phase、GPU capabilityを宣言する。
- correctness oracleとfallback比較を通過しなければproduction選択されない。

## 3. 目標アーキテクチャ

```text
served-model manifest / package
  -> ModelAdapter
  -> ModelGraph + WeightBindings + StateSchema
  -> Request/Scheduler state
  -> BatchPlanner
  -> ExecutionBatch + LoweredExecutionPlan
  -> GenericModelExecutor
  -> BackendOpRegistry
  -> CPU / HIP / future backend implementation
  -> ProductionExecutionTrace
  -> InferenceSession / resident worker / gateway / OpenWebUI
```

### 3.1 ModelGraph

最初のgraph nodeは次を対象にする。

- EmbeddingGather
- Norm
- Linear
- FusedLinearGroup
- RotaryPosition
- DenseSelfAttention
- RecurrentAttention
- Activation
- GatedMlp
- ResidualAdd
- FinalNorm
- LmHead
- TopK/Sampling

後続追加を想定するnode:

- MoE router / experts
- sliding-window attention
- cross attention
- convolution/SSM scan
- multimodal projector
- MTP/speculative head

nodeは特定model名ではなく、入力/output layout、weight、属性、state accessを持つ。

### 3.2 WeightBindings

logical weight IDをpackage tensorへ対応付ける。量子化formatはbinding metadataとして保持し、graph topologyと分離する。

旧packageはmodel adapterがgraphを導出する。将来のpackage schemaではgraph/state metadataを保存できるようにするが、旧packageを直ちに破棄しない。

### 3.3 StateSchema

各state entryは次を持つ。

- state kind
- logical shapeとdtype
- request/layer ownership
- block/sliding/recurrent layout
- initialize、append/scan、commit、reset、snapshot/restore契約
- backend workspace要求

対象state:

- paged KV
- recurrent matrix/state
- convolution history
- position/cache length

Rust schedulerはhandle、block table、lifetimeを管理し、payloadはbackendが所有する既存ADR方針を維持する。

### 3.4 ExecutionBatch

共通batch表現は少なくとも次を持つ。

- phase
- packed token IDsまたはhidden buffer
- request IDs
- sequence offsets/lengths
- prefix lengths
- absolute positions
- state handles/block tables
- chunk ranges
- graph compatibility key
- commit generation/nonce

最初は同一chunk幅の矩形batchから実装し、その後ragged batchへ広げる。paddingによる過大計算を避けるためlength bucketを使う。

### 3.5 BatchPlanner

plannerはmodel/formatに依存せず、graph/state/backend capabilityから次を決める。

- cold/cached-prefix/decode phase
- request bucket
- token chunk M
- workspaceとVRAM headroom
- operatorごとのimplementation候補
- atomic commit単位

prefill実行成功後だけscheduler、KV、recurrent/conv state、progressをまとめてcommitする。cancel/error時に一部requestだけ進まないよう、prepare/execute/commitを分離する。

### 3.6 BackendOpRegistry

registry key:

```text
OpKind
+ Phase
+ input/output layout
+ weight/activation/KV dtype and format
+ state layout
+ shape bucket
+ GPU/backend capability
+ optional model architecture constraint
```

既存`backend_dispatch`のoperation/phase/format/model/GPU matchingを拡張し、static descriptorだけでなく実行可能implementation、workspace estimator、capability probe、fallback chainを登録する。

### 3.7 GenericModelExecutor

executorはmodel graphを走査し、lowered planを実行する。Qwen/AQ4/SQ8という名前を知らない。

責務:

- buffer lifetime/reuse
- layer/node execution
- state prepare/commit/reset
- stream/event ordering
- execution trace
- operator fallback
- error/cancel propagation

`InferenceSession`のpublish-before-commit契約は維持する。

### 3.8 ProductionExecutionTrace

production pathが実際に最適化executorを使ったことを証明する。

必須項目:

- model graph/schema versionとdigest
- resolved executor ID
- phaseごとのchunk幅
- operator implementation IDs
- real token/request batch幅
- fallback count/reason
- workspace/VRAM estimateと実測peak
- state layout/KV dtype
- binary/manifest/product identity

component rowは`scope=component`、full modelは`scope=full_model`、server/OpenWebUIは`scope=production_server`として機械的に分離する。

## 4. 実装段階

### P0: 契約固定

Status: completed by ADR 0004, ADR 0005, production-execution-trace-v0.1, and prefill-validation-v0.1.

成果物:

- model graph/state schema ADR
- backend operation registry ADR
- generic execution trace spec
- prefill validation evidence spec
- 本計画のcommit

Gate:

- model、phase、state、format、backendの依存方向が明文化される。
- AQ4/SQ8固有型を共通APIへ漏らさない規則が決まる。
- component性能をproduction性能としてpromotionできないschemaになる。

### P1: typed graphとCPU reference executor

Status: P1-A, P1-B1, and P1-B2 CPU RMS normalization/RoPE/typed trace completed; causal GQA, state transaction semantics, and adapter fixtures remain.

実装:

- `model_graph`、`state_schema`、`execution_batch` module
- graph validation、shape inference、weight binding validation
- CPU reference executor
- Qwen3.5/Qwen3 adapterのgraph生成fixture

Gate:

- Qwen3.5 hybrid 32層とQwen3 dense stackを同じnode型で表現できる。
- graphにmodel名による実行分岐がない。
- invalid graph/state/weight bindingがGPU allocation前に拒否される。

### P2: executable backend registry

実装:

- runtime implementation descriptor
- capability probe
- shape/workspace estimator
- priority/fallback selection
- resolved execution trace

Gate:

- CPU、HIP generic、RDNA4、R9700 overrideを同じregistryで解決できる。
- unsupported shapeは明示fallbackまたはfail-closedになる。
- selection testがmodel/format/GPU specificityとpriorityを固定する。

### P3: stateless batched operator vertical slice

実装順:

1. embedding gather batch
2. norm batch
3. AQ4/SQ8/F32/BF16 linear batch
4. fused gate/up、activation、down
5. residual batch
6. final normとLM head batch

shape:

- M=1, 8, 16, 32, 64, 128
- hidden/intermediate/vocabの実model shape

Gate:

- 同一operator semanticsをAQ4/SQ8/passthrough formatが共有する。
- sampled output、finite、shape、fallbackがCPU/referenceと一致する。
- component結果は次段のfull-model接続前にはpromotion不可。

### P4: stateful prefill/decode operator

Dense attention:

- chunked cold causal attention
- cached-prefix attention (`L x M`と`M x M`)
- ragged/paged KV write
- context-scalable paged decode

Recurrent/linear attention:

- projection batch
- conv/state scan
- ordered state commit
- chunk boundary equivalence

Gate:

- all-M1とM=8/16/32/64/128でcache/state/logits/tokenが比較できる。
- full`N x N`attention matrixを確保しない。
- chunk境界を変えても最終stateとgreedy tokenが一致する。

### P5: generic batch plannerとmodel executor

実装:

- fixed-width bucket planner
- VRAM/workspace-aware chunk planner
- packed/ragged metadata
- atomic multi-token state commit
- generic executor
- `InferenceSession` adapter

Gate:

- component CLIではなく、full resident sessionがM>1を実行する。
- cancel、publisher failure、OOM/fallback後にstateがbaselineへ戻る。
- execution traceがM>1とresolved implementationを証明する。

### P6: Qwen3.5-9B AQ4 production migration

手順:

1. 旧tokenwise resident pathをM1 oracle/fallbackとして保持する。
2. Qwen3.5 adapterをgeneric graph/executorへ接続する。
3. M=16から32/64/128へ段階的に上げる。
4. linear-attention scanと8 dense-attention層をfull modelで検証する。
5. manifest worker、gateway、OpenWebUIを同じexecutorへ接続する。
6. canary manifestからatomic activationする。

Promotion minimum:

- prompt 1011のproduction prefillが現63.638 tok/sの5倍以上。
- prompt 2048でも現tokenwise baselineの5倍以上、OOMなし。
- 目標値はprompt 1024で1000 tok/s以上とし、minimum到達後もprofileを継続する。
- prompt 1339からのdecodeが現42.64 tok/sを25%以上改善するか、少なくともdecodeを5%以上回帰させずprefillだけを先行promotionする。
- short-context decode p50はbaseline比5%以上回帰しない。
- greedy token列、EOS/length、cancel/reset、連続requestが一致する。
- OpenWebUIでTTFT、token/s、finish/termination reasonを確認する。

### P7: Qwen3 SQ8 production migration

- Qwen3 adapterを同じgraph/executorへ接続する。
- 既存SQ8 M128、CK、paged KV、source oracleをregistry implementationとして再利用する。
- SQ8 worker/OpenWebUI release gateを回帰させない。

Gate:

- AQ4とSQ8のworker driver/session/executorに量子化名による制御flow分岐がない。
- format差分はweight bindingとregistry selectionに限定される。

### P8: 新モデル追加可能性の証明

二段階で証明する。

1. synthetic graph
   - dense-only、hybrid recurrent、MoE placeholderをexecutor変更なしで構築する。
2. 追加の実model adapter
   - 採用モデルを別途決め、adapter/weight binding/state schemaだけでfull-model smokeを通す。

Gate:

- generic executor、planner、worker protocol、gatewayを変更せず追加できる。
- 新しいoperator semanticsが必要な場合も、operator/registry extensionとして追加され、既存model分岐にならない。

### P9: request batchingとprefix reuse

generic executor安定後に進める。

- multiple active requests
- continuous batching
- ragged prefill/decode
- prefix cache ownershipとeviction
- state compatibility key
- fairness/backpressure

prefill token batchingとrequest continuous batchingを別機能として実装・検証する。

### P10: cleanupと正式化

- model専用token loopをoracle/diagnosticへ降格する。
- compatibility child path、一時alias、重複Qwen loopを削除する。
- package graph metadataのversioningを正式化する。
- README、deployment、operator catalogを更新する。

## 5. 検証matrix

### 5.1 正しさ

| 軸 | 初期値 |
|---|---|
| model | Qwen3、Qwen3.5 hybrid |
| format | AQ4_0、SQ8_0、可能な範囲でBF16/F32 reference |
| phase | cold prefill、cached-prefix prefill、decode |
| M | 1, 8, 16, 32, 64, 128 |
| prompt/context | 1, 8, 32, 128, 512, 1024, 2048, 3584、上限直前 |
| decode start | 16, 512, 1024, 1339, 2048, 3584 |
| backend | CPU reference、R9700 mandatory、V620 capability-based |

比較:

- finite/shape
- hidden/logits numerical metrics
- exact greedy token
- top-k agreement
- KV/state/cache length
- scheduler progress
- chunk boundary equivalence
- cancel/reset/publish failure recovery

量子化ごとの数値閾値はsource artifactと根拠runを持つversioned policyにし、SQ8閾値をAQ4へ無条件に流用しない。

### 5.2 性能

- prefill p50/p95
- TTFT p50/p95
- decode p50/p95 inter-token latency
- end-to-end latency
- operator/component time
- VRAM baseline/peak/workspace
- fallback count
- actual token/request batch width

同一hardware、binary、model、product、power条件で比較する。

既存承認cellに対し、prefill p50 5%超、p95 10%超の悪化、VRAM上限超過、または新規OOMがあればpromotionを停止する。

### 5.3 production gate

最終evidenceは次を含む。

- direct resident worker
- non-stream/SSE API
- 128/512/1011/2048/3584 prompt TTFT
- short/long-context decode
- EOS/length/overflow
- cancel各phaseと回復request
- 連続request/resource soak
- planned failure/restart
- OpenWebUI browser/Stop/metrics
- exact manifest/binary/product/execution trace hashes

## 6. OOM回避

- GPU推論processは原則1本ずつ実行する。
- chunk plannerはKV、weights、persistent state、workspace、temporary bufferを事前見積りする。
- configurable VRAM headroomを下回るplanは実行前にreject/縮小する。
- full attention matrixを持たず、streamed/chunked attentionを使う。
- logits/hidden/oracleは必要行だけstreaming保存する。
- model/weightを検証caseごとに複製しない。
- OOM結果を小さいcaseの成功で上書きせず、`oom`として保存する。
- V620/R9700の同時大規模検証を避ける。

## 7. 変更対象の想定

新規または抽出候補:

```text
crates/ullm-engine/src/model_graph.rs
crates/ullm-engine/src/model_adapter.rs
crates/ullm-engine/src/state_schema.rs
crates/ullm-engine/src/execution_batch.rs
crates/ullm-engine/src/batch_planner.rs
crates/ullm-engine/src/model_executor.rs
crates/ullm-engine/src/execution_trace.rs
crates/ullm-engine/src/backend_registry.rs
runtime/include/ullm_runtime.h
runtime/src/*batch* / *attention* / *state_scan*
docs/specs/model-graph-v0.1.md
docs/specs/production-execution-trace-v0.1.md
docs/specs/prefill-validation-v0.1.md
```

既存再利用:

- `InferenceSession`とprepare/publish/commit契約
- scheduler/KV allocator/atomic batch advance
- backend dispatch scoring
- AQ4/SQ8 batch projection
- causal/cached-prefix/paged attention kernels
- served-model manifest/activation rollback
- benchmark JSONL、SQ8 oracle、OpenWebUI release gates

## 8. 非目標

- 最初から全model family、全GPU、全quantizationを同時実装しない。
- genericityを理由に動的graph interpreterだけを作り、性能を失わない。
- component benchmarkの改善だけで完了扱いしない。
- request batchingを最初のAQ4 batched prefill完成条件へ混ぜない。
- prefix cacheをstate ownership設計前に追加しない。
- architecture-specific semanticsを無理に曖昧な共通演算へ押し込まない。

## 9. 完成条件

本計画は次をすべて満たしたときに完了する。

1. AQ4/Qwen3.5のOpenWebUI production requestがM>1 generic prefill executorを使う。
2. execution traceがcomponentではなくproduction server pathで最適化implementationを証明する。
3. Qwen3.5 AQ4のprefill promotion minimumを満たす。
4. 長context decodeが改善するか、少なくとも非回帰を証明して別phaseの改善計画を残す。
5. Qwen3/SQ8が同じexecutor/planner/session境界へ移行する。
6. 追加モデルをadapter/state schema/weight binding中心で追加できることを証明する。
7. generic executorにmodel ID文字列分岐がない。
8. correctness、cancel/reset、OOM、resource、OpenWebUI gateが全件成功する。

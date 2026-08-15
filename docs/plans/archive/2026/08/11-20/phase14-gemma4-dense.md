# Phase 14: google/gemma-4-12B Dense text-only

> 状態: complete（2026-08-15、A0-A7・integration review・最終check完了）
> 作成日: 2026-08-15

## 目的

Phase 13で抽出したmodel-neutral prepared execution制御へ、Qwen3.5とは異なる二つ目のproduction model adapterとして
`google/gemma-4-12B` Dense text-onlyを接続する。公式model sourceをimmutableにlockし、model固有のconfig、weight、
normalization、attention、position encoding、activation、logits処理をadapter/providerへ閉じ込める。

本Phaseの主目的はGemma 4を動かすことに加え、共通executorがQwen固有のshape、state、wait/cache policyに依存せず、
別architectureでもPhase 9の高速な実行骨格を再利用できることをproduction pathで証明することである。

## 開始条件

- Phase 13のmodel-neutral fixture、Qwen adapter移行、focused RDNA2/RDNA4 smokeが完了している。
- 共通execution moduleがQwen graph、Qwen定数、tensor名をimportしていない。
- Gemma 4のrepo/revision、license/evidence、config/tokenizer/template、全weight shardをmodel lock手順で固定できる。
- 公式sourceにDense 12Bとして解釈できない差異がある場合、別modelを黙って代用せず事実を記録して再計画する。

Phase 12のMI300X PASSは開始条件にしない。CDNA3実機が未配置でもhostとRDNAの実装を進める。

## スコープ

- official Gemma 4 12B Dense、single GPU、batch 1、text-only、BF16 production path。
- immutable model lock、verified cache、tokenizer、chat templateの有無を含むprompt mode、model alias。
- Gemma model config、weight manifest、graph lowering、model-resident owner、request-local state。
- Phase 13 prepared plan/transition/segment/boundaryを利用するadapter。
- 既存semantic op/providerの再利用と、Gemma意味に必要な最小の新op/provider。
- R9700 full modelをprimaryとし、V620では収容可能なmodel slice/operatorと、明示的に成立する場合だけfull modelを扱う。
- CLI、OpenAI non-stream/SSE、fixed/Unicode/stop、cancellation、cleanup。
- Qwen/Gemma共通最適化bridgeへ渡す短いdirect-engine profile。

次は含めない。

- Gemma MoE、MTP、vision、diffusion。
- Weight/KV NVFP4、KV FP8、runtime自動量子化。
- multi-request batching、multi-GPU、tensor parallel、CDNA3実機claim。
- Gemma専用の独立scheduler、独立generation loop、独自prepared cache/wait loop。
- profileで支配的と確認していないarchitecture/GPU固有の大規模tuning。

## 設計境界

1. `GemmaModelAdapter`はmodel lock/config、tensor名、graph topology、model固有semantic descriptor、publication boundaryを
   生成する。HIP stream/event、raw kernel symbol、VMM pointer、completion pollingを所有しない。
2. 共通executorはGemma名、layer種別定数、head数、vocabulary、weight名をmatchしない。prepared planとrequest binding、
   state transition、boundaryだけを処理する。
3. normalization、RoPE/position encoding、attention variant、activation、embedding/logits tying、soft cap等は公式固定sourceから
   導出し、Qwenの意味を名前だけ変えて再利用しない。意味が同じ場合だけ既存semantic opを使う。
4. tensor layoutの変換やrepackが必要ならmodel load時に一度だけ行い、manifestとresident ownerへ結合する。
   requestごとの全weight変換、host複製、暗黙transposeを行わない。
5. text-linear、embedding、normalization、attention、output処理のdtype/encoding capabilityをprepare時にfail-closedで決める。
   unsupported opをCPU、Qwen近似、別kernel成功へ読み替えない。
6. model-resident weight、prepared plan、workspaceと、request-local token/KV/sampling/cancellationを分離する。

## 受入条件

1. model lockが完全revision、全runtime/evidence file、license、tokenizer、templateの有無、weight fileのhashを固定する。
2. config parserとweight manifestが公式config/indexを独立検証し、missing/extra/duplicate tensor、shape/dtype不一致、未知layerを拒否する。
3. model adapterがQwen moduleをimportせず、共通semantic opとGemma固有opを明示してexecution planへlowerする。
4. Gemma adapterは独自のprepared cache、pending submission owner、completion wait policy、token loopを持たない。
5. host fixtureが小さな非整列shapeと境界前後でgraph topology、binding更新、state publication、forced failure、cancel/dropを確認する。
6. real-weight sliceを独立CPU oracleと比較し、embedding、normalization、attention/position encoding、MLP、logitsの少なくとも一例を通す。
7. R9700 exact `gfx1201`でfull-model prefill/decode/generationをCPU fallback、silent provider fallbackなしでPASSする。
8. V620 exact `gfx1030`はoperator/model sliceをPASSし、full BF16がVRAMへ収まらない場合は未実行をPASSとせず証拠範囲を限定する。
9. fixed、Unicode、stop、CLI、OpenAI non-stream/SSE、disconnect後recoveryでshared generation pathとcleanupを維持する。
10. short-oddと32/32のbounded profileがTTFT、prefill/decode tok/s、TPOT、E2E、resident/peak VRAM、submission、kernel、
    segment/boundaryを記録する。Qwenやllama.cppと条件が異なる比較をparity claimに使わない。
11. affected checks、1回のintegration review、findingだけのfocused re-review、model/runtime/API/main plan/history同期を完了する。

## 実装順序

### P14-A0: source lockとarchitecture inventory

- official repo、requested revision、resolved full SHA、license/model card evidenceを固定する。
- config、generation config、tokenizer、chat templateの有無、direct/indexed safetensorsと全payload fileのtransitive inputをlockする。
- official config/model definitionからlayer topology、attention variant、head grouping、head dim、position encoding、normalization、
  activation、embedding/output tying、logits処理、special tokenを一覧化する。
- Qwen3.5との差を`reusable semantic / new semantic / adapter-only / unsupported`へ分類する。
- full 12B BF16のweight、KV、workspace、temporary bufferのVRAM見積りをR9700/V620ごとに作る。

固定sourceの実態はbase `google/gemma-4-12B`であり、`chat_template.jinja`とsafetensors indexは存在しない。
したがって `model-lock-v2` は単一`model.safetensors`の完全file/header/catalog identityと
`raw-text-only`、`chat_template_path=null`を固定する。`-it`のtemplateを混在させない。

| resident要素 | byte見積り | 備考 |
| --- | ---: | --- |
| loadable text BF16 weight | 23,814,700,640 | tied embeddingを一度だけ保持 |
| known-unconsumed audio/vision | 104,759,808 | text-onlyではGPUへloadしない |
| sliding KV最大 | 335,544,320 | 40 layer、8 head x 256、K/V BF16、window 1024 |
| full KV最大 | 4,294,967,296 | 8 layer、1 head x 512、K/V BF16、262144 token |
| short 4096 full KV | 67,108,864 | A5/A6の初期bring-up範囲 |

最大contextでもtext weight+KVは約26.5 GiBであり、R9700/V620の公称VRAMだけからfull model不可とは判定しない。
runtime/allocator/workspace込みの実測をA5/A6で行い、OOM未実行をPASSへ読み替えない。

### P14-A1: model frontendとadapter contract

- model kind/config型、immutable alias、tokenizer/template選択をfrontend/coreへ追加する。
- generic model registryがQwen/Gemmaをaliasとlock fingerprintから選び、server/CLIがmodel別token loopを持たないようにする。
- adapter入力をverified weight manifest、model config、semantic graph、state/boundary declarationへ限定する。
- tiny synthetic Gemma-like configでunknown field、unsupported variant、overflow、zero/negative、shape関係不整合を拒否する。

### P14-A2: weight manifestとgraph lowering

- safetensors indexからtensor名、shape、dtype、shard/rangeを検証し、layerごとのrequired/optional集合を生成する。
- embedding、normalization、attention projection、MLP、outputのweightをlogical opへ対応づける。
- prefill/decodeのimmutable graphとrequest-local bindingを分け、position、KV長、token row、output bindingだけをtransitionで更新する。
- model固有stateがある場合はgeneration付きtransactional descriptorとして宣言し、成功前にpublishしない。

### P14-A3: semantic opとprovider差分

- 既存RMSNorm、Matmul、elementwise、RoPE/attention、KV、argmaxがGemmaの数値意味と一致するかhost oracleで確認する。
- 不一致はdescriptor optionで正確に表せる場合だけadditiveに拡張し、意味の違うopをQwen modeへ押し込まない。
- 必要な新opはbaseline oracle、public semantic descriptor、HIP registry、exact target providerの順に実装する。
- 非整列M/K/N、head/token境界、NaN/Inf classification、unsupported layout/encoding、aliasを含める。

Direct-scale RMSNormはbackend-neutral contract、Rust/C ABI、HIP baseline provider、fake-HIP数値contract、
exact-target GPU testまでadditiveに接続した。V620 `gfx1030`とR9700 `gfx1201`で幅
`1/3/17/255/256/257/3839/3840/3841/4095/4096`、row `1/3`、zero/正/負scaleを独立BF16-FP32
oracleへ通し、11/11 case、fallbackなし、`max_abs=0`、`max_rel=0`を確認した。これはDirect RMSNorm
providerだけの証拠であり、Gemma full graph/full model PASSを意味しない。

shared elementwise ABIへscalar multiply、GELU-tanh multiply、tanh softcapを既存ID不変のadditive opとして追加した。
V620 `gfx1030`とR9700 `gfx1201`で各opを長さ
`1/3/17/255/256/257/3839/3840/3841/262144`の独立BF16-FP32 oracleへ通し、両targetとも
30/30 operation、fallbackなし、cleanup anomalyなし、全要素exact一致を確認した。Gemma graphの
embedding/layer scale、GELU、logit softcapはこれらのshared semantic kindへ対応した。この証拠も各単体opに限定し、
full graph PASSを意味しない。

dual RoPEはQwen fused mRoPEへ流用せず、split-halfのfrequency分母、active rotary次元、theta、Q/K head数、
絶対position範囲を持つbackend-neutral contractを追加した。draft HIP kernelはsliding
`Hq=16/Hkv=8/D=256/rotary=256/theta=10000`とfull
`Hq=16/Hkv=1/D=512/rotary=128/theta=1000000`を同じ明示contractで扱う。V620 exact `gfx1030`と
R9700 exact `gfx1201`で各variantのM=`1/3/17`、position `0/255/262127`開始をCPU oracleへ比較し、
6/6 case、fallbackなし、inactive次元bit-exact、`atol=rtol=0.03125`内、`max_abs=0.0214844`を確認した。
public C ABI registry、safe Rust owner、Phase 13 owned execution bridgeへ接続し、通常のpublic static library経由でも
同じ7 caseをV620/R9700へ通した。両targetともfallbackなし、inactive次元bit-exact、`max_abs=0.0214844`である。
この結果はsplit-half RoPE単体の証拠であり、Gemma full graph/full model PASSを意味しない。

sliding/full attentionもBF16 Q/K/V、GQA head grouping、scale `1.0`、FP32 softmax、inclusive window
`max(0, position + 1 - window)`を持つbackend-neutral contractとdraft HIP kernelへ分離した。V620
`gfx1030`とR9700 `gfx1201`でsliding `Hkv=8/D=256/window=1024`のshortおよびposition
1022開始window境界、full `Hkv=1/D=512`のM=`3/17`を独立two-pass CPU softmax oracleへ比較した。
両targetとも4/4 case、fallbackなし、`atol=0.015625/rtol=0.03125`内、`max_abs=0.000244141`だった。
public C ABI registry、safe Rust owner、Phase 13 owned execution bridgeへ接続した。通常のpublic static library経由で
非整列`M=3/Hq=3/Hkv=1/D=6`を加えた5 caseを両targetへ通し、fallbackなし、exact device symbol、
`max_abs=0.000244141`を確認した。providerはrequest-owned BF16 K/V bufferのpublished prefixを入力とし、
未確定tailの可視性はA4のtransactional `committed_length`で制御する。この結果もattention単体に限定する。

### P14-A4: model-neutral executor統合

- Gemma graphをPhase 13 execution plan/transitionへlowerし、同じprepared cache、segment owner、completion、auditを使う。
- KV/state publication、terminal logits/argmax readback、cancel/error boundaryをadapterから宣言する。
- plan cache keyへmodel fingerprint、descriptor/layout、binding identityを含め、Qwen/Gemma間の誤再利用を拒否する。
- forced submit/query failure、timeout、dropでowner lifetime、rollback/poison、resident model再利用を確認する。

Phase 13接続チェックリストは次を正とする。

1. Gemma固有nodeと各node後の`ExecutionBoundaryKind`から`PreparedPlanNode<GemmaGraphNode>`列を生成し、
   `PreparedExecutionPlan::new`へ渡す。共通plan側へGemma enumのmatchを追加しない。
2. request admissionごとにtoken数、開始position、binding generation、state generationを検証し、
   `PreparedTransition::new`でoverflowを拒否する。stateful cacheには`dynamic_identity()`、stateless cacheには
   `stateless_identity()`を使う。
3. semantic descriptorとowned bindingをmodel adapterで構築し、exact descriptor/view/buffer/accessが再利用可能な場合だけ
   `PreparedCachePolicy::Reusable`を宣言する。position encoding等の交換不能なmetadataは`Transient`にする。
4. model-resident ownerごとにprepared cacheを分離し、model fingerprint、resident allocation、binding generationが異なる
   Qwen/Gemma entryを共有しない。
5. semantic、causal attention、linear/state submissionのownerを共通segmentへ移し、KV/state publicationとterminal
   readbackだけをadapterからboundaryとして宣言する。adapter内へper-op waitまたは独自flush loopを作らない。
6. request開始から最終state/output公開まで共通transaction guardを保持し、cancel/error/drop/pendingではcommitしない。
   成功後の共通auditからbackend/target、submission/kernel、fallback、segment/boundaryをservice/CLI証跡へ写す。

structural graphは共通`PreparedExecutionPlan`へnode順と2 boundaryを保ったままlowerし、graphの
token/start/expected lengthとbinding/state generationから共通`PreparedTransition`を生成するhost contractまで接続した。
RoPE/attentionのpublic provider submissionは共通owned execution bridgeへ接続した。request-local KV publication ownerは
共通transaction guardを用い、state-publicationとterminal-readbackの両boundary成功後だけ`committed_length`と
state generationを進める。非整列3/17 token、binding更新、capacity/stale start、同時transition、boundary順序、
forced drop、cancelをhost contractで確認した。

全958 graph nodeをexact tensor view、semantic descriptor、buffer backingへ一対一でmaterializeした。model weight、定数、
workspace、token/position、request K/V、reshape/rotary/prefix aliasを区別し、decode K/V tailはattentionが読む同一
request-state backingのnonzero-offset viewへ`Copy`する。ordered queueでは通常nodeをretainし、adapterが宣言した
state-publicationとterminal-readbackだけでwaitする。Argmax readbackと両boundary成功後にだけtransactionをcommitする。
model-resident ownerは23,814,729,316 byteのweight/定数を一度だけuploadし、requestごとにKV/workspaceを新設して
終了時にゼロへ戻す。prefill/decode間はweight、定数、KV、queueを再利用し、transition-local bindingだけを更新する。

### P14-A5: real-weight sliceとGPU bring-up

- verified cacheからraw非保存sliceを抽出し、independent higher-precision oracleで主要opと小graphを比較する。
- R9700で最小layer、複数layer、短いprefill/decodeの順に進み、provider/fallback/auditを確認する。
- V620は収容可能なslice/operatorを同じoracleへ通し、architecture共通providerの回帰を検査する。
- model download、slice、raw logits/profileをrepositoryへ追加せず、hash/recipe/bounded summaryだけを残す。

official verified cacheを毎回full-file検証した後、tied embeddingの3実rowをcompact tableとしてbit-exact gatherし、
locked final norm weight 3,840要素をDirect RMSNormの3非整列rowへ適用するfocused evidence runnerを追加した。
さらにlayer 0のgate/up/down実weightを非整列`M=3/K=17/N=3`等のcompact MLPへ通し、GELU-tanh multiplyと
tied embedding由来の小logits matmulへ接続した。同じlayerの実Q/K/V weightを`M=3/K=17/N=18|6`でprojectし、
実q/k norm先頭6要素、unit-scale v norm、split-half RoPE、sliding attentionまで合計15 operationを連結した。
V620 exact `gfx1030`とR9700 exact `gfx1201`で
全operation、fallbackなし、cleanup anomalyなしを確認した。
embeddingはbit-exact、RMSNormはBF16-FP32 oracleの`atol=rtol=0.03125`内（`max_abs=0.0625`、
`max_scaled_rel=0.006451613`）、real-weight matmul、compact q/k/v norm、RoPE、attentionは両targetとも
`max_abs=0`である。これはbounded/compact sliceであり、full dimensionのsingle layer、複数layer、decode state再利用は
後続full graph証拠と区別する。

R9700 exact `gfx1201`では公式cacheを再度full-file検証し、23,814,700,640-byte text weightをtensor-sized allocationへ
一度だけuploadした。48 layer、958 semantic nodeをprefill 1回+decode 7回実行し、8 tokenすべて固定参照
`258882`と一致した。各transitionは1,054 submission/kernel、2 segment/boundaryで、累積8,432 dispatch、16 boundary、
fallbackなしだった。peak accountedは23,843,578,492 byte、model resident peakは23,814,729,316 byte、
request-state peakは5,849,088 byte、最終cleanupはゼロである。最初に23.8 GBを単一bufferへ詰める案はpublic runtimeの
bounded single-allocation contractで安全に拒否され、packed weight identityを維持したtensor-sized allocationへ修正した。

### P14-A6: full model、generation、service

- R9700へ12B BF16を一度loadし、fixed token、Unicode、stop、短いmulti-turn、連続requestを実行する。
- reference logits/tokenと許容誤差を固定し、単に自然な文章が出たことをcorrectness oracleにしない。
- CLIとOpenAI non-stream/SSE、reasoning非対応profile、disconnect/cancel/recoveryをshared serviceでsmokeする。
- request終了、model unload、process終了時のVRAM/resourceを確認する。

CLI raw prompt `Hello`はBOS込み`[2,9259]`から`[236764,108,236777]`を生成し、外部の固定reference token列と
3/3一致した。executionは3 transition、3,162 submission/kernel、6 segment/boundary、exact `gfx1201`、fallbackなし、
cleanup anomalyなしだった。Gemmaはlocked chat templateを持たないため、CLI messagesは引き続きfail-closedとする。

OpenAI serverはmodel kindをreviewed lockから選び、Gemmaでは明示的なraw transcript
`Role: content\n...Assistant:`をtokenizerへ渡す。Qwen/Gemmaでscheduler、generation loop、non-stream/SSE framing、stop、
cancellationを共有する。実R9700の同一resident processでfixed、Unicode SSE、stop string、短いmulti-turn、連続requestを
PASSした。100-token SSEをclient側200 ms timeoutで切断したrequestはcancelledとなり、その直後のrecovery requestが成功した。
5 completed+1 cancelled requestの各cleanupはrequest-state/workspaceとも0、shutdown後は全category 0、retryable/durable
cleanupとも0だった。integration reviewでOpenAI既定samplingがterminal logits未公開のため失敗するgapを検出し、
Argmax完了後・transaction commit前に最終BF16 vocabulary rowだけをbounded readbackする経路を追加した。実R9700で
temperature省略のserver requestとCLI `temperature=1/top_p=0.9`がshared samplerを通り、fallback/cleanupなしで成功した。
base modelのreasoning mode/historyだけは引き続き明示的に拒否する。

### P14-A7: performance bridgeとcloseout

- R9700 short-odd、32/32をO1で取得し、可能なV620 caseは証拠範囲を明記して取得する。
- Qwenと共通のhost launch、M=1 matvec、MLP、RMSNorm、attention providerを比較し、Q2候補を順位付けする。
- Gemma固有tuningを開始せず、共通candidateと固有残差を分けてhistoryへ渡す。
- affected final checks、integration review、plan/history/main-plan/runtime/model-lock文書を同期し、本planをarchiveする。

同一resident uploadを再利用するO1 direct-engine profileをR9700 exact `gfx1201`で取得した。`3/17`はTTFT
998,810,379 ns、prefill 3.019 tok/s、decode 13.774 tok/s、decode TPOT 71.486/72.556/73.545 ms
（min/median/max）、E2E 2.160 s、peak accounted 23,867,610,772 byteだった。`32/32`はTTFT 88,082,259 ns、
prefill 406.642 tok/s、decode 13.434 tok/s、TPOT 73.442/74.400/75.506 ms、E2E 2.396 s、peak accounted
24,216,250,864 byteだった。両laneともsubmission=kernel（17,918/33,728）、boundary 34/64、fallbackなし、
request cleanup 0、最終全runtime cleanup 0である。比較条件が異なるQwen/llama.cppとのparity claimは行わない。

## 計測lane

| lane | 内容 | 通常の使用 |
| --- | --- | --- |
| G14-H | config/manifest/adapter/cache/failure host contract | 各work unit |
| G14-S | real-weight single-op/small-graph slice | semantic変更時 |
| G14-R | R9700 short prefill/decode/full-model smoke | integration単位 |
| G14-V | V620 operator/slice、収容時だけfull model | affected provider確認 |
| G14-P | short-odd、32/32 O1 | A7と性能変更時 |
| G14-I | CLI、non-stream、SSE、disconnect | A6/A7で一回 |

## 再計画条件

- official Dense 12B sourceが固定できない、license/evidenceが不明、remote codeなしでconfig/weight意味を確定できない場合は
  別Gemmaを代用せず、lock/frontend以外のmodel-bound実装を止める。
- full modelがR9700へ安全に収まらない場合はOOMを繰り返さず、slice evidenceと必要memoryを記録して量子化後の再実行へ分ける。
- 共通executorへGemma固有branchが増える、adapterが独自wait/cacheを必要とする場合はPhase 13境界を見直す。
- 同じwork unitの2回reject、1時間以上の機能進捗停止、検証/docs 30%超、見積り1.5倍超では追加runを止めて記録する。

## 完了確認

- integration reviewは1回実施した。correctness blockerだったterminal logits未公開による既定sampling失敗を、最終語彙行の
  bounded readbackとshared sampler接続で修正した。release evidenceの不整合だったC++ format/source hashとRust binの
  dependency closureも同期し、findingだけをfocused re-reviewした。
- host laneはH0 `513/513`、H1 `421/421`、H2 `36/36`をPASSした。workspace Rust test/clippy、C++ format/static、
  manifest/schema/workflow、matrix registration、Rust dependency closureもPASSした。
- GPU証拠はR9700 exact `gfx1201` full model/generation/service/profile、V620 exact `gfx1030` operator/real-weight sliceへ
  範囲を限定する。CPU fallback、silent provider fallback、timeout、zero selectionをGPU PASSへ読み替えていない。
- 計画の11受入条件を満たしたため本planをarchiveし、次のlocal forward queueをQwen/Gemma共通RDNA性能bridgeとする。

[対応する履歴](../../../../../history/2026/08/11-20/phase14-gemma4-dense.md)

# Phase 14: google/gemma-4-12B Dense text-only

> 状態: planned
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
- immutable model lock、verified cache、tokenizer/chat template、model alias。
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

1. model lockが完全revision、全runtime/evidence file、license、tokenizer/template、weight shardのhashを固定する。
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
- config、generation config、tokenizer、chat template、safetensors indexと全shardのtransitive inputをlockする。
- official config/model definitionからlayer topology、attention variant、head grouping、head dim、position encoding、normalization、
  activation、embedding/output tying、logits処理、special tokenを一覧化する。
- Qwen3.5との差を`reusable semantic / new semantic / adapter-only / unsupported`へ分類する。
- full 12B BF16のweight、KV、workspace、temporary bufferのVRAM見積りをR9700/V620ごとに作る。

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

### P14-A4: model-neutral executor統合

- Gemma graphをPhase 13 execution plan/transitionへlowerし、同じprepared cache、segment owner、completion、auditを使う。
- KV/state publication、terminal logits/argmax readback、cancel/error boundaryをadapterから宣言する。
- plan cache keyへmodel fingerprint、descriptor/layout、binding identityを含め、Qwen/Gemma間の誤再利用を拒否する。
- forced submit/query failure、timeout、dropでowner lifetime、rollback/poison、resident model再利用を確認する。

### P14-A5: real-weight sliceとGPU bring-up

- verified cacheからraw非保存sliceを抽出し、independent higher-precision oracleで主要opと小graphを比較する。
- R9700で最小layer、複数layer、短いprefill/decodeの順に進み、provider/fallback/auditを確認する。
- V620は収容可能なslice/operatorを同じoracleへ通し、architecture共通providerの回帰を検査する。
- model download、slice、raw logits/profileをrepositoryへ追加せず、hash/recipe/bounded summaryだけを残す。

### P14-A6: full model、generation、service

- R9700へ12B BF16を一度loadし、fixed token、Unicode、stop、短いmulti-turn、連続requestを実行する。
- reference logits/tokenと許容誤差を固定し、単に自然な文章が出たことをcorrectness oracleにしない。
- CLIとOpenAI non-stream/SSE、reasoning非対応profile、disconnect/cancel/recoveryをshared serviceでsmokeする。
- request終了、model unload、process終了時のVRAM/resourceを確認する。

### P14-A7: performance bridgeとcloseout

- R9700 short-odd、32/32をO1で取得し、可能なV620 caseは証拠範囲を明記して取得する。
- Qwenと共通のhost launch、M=1 matvec、MLP、RMSNorm、attention providerを比較し、Q2候補を順位付けする。
- Gemma固有tuningを開始せず、共通candidateと固有残差を分けてhistoryへ渡す。
- affected final checks、integration review、plan/history/main-plan/runtime/model-lock文書を同期し、本planをarchiveする。

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

[対応する履歴](../../../../../history/2026/08/11-20/phase14-gemma4-dense.md)

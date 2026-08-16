# Phase 19: Qwen3.5 MoE text-only production path

> 状態: completed（2026-08-16）
> 作成日: 2026-08-16

## 目的

Qwen3.5 Denseで実装済みのfull attention、Gated DeltaNet、tokenizer/chat template、generation service、
FP8/MXFP4 encoding境界を再利用し、Qwen3.5-35B-A3Bのsparse MoE text-only modelを単一AMD GPUの
通常CLI/OpenAI APIから実行できるproduction pathにする。

MoE対応は「256 expertをすべてdense計算するpath」ではない。routerが選んだtokenごとのtop-8 routed
expertと1 shared expertだけを実行し、decodeとprefillで別のwork organizationを持つ。router、expert
packing、dynamic activation quantization、weighted reduceを独立NumPy oracleと実GPUで検証し、CPU fallbackや
host-side expert numerical executionを通常経路にしない。

## ユーザー決定とPhase境界

- Phase 19はGemma 4 MoEではなくQwen3.5 MoEをprimaryにする。現行Gemma 4 12Bはconfig上MoE無効で、
  31B artifactはlocal 32 GiBの単一GPUにworkspace/KV込みで収容できない。Qwen3.5 MoEは既存の
  Qwen3.5 Dense GDN/attention実行とPhase 16Fのlow-bit descriptorを再利用できる。
- primary candidateは[`amd/Qwen3.5-35B-A3B-MXFP4`](https://huggingface.co/amd/Qwen3.5-35B-A3B-MXFP4)
  のexact revision、text-only componentとする。同artifactはQwen公開FP8 modelからAMD Quarkでrouted expertを
  OCP MXFP4化したmixed recipeであり、全tensorをMXFP4と仮定せずartifact metadataとinventoryを正とする。
- architecture/configとupstream semantic controlは
  [`Qwen/Qwen3.5-35B-A3B-FP8`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8)のexact revisionに固定する。
  公式FP8全weightは37 GiB級でlocal 32 GiBのfull residentに使わず、config、tensor schema、独立slice
  oracleとlineage照合に使う。
- Phase 19のproduction範囲はtext-only、target-only、single request、batch 1、single GPUとする。MoE modelへの
  vision、MTP、request batching、expert/tensor parallelを同時導入しない。
- Phase 20のGGUF writer/readerを先行実装しない。Phase 19は現行safetensors/compressed-tensors importerから
  container-neutral model/recipe descriptorを生成し、Phase 20が同じdescriptorをGGUFから生成できるようにする。

## 外部参照とprovenance境界

- Qwen/AMD model cardとlock対象artifactはmodel semantics、lineage、quantization recipeの正本にする。
- llama.cppはproject方針どおり直接reuse候補である。実際にMoE実装をcopy/adapt/portする場合は、対象path、
  固定commit、hash、変更内容、noticeを`docs/provenance/README.md`に記録し、import commitをrelease前に解決する。
- ROCm/ATOM、vLLM、SGLang、LMDeployはreader-onlyとする。router/expertの入出力contract、shape、性能分解の
  調査に限定し、source expression、control flow、kernel構造をsLLMへcopy・adapt・portしない。reader記録と
  implementationを分けるが、別agentの使用は必須にしない。
- 固定local sourceは[source-lock](../../../../../references/source-lock.md)と
  [inference-engine reader方針](../../../../../references/inference-engines.md)を正とする。Phase中にsourceを更新する場合は
  model/artifact lockと別の変更として記録する。

## 固定対象と開始baseline

### Modelとartifact

- architecture: `Qwen3_5MoeForConditionalGeneration`、text-only language component。
- primary shape: hidden 2048、40 layer、256 routed expert、tokenごとにtop-8、routed expert intermediate 512、
  shared expert intermediate 512、vocabulary 248,320。attention scheduleは3 GDN + 1 full attentionの10回反復。
- primary storage/compute recipe: artifact inventoryにあるOCP MXFP4、FP8、BF16/F32/ignoreの混在。routerと
  shared-expert gateを量子化対象と推測せず、manifestが指定するdtypeとscale contractを厳密にbindする。
- exact revision、file SHA-256/size、index/config/tokenizer/chat template、base FP8 revision、AMD Quark recipe/version、licenseを
  model lockへ固定する。branch `main`やmodel card表示名だけで実行しない。

### GPU targetと収容性

- primary: R9700 exact `gfx1201`、32 GiB。
- secondary: V620 exact `gfx1030`、32 GiB。
- A0でtext-only resident weight、最小KV/state、request workspace、HIP runtime reserveを同時に見積もり、実uploadで
  peak/current bytesを確認する。収容できない場合はCPU offload、部分resident、multi-GPUを暗黙追加せず、
  同一modelの別の提供元low-bit artifact候補またはPhase範囲をユーザーと再計画する。
- AMD model cardのvendor検証targetはMI300/MI350/MI355であり、RDNA対応を証明しない。sLLMの
  packed software providerをexact RDNA targetで独立にfail-closed検証する。

### 開始baselineと未達点

- Qwen3.5 Denseはembedding、RMSNorm、full attention、GDN/linear state、dense MLP、LM head、CLI/serverを実行済み。
- Phase 16Fはcontainer-neutral encoding/recipe descriptor、OCP MXFP4/MXFP8 schema/import boundary、NVFP4/FP8 mixed
  full-model pathを実装済み。OCP MXFP4 MoE expertのfull-model grouped executionは未実装。
- 共通execution graphにrouter、top-k、expert dispatch/combineのsemantic nodeがなく、Qwen graph/config/tensor namingも
  Dense 2B/4B/9Bの固定shapeを前提にする部分が残る。
- Phase 18 MTP providerはDense 4Bのみ性能採用済み。MoE modelのMTP tensor/graphをPhase 19の完了条件にしない。

## MoE semantic contract

### Router

1. pre-MLP normalized hidden rowにBF16/F32 router weightを適用し、256 expert scoreを得る。
2. artifact/configで指定されたsoftmaxを適用し、各tokenのtop-8 expert IDをscore降順で選ぶ。同値は
   expert ID昇順の固定tie-breakとし、host/GPU、decode/prefillで選択を変えない。
3. selected weightは提供元semanticsどおりtop-k内でnormalizeする。router auxiliary lossはtraining用であり
   inference outputへ加えない。
4. router outputは`expert_ids[M,8]`、`expert_weights[M,8]`、expertごとのcount/offsetを持ち、
   out-of-range ID、nonfinite score/weight、count/offset不整合をfail closedにする。

### Routed expertとshared expert

1. routed expertは選択されたtoken-expert pairだけを`gate/up -> SiLU(gate) * up -> down`で実行する。
2. OCP MXFP4のpacked E2M1、E8M0 block scale、group 32、dynamic activation MXFP4をNVFP4/FP8と別encodingにし、
   scale、packing、paddingを推測で共有しない。artifactが別recipeを指定するtensorはそちらを優先する。
3. routed outputはrouter weightで乗算し、original token orderへdeterministicにreduceする。expert groupingの順と
   加算順は数値contractに固定する。
4. shared expertは毎tokenで一回実行し、独立shared-expert gateのsigmoidを乗じてrouted sumへ加える。
   routed expert ID空間へ暗黙に追加せず、recipeが指定するdtype/providerを使う。

### Decodeとprefill

- decode `M=1`はrouterの8選択とshared expertだけをresident weightから直接実行する。request/tokenごとの
  allocation、expert weight upload、CPU sort、256 expert全件のzero-masked GEMMを通常経路にしない。
- prefill `M>1`はselected pairをexpert IDごとにstable group化し、expertごとの可変row grouped Matmulを行う。
  outputはoriginal tokenとtop-k orderへscatter/reduceする。expert count `0/1/non-aligned/high skew`とM境界を必ず試す。
- decode/prefillともroutingとexpert数値contractを共有するが、kernel/providerとworkspace layoutは分けて計測・採用する。

## 受入条件

受入条件は実装前の本節で固定する。数値・state・artifact解釈の欠陥はblockerとし、後から新しい
process gateを追加しない。

### Correctness blocker

1. exact model lockがconfig/index/tensor inventory/quantization recipe/lineage/licenseを検証し、欠落、余分、重複、
   誤shape、誤dtype、誤scale、誤expert IDをload前に拒否する。
2. NumPy router oracleとGPUがfixed/random/tie/nonfinite、M=`1/2/3/7/8/31/32/33`でtop-8 IDをexactに一致させ、
   score/normalized weightを明示誤差budget内にする。selected expert以外を実行しないことをdispatch auditで確認する。
3. OCP MXFP4 decode、non-aligned prefill、shared expert、weighted combineの各stageを独立NumPy decoder/oracleと比較する。
   packed bytes、scale、dynamic activation、intermediate、final MoE outputを段階別に照合し、別formatへの暗黙fallbackで通さない。
4. full layerで`input norm -> router -> routed/shared experts -> residual`を照合し、first/middle/last layerの
   hidden/logits digestから最初の不一致stageを特定できる。
5. R9700/V620でtext-only prefill + decode + generationをHIP-only、fallback false、selected case数非0、cleanup 0で実行する。
   CPU emulation、host expert execution、timeout、crash、OOMをGPU PASSにしない。
6. fixed/Unicode/code/stop、greedy/seeded sampling、CLI/OpenAI non-stream/SSE、連続request、cancel/recovery、shutdownで
   token/finish/usage/framingとrequest-local GDN/KV stateの分離を維持する。
7. 既存Qwen3.5 Dense 2B/4B/9B、Dense MTP、vision、FP8/NVFP4 pathの影響範囲testを通し、MoE schemaを
   Dense modelと誤検出しない。

### Product統合

- ユーザーはmodel directoryだけを指定し、runtimeがMoE architecture、mixed recipe、exact targetからproviderを
  自動選択する。MoE flag、low-bit opt-in、起動コマンド差、通常警告を追加しない。
- vision/MTP tensorをtext-only residentへuploadせず、モデルに含まれることだけで未検証のMoE vision/MTPを
  自動有効化しない。
- 破損artifact、未対応recipe、実行不能targetはwarning付き継続や別dtype変換ではなくerrorにする。

### 性能と採用

- 一律の必達倍率は設けない。decode/prefillを分け、router、gather/group、gate/up、activation、down、combine、
  shared expert、attention/GDN、host submission/sync、TTFT/TPOT/token/s、resident/peak/workspaceを記録する。
- production providerは独立oracleに合格し、decodeでselected 8 + sharedだけを実行し、prefillでactive pair数に応じた
  grouped workになることを必須にする。256 expert全件実行、requestごとのweight upload、CPU numerical routingは採用しない。
- 同一binary/model lock/prompt/token budgetで反復し、median、MAD、p10/p90、expert load skewを記録する。
  screeningの最良runだけで高速化を主張しない。

## 実装・検証順序

### P19-A0: artifact lock、収容性、reference記録

- Qwen公式FP8とAMD MXFP4のexact revisionsをresolveし、model lock、file inventory、lineage、license、
  quantization metadata、text/MTP/vision component boundaryを作る。
- shard indexからtext-onlyの必須tensor、storage bytes、alignment/padding、resident planを算出し、R9700/V620で
  minimum viable contextを持つupload/admission probeを行う。raw model、shard、traceをGit管理しない。
- Qwen model card/config、AMD recipe、llama.cpp direct-reuse候補、reader-only engineの調査範囲を記録する。
  A0終了時にprimary artifact、revision、target、実収容bytesを履歴へ固定する。

### P19-A1: MoE config、tensor schema、container-neutral recipe

- Dense Qwen configと分離した`Qwen35MoeConfig`を追加し、layer schedule、expert count/top-k、shared expert、
  intermediate size、MTP/visionセクションをstrictに検証する。
- per-layer router、packed gate/up/down expert、shared expert、shared gate、scale/tensor-scaleをexpert IDとprojection role付きで
  inventory化する。shard順やlexical名順でexpert IDを推測しない。
- Phase 16Fのrecipe descriptorをMoEのprojection/expert axisへ拡張し、safetensors importerと将来GGUF readerが
  同じverified load planを生成できる境界にする。

### P19-A2: routerとdispatch metadata oracle

- Python+NumPyでBF16/F32 router Matmul、softmax、stable top-8、renormalization、shared gate sigmoidを実装する。
- Rustにmodel-neutral sparse routing descriptor、expert ID/weight/count/offset buffer contract、shape/alignment validationを追加する。
- HIP router/top-kをbaselineから実装し、CPU sortやD2H decisionを正常経路にしない。tie、nonfinite、
  expert 0/255、non-aligned M、extreme skewをfocused testに含める。

### P19-A3: OCP MXFP4 expert numerical baseline

- artifactのactual packed bytesとscaleから一expert gate/up/downをNumPyでdecodeし、dynamic activation MXFP4、
  Matmul、SiLU-mul、down、router weight、shared expert、combineをstageごとにoracle化する。
- HIPにdecode M=1のselected-expert providerと、prefillの可変row grouped providerを分けて追加する。
  Phase 16FのOCP encodingと既存FP8/BF16 providerを再利用し、NVFP4との書式混同を避ける。
- baseline correctness後にだけgate/up fusion、packed activation reuse、selected pair grouping、workspace reuseを行う。
  first/middle/last expert IDとK/M非整列caseを含める。

### P19-A4: graphとQwen3.5 MoE adapter

- common execution graphへ`SparseMoe` semantic boundaryを追加し、router/gather/grouped expert/combineのowned buffers、
  access mode、completion、workspace lifetimeを明示する。Qwen固有のtop-k、shared expert、tensor namingはadapterに残す。
- Qwen3.5 graph builderをreviewed config固有shapeへ一般化し、MoE 40 layerのattention/GDN後MLPを
  sparse MoE nodeへ置き換える。Dense graphの同じsemantic opは変更しない。
- text-only load planはvision/MTP tensorをresidentから除外し、router/expert/shared tensorは全layerをmodel resident lifetimeに
  固定する。requestごとのexpert loadを禁止する。

### P19-A5: full-model direct executionとstate

- exact primary artifactでembeddingから40 layer、final norm/LM headまでtext prefill/decodeを直結する。
- GDN recurrent stateとfull-attention KVは既存opaque request stateを使い、MoE workspaceはrequest stateやKVに混ぜない。
- layerごとのrouter histogram、selected pair数、expert provider、dispatch、fallback、workspace/current/peakをaudit可能にする。
  raw routing/logits/trace/model sliceはrepositoryに入れない。

### P19-A6: CLI/server統合

- 現行model pathからMoE config/recipeを自動検出し、Denseと同じCLI `generate`とOpenAI Chat Completionsで
  text-only generationを実行する。
- non-stream/SSE、greedy/sampling、stop/EOS/max token、cancel/drop/recovery、continuous requests、shutdown cleanupを
  既存generation serviceの上で維持する。
- unsupported vision/MTP requestは数値provider fallbackをせず、対応境界がわかるerrorにする。

### P19-A7: GPU correctness、性能、採用

- R9700/V620でrouter matrix、selected expert slice、full layer、full-model fixed generationをfail-closedに実行する。
  evidenceはexact target/UUID、ROCm/toolchain、source/build identity、model lock、provider、selected count、digest、fallback、
  cleanupを持つ。
- decode/prefillを分け、fixed short/code/long promptでTTFT、TPOT、token/s、router/expert/attention内訳、
  expert skew、dispatch/sync、resident/peak/workspaceを反復測定する。
- correctness後にprofileで支配的な箇所だけを最適化し、各変更は現行のtarget/case別noise-envelope
  採用要件で判定する。機能成立と高速化の採否を混同しない。

### P19-A8: integration reviewとcloseout

- 影響test、host contract、actual target build/GPU evidence、markdown/link/diff checkを行い、integration reviewを1回行う。
  findingが変わった箇所だけfocused re-reviewし、docs-only closeout stageを追加しない。
- main plan、runtime、model lock、GPU/software compatibility、provenance、履歴を実際の結果に同期し、
  planとhistoryの相互linkを確認してarchiveする。
- Phase 20へ渡すのはMoE config、tensor inventory、expert-axis recipe descriptor、verified load plan、tokenizer/chat metadataであり、
  GGUF writer/readerや別の残機能をPhase 19へ混ぜない。

## 検証matrix

| lane | case | 必須証拠 |
| --- | --- | --- |
| schema/lock | missing/extra tensor、expert 0/255、誤shape/dtype/scale、lineage | load前拒否、lock/inventory digest |
| router host/GPU | M=`1/2/3/7/8/31/32/33`、tie/nonfinite/skew | top-8 ID exact、weight budget、count/offset |
| MXFP4 expert | first/middle/last expert、K/M非整列 | packed/scale/activation/intermediate/output oracle |
| combine/shared | duplicate token pairs、shared gate 0/0.5/1近傍 | fixed reduce order、sigmoid/shared sum |
| full layer | first/middle/last layer、GDN/full-attention後 | hidden digest、router histogram、fallback false |
| full model | fixed/Unicode/code/stop、greedy/seeded sampling | logits/token/finish/usage、HIP-only、cleanup 0 |
| service | CLI、non-stream、SSE、cancel/recovery/shutdown | framing、state分離、memory cleanup |
| performance | decode/prefill、short/code/long | TTFT/TPOT/token/s、stage内訳、skew、dispatch/sync、VRAM |

## 非対象

- Gemma 4 MoE、Qwen3.5 122B/397B、Kimi K3 full-model architecture、DeepSeek/MiniMax。
- Qwen3.5 MoEのvision、video、MTP/speculative generation、それらの同時実行。
- request/continuous batchingのscheduler実装、chunked prefill、cross-request prefix cache、簡易永続化。
- tensor/expert parallel、multi-GPU、Infinity Fabric/RDMA、CPU offload、partial expert residency。
- GGUF writer/reader、safetensors廃止、新しいユーザー向けcontainer。これらはPhase 20のGGUF統一で扱う。
- BF16/FP8 sourceからのsLLM独自MoE PTQ、提供元artifactの品質をsLLM converter KLD基準で再判定すること。

## 停止・再計画条件

- primary low-bit artifactがtext-only minimum resident/workspaceとして単一32 GiB GPUに収容できない場合、
  CPU offload/multi-GPUへ自動拡大せず、別artifact候補またはPhase範囲をユーザーと再計画する。
- artifact metadataからrouter normalization、expert packing、scale、shared expert recipeを一意に固定できない場合、
  実行を進めず提供元正本とinventoryを追加確認する。
- router/expert providerが独立oracleに合わない場合、top-1や最終tokenだけで通さず、最初の不一致stageへ戻る。
- 同じwork unitの2回reject、review時間が実装時間超、1時間以上の機能進捗停止、検証/docsが30%超、
  見積り1.5倍超、受入条件変更時は追加探索を止め、同じwork unitを再計画する。

## Closeoutで必要な結論

- どのmodel/artifact/revision/mixed recipeを、どのexact targetでfull resident実行したか。
- router top-8、OCP MXFP4 expert、shared expert、weighted combineが独立oracleとどこまで一致したか。
- decode/prefillでactive expert work、dispatch/sync、TTFT/TPOT/token/s、resident/peak/workspaceはどうなったか。
- CLI/APIのユーザー操作を増やさず、Dense、MoE、low-bit recipeを正しく内部選択できたか。
- Phase 20 GGUFへ渡すcontainer-neutral metadata、expert-axis tensor inventory、recipe descriptorは何か。

## Closeout結果

- `amd/Qwen3.5-35B-A3B-MXFP4` exact revision `2e19c6576db91e5d5a93455415619262218bf8a1`を
  text-only mixed OCP MXFP4 artifactとして固定し、semantic lineageを`Qwen/Qwen3.5-35B-A3B-FP8`
  revision `9d1823d2dee688a6b25e77009dc727688c44936e`へ固定した。verified planは493 entry、digest
  `sha256:f96a3389cfaca4ab947fe060ccd6f048d078946e704464277d87019a13fb7ae4`である。
- NumPy/HIP router matrixとactual expert matrixをR9700/V620でPASSした。actual matrixはlayer 0/19/39、
  expert 0–7/124–131/248–255、M=1/3/7を含み、最大誤差`1.86265e-9`、fallback 0だった。
  routerのM=1/2/3/7/8/31/32/33ではactive expert 8〜166、最大expert count 1〜3を記録し、
  all-tie skew caseはactive expert 8、最大count 3となってstable ID tie-breakとgroup offsetを維持した。
  検証中にE2M1 code 7/15（±6）を0へdecodeしていた欠陥を検出・修正し、修正後は上記matrixとfull generationを再実行した。
- exact R9700 `gfx1201`とV620 `gfx1030`で40層full-model prefill/decode/replayをHIP-onlyでPASSした。
  residentは22,009,574,016 byte、peakは22,230,758,892 byte、active pairはprefill 960、decode 320、
  SparseMoe submissionは各40で、256 expert全件実行やCPU numerical routingはない。
- 2 warmup + 11 measuredの中央値はR9700がprefill/decode 216.258/204.198 ms、V620が
  537.832/370.711 msだった。R9700 MADは0.501/0.514 ms、V620 MADは1.747/0.202 msである。
- 通常CLIとOpenAI non-stream/SSEで英語、Unicode、EOS/stop、usage、連続requestを確認した。stream切断requestは
  `cancelled`となり直後のrequestが回復し、明示seed付きsamplingは同一text/usageを再現した。shutdownはmodel/request/workspace、
  retryable cleanup、durable quarantineがすべて0だった。MoE/low-bit専用flagや通常警告は追加していない。
- Phase 20へは`Qwen35MoeConfig`、expert-axis tensor/plane inventory、mixed recipe、verified load plan、
  tokenizer/chat metadataとSHA-256 model identityを渡す。GGUF writer/readerはPhase 19へ含めなかった。
- host側は`cargo fmt --check`、`cargo clippy --workspace --all-targets -- -D warnings`、`cargo test --workspace`、
  Python oracle 7件をPASSした。両GPUのrouter matrixは最終sourceで再build・再実行した。integration reviewでは
  shard pathの検証後re-openとMoE C ABI layout probe漏れをblockerとして修正し、現行sourceの24.6 GB artifact全identity検証と
  R9700 full-model focused re-reviewをPASSした。OpenAI `seed`も固定OpenAPIどおりsigned `int64`へ補正した。

[対応する履歴](../../../../../history/2026/08/11-20/phase19-qwen35-moe.md)

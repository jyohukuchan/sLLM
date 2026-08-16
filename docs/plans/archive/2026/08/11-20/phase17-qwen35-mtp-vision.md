# Phase 17: Qwen3.5 MTP、vision

> 状態: completed (2026-08-16)
> 作成日: 2026-08-16

## 目的

既にlock済みの`Qwen/Qwen3.5-4B` revisionで、known-unconsumedのMTP 15 tensorとvision 297 tensorを正式に消費する。
同じmodel lock、Phase 13のmodel-neutral execution、Phase 16のopaque KV encoding、既存generation serviceを再利用し、
最初にtext-only MTPを独立して完成させ、その後にimage processor/vision encoder/projector/multimodal promptを実装する。
両者を同じwork unitで同時debugしない。

MTPは生成結果の意味を変える別model modeではなく、target modelのdecodeをspeculativeに進める内部providerとする。
visionはCLIとOpenAI-compatible Chat Completionsの明示的なimage contentを追加する。量子化artifactの内部分類と同様、
provider成熟度を通常応答の警告へ出さない。ただしvision入力のwire shapeは新機能なので、text-only profileとの互換性を保った
versioned API拡張として文書化する。

## 固定model、processor、API source

### Qwen3.5-4B

- repository: `Qwen/Qwen3.5-4B`
- resolved revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- model lock fingerprint: `sha256:f143d7b504170d071c77818105f7a07dc0297c6bea0c61a5404b071fed0c1fae`
- text 426 tensor、vision 297 tensor、MTP 15 tensorを既にexact catalog/shape/dtypeでlock済み。
- MTPは1 layer、shared embedding/lm head、`mtp.fc.weight=[2560,5120]`、独立embeddingなし。
- visionはdepth 24、hidden 1024、intermediate 4096、output 2560、patch projection/position/24 block/mergerを含む。

### Processor contract

- 同じrevisionの`preprocessor_config.json`を正とし、pixel area `65,536..=16,777,216`、patch size 16、
  temporal patch 2、merge 2、RGB mean/stdとも`[0.5,0.5,0.5]`を固定する。
- `vision_start=248053`、`vision_end=248054`、`vision_pad=248055`、`image_pad=248056`をlocked tokenizerから使う。
- resize、aspect ratio、padding、patch order、position ID、merge orderはreader記録と独立NumPy/Pillow oracleで固定する。
  PyTorchは使用しない。

### OpenAI-compatible image input

- 2026-08-16取得のOpenAI OpenAPI `2.3.0`と
  [Images and vision guide](https://developers.openai.com/api/docs/guides/images-vision)をsourceとして、Chat Completionsの
  user message `content` arrayに`{"type":"text","text":...}`と
  `{"type":"image_url","image_url":{"url":...}}`を受理するprofile revisionを追加する。
- 初期serverはBase64 `data:image/...` URLだけを受理する。HTTP(S) remote fetchとFiles API IDは実装せず、outbound fetch/SSRF
  boundaryを新設しない。CLIはmaintainerが明示したlocal pathを読める。unsupported sourceはfieldを無視せずerrorにする。
- text string inputは従来どおり受理し、既存profile v1のrequest/response/SSEを壊さない。image outputやResponses APIは対象外とする。

## Work unit 1: MTP

### MTP受入条件

1. MTP 15 tensorを`known-unconsumed`からcomponent-enabled時のrequiredへ昇格し、missing/extra/wrong shape/dtype、
   dedicated embedding/tie矛盾、unknown MTP depthをload前に拒否する。MTP disabled text pathのlock互換性は維持する。
2. MTP graphはtarget hidden stateとcandidate token embeddingをlocked norm/fc/one-layer decoderへ通し、shared tied embeddingで
   draft logitsを生成する。Q/K/V packing、RoPE、GQA、causal positionを独立oracleと照合する。
3. draft/verify/acceptをmodel-neutralなspeculative transactionとして表し、target KV、MTP state、sampling history、stop matcher、
   usageを同じaccepted prefixだけ進める。rejected/unused draft tailを公開KVへ残さない。
4. greedyではMTP on/offのvisible token IDs、finish reason、usage、stop境界がbit-exactに一致する。EOS/stop/max token、
   candidate全accept、先頭reject、途中reject、cancelを含める。
5. stochastic samplingはtarget分布を保存するrejection/residual samplingを実装し、seeded random-source seamでNumPy oracleと
   acceptance decision、residual distribution、RNG消費順を照合する。greedyだけの実装をsampling対応と呼ばない。
6. target forward failure、MTP failure、partial verify、cancel/dropではtransactionをcommitせず、同request stateを再利用しない。
   target-onlyへruntime fallbackして部分結果を成功扱いにしない。
7. MTP providerはartifact/configとtarget capabilityから内部選択する。明示provider overrideはbenchmark用だけにし、通常CLI/APIに
   MTP許可flagや品質警告を要求しない。性能上採用しないtargetではtarget-only providerを内部選択できる。

### P17-M0: reader、model lock、oracle

- official config/model card、固定tensor catalog、必要ならno-copy reference engineのreader記録からMTP forwardとspeculative algorithmを固定する。
- MTP 15 tensorのexact range/hash、shared embedding、head shape、position、draft countをcomponent manifestへ追加する。
- tiny NumPy oracleでnorm/fc/attention/MLP/logitsと、greedy/stochastic verify-acceptを別々に実装する。
- token length/draft widthの`0/1/2/3/7`、position `255/256/257`、vocab tie、overflow、stale generationをhost testへ追加する。

### P17-M1: MTP graphとreal-weight slice

- Qwen adapterにMTP graph nodeを追加し、共通executorへQwen tensor名やMTP固有enumを逆流させない。
- MTP residentはmain model residentと同じmodel fingerprint/weight lifetimeへ結び、requestごとにuploadし直さない。
- synthetic tiny、fixed real-weight layer、production shapeの順にR9700/V620でoperatorを実行し、selected provider、fallback false、
  output error、cleanupを記録する。

### P17-M2: speculative transactionとgeneration統合

- target-only generation loopを、draft proposal、batched target verify、accepted prefix publication、reject token samplingへ分解する。
- target KVへ直接複数candidateをappendせず、shadow/staged stateでverifyしてaccepted prefixだけをatomic publishする。
  Phase 16のFP16/FP8/NVFP4 descriptorをopaqueに扱い、encoding別rollback実装を上位へ作らない。
- stop/EOSがdraft中に現れるcase、UTF-8/stop prefix保留、max token、sampling penalties/history、usageをaccepted token基準で統合する。
- non-streamとSSEが同じaccepted token列をpublishし、rejected draftをclientへ送らない。

### P17-M3: correctness、service、performance

- fixed/Unicode/code/stop/long promptでMTP off/onのgreedy token完全一致をR9700/V620で確認する。
- seeded samplingはoracle decisionとdistribution sanityをhost/tiny GPUで確認し、同seedのservice出力を比較する。
- FP16 KVをprimaryにし、Phase 16 FP8 KVで少なくとも一つのfull generation/cancel caseを通してopaque stateを回帰する。
- draft acceptance length、target forward数/emitted token、TTFT、TPOT、token/s、resident/peak、MTP overheadを測る。
  一律の必達倍率は置かず、明確に遅いtargetでは内部target-only優先を維持する。
- MTP単独のCLI/OpenAI non-stream/SSE、連続request、disconnect/cancel/recovery、shutdown cleanupを完了してからvisionへ進む。

## Work unit 2: vision

### Vision受入条件

1. decoderはPNG/JPEG/WebP/non-animated GIFのうち実装した形式をmagic bytesとdecode結果で検証し、extension/MIMEだけを信頼しない。
   animated、malformed、decompression bomb、pixel area範囲外、overflow、unsupported colorspaceをbounded errorにする。
2. EXIF orientationを適用するか無視するかを明示し、RGB変換、resize、padding、normalize、patch/merge順をdeterministicにする。
   decoded imageとprocessor intermediateをraw artifactとしてGitへ追加しない。
3. processor outputはNumPy/Pillow oracleと、pixel area `65,535/65,536/65,537`および
   `16,777,215/16,777,216/16,777,217`、odd aspect、1/2 images、patch/merge境界で比較する。
4. vision 297 tensorをcomponent-enabled時のrequiredへ昇格し、patch embed、24 block、merger、position embeddingを
   exact shape/dtypeでconsumeする。text-only loadでは不要なvision residentを強制しない。
5. image embedding、visual token count、placeholder、mRoPE positionをtyped multimodal promptへ結び、text tokenizerが
   image bytesやuntrusted placeholder文字列を直接解釈しない。
6. image encode/projectはrequest中一度だけ行い、decodeごとに再実行しない。cacheはmodel fingerprint、processor config、
   exact image digest、layoutへ結び、request cancel/dropで安全に解放する。異なるrequest間共有は本Phaseで行わない。
7. OpenAI content arrayのunknown part、image on non-user role、remote URL、file ID、unsupported `detail`、重複/空textをstrictに
   検証する。既存text string input、error envelope、SSE framingを回帰させない。
8. fixed image/questionでreference processor intermediate、vision slice、projected token、full output token/taskを比較し、
   「自然な説明」を唯一のoracleにしない。

### P17-V0: API/processor contractとfixture

- OpenAPI source identity、対応content part、data URL grammar、supported MIME、encoded/decoded byte、image count、pixel area、
  total visual tokenの上限をversioned compatibility profileへ固定する。
- `detail`はQwen processorの意味と一致する値だけを受理する。初期はomitted/`auto`を同じlocked processor contractとし、
  OpenAI model固有の`low/high/original`を無言で近似しない。
- CLI local fileとserver data URLを共通のbounded image byte/decoded RGB型へlowerする。serverはHTTP(S)をfetchしない。
- licenseが明確な小さなgenerated fixtureをPNG/JPEG/WebP/GIF、odd aspect、orientation、malformed/bomb negativeとして用意する。

### P17-V1: Rust image decodeとpreprocessor

- Rustでbounded decode、orientation/RGB、aspect-preserving resize、normalize、temporal duplicate/pad、patchify、merge metadataを実装する。
- processor計算はchecked arithmeticで、encoded body、decoded bytes、pixel count、patch count、visual token/context budgetを
  allocation前に検証する。
- Python+NumPy/Pillow oracleは同じ実装構造を写さず、最終pixel/patch/position値とhashを比較する。
- text-only requestがimage decoder/vision GPU bufferを確保しないことをhost allocation auditで確認する。

### P17-V2: vision graph、provider、projector

- patch Conv3d、position embedding、24 transformer block、merger/projectorをsemantic opへlowerし、既存linear/norm/attention
  providerを再利用する。vision専用の別executor/wait loopを作らない。
- tiny synthetic、first/mid/last block real-weight slice、full image encodeの順にR9700/V620でbring-upする。
- bias、fused QKV、GELU-tanh、position interpolation/lookup、spatial mergeの数値oracleとprovider auditを固定する。
- vision resident/request workspaceとtext residentを区別し、same model instanceの連続text-only/image requestでlifetimeを確認する。

### P17-V3: multimodal prompt、full model、service

- content partsの順序を保ってtext tokenとvisual placeholder/embeddingを組み立て、special token、mRoPE position、usageを一致させる。
- single image、two images、text-before/after/between-images、odd aspect、Unicode questionをfull Qwen3.5-4Bで実行する。
- reference processorのpixel/patch/token count、vision/projected slice、fixed full output tokenまたはtask resultを段階別に照合する。
- CLI local path、OpenAI non-stream/SSE data URL、stop、連続request、disconnect/cancel/recovery、malformed input後のrecoveryを通す。
- image processing、vision encode、text prefill、decodeを分けてTTFT、TPOT、token/s、resident/peakを記録する。

### P17-V4: MTPとの最終統合、文書、closeout

- MTPとvisionを個別にPASSした後だけ、image prompt + MTP greedyを実行し、MTP off/onのvisible token/finish/usageを一致させる。
- stochastic MTPをproduction採用したtargetではseeded image promptも1 case通す。未採用targetへmatrixを無条件に増やさない。
- Phase 16Fのquantized modelはtext-only smokeだけを回帰し、vision weight quantizationまで本Phaseへ広げない。
- model lock、runtime、OpenAI compatibility、GPU/software compatibility、main plan、historyを同期する。1回のintegration reviewと
  findingだけのfocused re-review後にarchiveする。後続順序は2026-08-16訂正によりPhase 18 MTP性能統合、Phase 19 MoEとする。

## 計測matrix

| work unit | case | 主指標 |
| --- | --- | --- |
| MTP format/graph | 15 tensor、tiny/real/production shape | output error、provider、fallback |
| MTP algorithm | all accept/first-mid reject/EOS/stop/cancel | token、accepted prefix、RNG、state |
| MTP model | fixed/Unicode/code/long、FP16+代表FP8 KV | exact greedy、sampling、TPOT、VRAM |
| processor | area/patch/merge境界、odd aspect、1/2 images | pixel/patch hash、token count、error |
| vision graph | first/mid/last/full image | slice error、projected token、provider |
| multimodal | part順序、single/two image、CLI/API | output token/task、usage、SSE、cleanup |
| combined | image + MTP | off/on token一致、acceptance、latency |

## 非対象

- video/audio、image generation、remote HTTP(S) image fetch、Files API、Responses API。
- cross-request vision embedding cache、prefix cache、continuous batching、multi-GPU vision。
- Gemma 4 MTP/vision、Qwen MoE、Kimi K3、DeepSeek/MiniMax architecture。
- vision weightのFP8/NVFP4/MXFP4量子化、vision専用PTQ/QAT。
- MTPを使ったmodel品質変更、draft-only tokenの公開、approximate sampling。

## 停止・再計画条件

- official MTP semanticsとfixed tensor/configからdraft/verify contractを一意に確定できない場合、推測実装せずreader/source固定へ戻る。
- accepted prefixだけをopaque KVへatomic publishできない場合、encoding別rollbackを上位へ追加せずshadow transactionを再設計する。
- processor oracle、vision slice、full outputのどの段階で差が生じたか分離できない場合、full model再試行を止めて最初の不一致段階へ戻る。
- remote image取得が必要になった場合、本Phaseへ暗黙追加せず、outbound policy/resource/security contractを別途ユーザーと決める。
- 同じwork unitの2回reject、review時間が実装時間超、1時間以上の機能進捗停止、検証/docs 30%超、見積り1.5倍超、
  acceptance変更時は追加探索を止めて同じwork unitを再計画する。

## Closeout結果

- MTP 15 tensor manifest/graph、shared embedding、target hidden-state seam、greedy/stochastic verifier、opaque transactionを実装した。
  canonical 2 targetのreal-weight draft/verifyは一致したが、逐次verifyはtarget forwardを削減しないため、通常providerはtarget-onlyを選ぶ。
- vision 297 tensor、bounded decoder/processor、HIP dense projection、24 block、merger/projector、typed embedding replacement、3-axis
  mRoPE、lazy server resident、CLI local path、Chat Completions data URLを実装した。
- V620/R9700でMTPとvision/multimodal textをHIP-only、fallbackなし、deterministic、cleanup 0でPASSした。MTPを性能採用したtargetが
  ないためimage+MTP production matrixは非選択providerの追加gateにせず、imageはtarget-only text providerでcloseoutした。
- low-bit modelのtext-only挙動を維持し、visionはBF16 text artifactに限定した。remote fetch、Files API、video、low-bit visionは非対象のまま。

## 2026-08-16訂正

- MTPの完了範囲はcomponent graph/verifier/real-weight evidenceまでであり、generation serviceのMTP off/on倍率、
  batched target verify、target forward削減は未確認だった。通常逐次decodeとの数値同一性と最低限の高速化は
  [Phase 18 archive](phase18-mtp-exact-sequential-speedup.md)へ移管し、MoEはPhase 19へ繰り下げる。

[対応する履歴](../../../../../history/2026/08/11-20/phase17-qwen35-mtp-vision.md)

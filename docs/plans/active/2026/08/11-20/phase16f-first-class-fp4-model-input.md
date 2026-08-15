# Phase 16F: first-class FP4 model input

> 状態: planned
> 作成日: 2026-08-16

## 目的

提供元が公開するNVFP4 PTQ/QAT checkpointと、OCP MXFP4/MXFP8でQATまたはnative公開されたmodelを、
BF16/FP8と同じmodel指定操作で読み込める公式経路へ置く。artifact metadataからencoding、mixed-precision recipe、
target別providerを自動選択し、低bit専用の許可flag、確認prompt、通常警告を追加しない。

本Phaseは「sLLMがBF16から高品質なPTQを生成できること」と「提供元low-bit artifactを正しく実行できること」を分離する。
Phase 15/15Qで不採用となったS0/U0/O0 converter candidateのKLDを、提供元checkpoint、encoding decoder、runtime providerの
不採用理由へ転用しない。提供元artifactは同じartifactのreference runtime、提供元評価、task oracle、sLLM operator/full-model
evidenceで判定する。BF16正本がないnative low-bit modelへBF16 KLDを要求しない。

## 実行順序をPhase 16後に置く理由

primaryにする`unsloth/gemma-4-12b-it-NVFP4`は、MLP W4A4、attention W8A8、KV FP8、残りBF16というmixed recipeである。
Phase 16より前に実行するとFP16 KVへの置換を黙認するか、artifactの一部だけを実行してfull supportと呼ぶことになる。
Phase 16でFP8 KVを完成させた後にW4A4/attention/loaderを統合し、公開recipeを一体で検証する。

## 固定するsourceと証拠範囲

### Primary full-model: Unsloth Gemma 4 12B NVFP4

- repository: [`unsloth/gemma-4-12b-it-NVFP4`](https://huggingface.co/unsloth/gemma-4-12b-it-NVFP4)
- resolved revision: `b1f649734b34aa5575b03d186abd1b9be3d0d5c4`
- `model.safetensors`: `9,304,966,064` byte、SHA-256
  `7c2ee23298e7c3a9247e8947597dca5a38f8b791a0322487466d2bfad8ce704b`
- Phase 15Qでheader/catalog、BF16 source 349 tensor、MLP 144 NVFP4 tensorを固定済み。Phase 16FではMLP input
  activation、attention FP8、KV FP8、ignore/BF16を含む公開mixed recipe全体を対象にする。
- 12B artifactとruntime workspaceがR9700 32 GiBに収まり、現行Gemma 4 adapterを再利用できるため、AMD full-modelの主証拠とする。

### Secondary schema/model-lock: NVIDIA Gemma 4 31B NVFP4

- repository: [`nvidia/Gemma-4-31B-IT-NVFP4`](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4)
- 2026-08-16時点resolved revision: `4135a98a9b728a548947683219633b25682223ac`
- 4 safetensors shard合計`32,633,477,808` byte。R9700 32 GiBへworkspace込みで収容できないため、header/index/config/
  quantization metadataのbounded compatibilityとreference-runtime laneに使い、local single-GPU full-model PASSを主張しない。
- model cardはModelOpt 0.42.0によるweight/activation NVFP4と公開benchmarkを記録する。公開値をsLLM実行結果へ転用しない。

### MXFP4/MXFP8 contract: OCPとKimi K3

- format source: [OCP Microscaling Formats (MX) v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
  のMXFP4 E2M1、block 32、E8M0 scale、およびMXFP8の明示encoding。
- model source: [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3)、2026-08-16時点resolved revision
  `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`。model cardはQATしたMXFP4 weight/MXFP8 activationを記録する。
- Kimi K3は2.8T級MoEかつ現行未対応architectureである。Phase 16Fはconfig/index/model lock、encoding/import descriptor、
  tiny oracle/provider bindingを完成させるが、Kimi full-model supportを主張しない。MoE/architectureはPhase 18以降へ渡す。

実装開始時にHub identity、license、file listを再解決し、上記完全SHAと一致するbyteだけを使用する。floating `main`の更新を
同じlockとして扱わない。model、shard、raw trace/logits、large sliceをGitへ追加しない。

## 製品・architecture契約

- model inputはcontainer-neutralな`QuantizedTensorEncoding`、tensor role、mixed recipe、provider requirementへlowerする。
  safetensors/compressed-tensorsは移行期間のimporter、Phase 19のGGUFは同じ内部descriptorを生成する。
- NVFP4 1D/2D、row/column layout、block-16、E4M3 block scale、global scale、packed nibble orderをexact schemaで表す。
  Phase 15 sidecar v1、Unsloth compressed-tensors、NVIDIA ModelOptをfield名だけで同一視しない。
- MXFP4はOCP block-32/E8M0であり、NVFP4 block-16/E4M3/FP32 outer scaleとは別encodingにする。MXFP8 activationも
  OCP encodingとscale layoutを独立に表す。
- mixed recipeはtensor selector、weight encoding、input activation encoding、output dtype、attention/KV policyを固定する。
  ignore対象を勝手に量子化せず、unknown/missing/overlap selectorをload前に拒否する。
- exact targetに実装providerがなければerrorにする。別dtypeへの変換、W4A16代用、FP16 KV代用を通常起動で行わない。
- RDNA2/RDNA4/CDNA3の現行hardwareでnative FP4命令を証明していないproviderは`packed-dequant`と呼び、nativeと表記しない。

## 受入条件

### format、loader、runtime correctness

1. NVFP4/MXFP4/MXFP8の全code、rounding、saturation、NaN/Inf、zero、scale式、block tail、row/column packingを
   Python+NumPy独立decoderで固定し、importer/runtimeとbit/数値比較する。
2. source revision、全使用file hash/size、safetensors header/catalog、quant config、selector、tokenizer/template、processorを
   model lockへ含める。range overflow、duplicate tensor、unknown encoding、metadata矛盾をresident allocation前に拒否する。
3. normal CLI/serverはmodel directory/artifactを同じ引数で指定し、quantized modelだけにmanifest/provider/許可flagを要求しない。
   現行sidecar/provider optionはdevelopment overrideとして残せるが、primary acceptanceでは使用しない。
4. W4A4はactivationを各linear直前に指定NVFP4 layoutへ量子化し、packed weight/activationを直接consumeする。requestごとの
   BF16 weight展開、別provider fallbackを禁止し、activation quantization時間とlinear時間を別にauditする。
5. mixed recipeのattention W8A8、MLP W4A4、Phase 16 FP8 KV、BF16/ignored tensorが、artifact selectorと一対一で一致する。
   W4A16/FP16 KVで通した結果をfull artifact PASSにしない。
6. operator/sliceはK/N `15/16/17`・`31/32/33`、M `1/3/7/17/32/33`、production shape、scale極値、odd tailを含め、
   CPU oracleとexact `gfx1030`/`gfx1201`で比較する。fallback、selected provider、nonfiniteを記録する。
7. primary full-modelはfixed/Unicode/code/math/long prompt、teacher-forced logits、greedy、sampling、stop、連続request、OpenAI
   non-stream/SSE、cancel/recoveryを実行し、same-artifact reference、提供元評価、task oracleと証拠scopeを分けて記録する。
8. runtime成熟度、target provider順位、sLLM converter品質、model evidenceを独立fieldに保持する。内部状態を通常応答の警告や
   別起動modeへ変換しない。

### 品質判定

- primary提供元checkpointにはPhase 15のBF16 KLD `0.05`をconverter gateとして適用しない。同じquantized artifactの
  documented reference runtimeでfixed token/logitまたはtask resultを取得し、sLLMのtask/token挙動と照合する。
- 提供元公開benchmarkはmodel品質の外部根拠だが、sLLM runtime correctness PASSの代用にはしない。sLLM独自のbounded task set、
  nonfinite、first divergence、finish reasonを記録する。
- exact reference runtimeを利用できるhardwareがないdraftではoperator/full graph AMD evidenceを進められるが、reference未実行を
  PASSへ読み替えない。Phase closeout時は適合hardware上のblack-box reference実行を取得するか、内部model evidenceを
  `experimental`と明記してscopeを限定する。いずれもユーザー操作や通常警告は変えない。
- sLLM PTQ converterを同時に改善しない。新converter candidateを提案する場合は別のBF16 source/KLD laneへ分ける。

### 性能・memory判定

- W4A4 activation quantization、linear kernel、FP8 attention、FP8 KV append/attention、host/launch、E2Eをdecode/prefillに分解する。
- BF16、現行W4A16、artifact faithful mixed recipeを同じmodel/token/targetで比較し、resident/peak VRAM、TTFT、TPOT、token/sを記録する。
- 一律の必達倍率は置かない。targetごとの自動provider順位はfresh 3 warmup + 10 measured、noise envelope、fallbackなし、
  memory削減と複雑性から決める。正しいが遅いproviderを別ユーザーmodeへ隔離せず、内部順位/evidenceで表す。

## 実装・検証順序

### P16F-A0: source lock、schema、recipe inventory

- primary/secondary/MX sourceのrevision、license、file/header/index/configを固定し、tensorごとのencoding/selector表を生成する。
- NVIDIA NVFP4、Unsloth compressed-tensors、OCP MX、Kimi configをfacts-onlyで読み、異なるblock/scale/layoutを別versionへ分ける。
- vLLM/SGLangはdocumented black-box reference commandに限って使用し、no-copy sourceの表現を実装へ持ち込まない。
  llama.cppに該当encodingの直接再利用候補があればprovenance手順で先に比較する。
- primary recipeのMLP/attention/KV/ignore tensor countとresident byteを事前計算し、R9700収容性を確認する。

### P16F-A1: container-neutral encodingとimporter

- `QuantizedTensorEncoding`、scale plane、logical/physical shape、tensor role、recipe selector、source range/hashをmodel-neutralにする。
- current Unsloth importerをfull mixed artifactへ拡張し、model directoryだけから自動検出する。sidecarへの再書き出しを通常loadの
  必須手順にせず、positional readから直接verified resident uploadへ結ぶ。
- NVIDIA 31Bは全payload downloadを必須にせず、locked index/header metadataと利用可能なbounded rangeでschema互換を検証する。
- OCP MXFP4/MXFP8のtiny deterministic fixtureを生成recipeとhash付きで追跡し、Kimi config/indexがencodingを正しく選ぶことを確認する。

### P16F-A2: NVFP4 W4A4 activationとlinear

- artifact指定のper-projection/local block activation scaleを実装し、dynamic quantization結果をresident weight providerへ渡す。
- baselineはpacked value/scaleをtile内でdequant/accumulateし、BF16 outputを返す。W4A16 providerへ偽装せず別descriptor/providerにする。
- decode M=1とprefill M>1を分け、activation quantization dispatch、scale lifetime、workspace再利用、producer-consumer fusion候補を計測する。
- independent oracle、synthetic boundary、Gemma layer 0/mid/final real weightで両canonical GPUを確認する。

### P16F-A3: mixed attention/KVとfull graph統合

- artifact selectorに従いattention projectionを既存FP8 W8A8 path、MLPをW4A4、KVをPhase 16 FP8、残りをBF16へbindする。
- Gemma adapterのgraph topology、Phase 13 transaction、service lifecycleを再利用し、quantized専用executorを複製しない。
- model resident identityへsource lock、header/catalog、recipe digest、encoding/provider、exact targetを含め、異なるartifact間でcacheを共有しない。
- normal CLI/serverからmodel pathだけでloadし、doctor/auditでactual recipe/provider/fallback/bytesを確認する。

### P16F-A4: full-model fidelityと品質

- primary artifactをR9700で一度loadし、fixed prompt/token manifest、複数teacher-forced位置、greedy/samplingを取得する。
- same artifactのdocumented reference runtimeを適合hardwareでblack-box実行し、token IDs、logit取得可能範囲、task result、runtime/versionを固定する。
- Japanese/English/code/math/long-contextを含むbounded task setでnonfinite、first divergence、finish reason、task scoreを記録する。
- V620は収容可能なprimary full-modelまたはreal-weight bounded graphを実行し、evidence範囲をR9700 full supportと分ける。

### P16F-A5: performance、UX、service

- BF16/W4A16/mixed W4A4のdecode/prefill/E2E、resident/peakを同じtoken条件で測る。activation quantizationとlinearの割合を記録する。
- model artifactを替える以外はBF16/FP8/NVFP4で同じgenerate/server操作になることをCLI/HTTP contractで確認する。
- fixed/Unicode/stop、連続request、OpenAI non-stream/SSE、disconnect/cancel/recovery、shutdown cleanupをR9700で通す。
- low-bitを理由とするwarning、確認、quality fieldを通常outputへ追加しない。破損/unsupportedは既存error envelopeでfail closedにする。

### P16F-A6: MX handoff、文書、closeout

- MXFP4/MXFP8 decoder、descriptor、tiny provider boundaryを完成させ、Kimi K3が「encoding unsupported」ではなく
  「architecture/MoE unsupported」と正しく診断されるところまで分離する。
- Phase 18へKimi router/expert/attention/visionとfull-model hardware要件をhandoffし、Phase 19 GGUFへencoding metadata、
  tensor inventory、recipe selectorをhandoffする。
- runtime、model lock、FP4仕様、GPU/software compatibility、provenance、main plan、historyを同期する。1回のintegration reviewと
  findingだけのfocused re-review後にarchiveし、Phase 17へ進む。

## 計測matrix

| level | artifact/encoding | 主case | 指標 |
| --- | --- | --- | --- |
| format | NVFP4/MXFP4/MXFP8 | 全code、scale、row/column、15/16/17・31/32/33 tail | byte、decode、rounding |
| loader | Unsloth/NVIDIA/Kimi metadata | exact、missing、extra、overlap、wrong version | lock、selector、range、diagnostic |
| operator | Unsloth W4A4、attention W8A8 | M=1/3/7/17/32/33、real/production shape | output error、provider、fallback |
| full model | Unsloth Gemma 4 12B mixed | fixed/Unicode/code/math/long、greedy/sampling | token/task、nonfinite、reference |
| service | primary artifact | normal model path、non-stream/SSE/stop/cancel | UX、usage、cleanup |
| performance | BF16/W4A16/W4A4 mixed | decode/prefill/E2E | quant/kernel比、TTFT、TPOT、token/s、VRAM |
| handoff | Kimi K3 MX | config/index/tiny provider | encoding可否とarchitecture未対応の分離 |

## 非対象

- sLLM BF16→NVFP4/MXFP4 PTQ converterの再設計、KLD thresholdの緩和、calibration corpus配布。
- Kimi K3 full model、Gemma/Qwen MoE、multi-GPU、tensor/expert parallel。
- NVIDIA/CUDA kernel、vLLM/SGLang sourceの移植、NVIDIA benchmarkのAMD性能値への転用。
- Phase 19より前の最終GGUF writer/runtime全面移行。内部descriptorはGGUFへ渡せる形にするが、暫定独自containerを増やさない。
- vision/audio、MTP、Responses API。

## 停止・再計画条件

- artifact metadata、independent decoder、runtime decodeのいずれかが一致しない場合、full-model品質比較を無効としてformat mappingへ戻る。
- primary artifactのresident+workspaceがR9700に収まらない場合、host offloadを暗黙追加せず、既存12B recipe内のexact必要tensorと
  workspace lifetimeを再監査する。
- full mixed recipeが未実装のcomponentを要求する場合、そのcomponentをBF16へ置換してPASSにせず、依存Phaseまたは別encodingへ戻る。
- 同じwork unitの2回reject、review時間が実装時間超、1時間以上の機能進捗停止、検証/docs 30%超、見積り1.5倍超、
  acceptance変更時は追加探索を止めて同じwork unitを再計画する。

[対応する履歴](../../../../../history/2026/08/11-20/phase16f-first-class-fp4-model-input.md)

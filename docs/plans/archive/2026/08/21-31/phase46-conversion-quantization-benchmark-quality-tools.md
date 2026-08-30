# Phase 46: conversion・quantization・benchmark・品質評価tool

> 状態: complete（2026-08-27）
> 対象: 変換、量子化、imatrix、split/merge、benchmark、品質評価、bounded debug dump
> 後続: KV品質判定policyだけをPhase 53のdefault採用判定へ渡す。Phase 46全体の完了をPhase 53実装開始のgateにしない。

## 目的

sLLMが受理するmodel artifactを、source identityから実行用GGUF・derived lockまで再現可能に変換し、同じidentity契約で
性能と品質を比較できるtool群を整備する。変換成功、短い生成の完走、単一token一致だけを品質保証へ読み替えず、
perplexity、logit KLD／top-1、task品質、long-context、性能、資源を別のmetricとして保存する。

本Phaseは2026-08-26のユーザー指示により、Phase 53で追加する`kv-fp8-e4-block16`と
`kv-fp8-e5-block16`のdefault判定に必要な評価基盤も所有する。新しいKV encoding、HIP kernel、target selector、
default変更そのものはPhase 53が所有する。

## 正本と固定方針

- 全体順序は[Phase 37以降のロードマップ](../../../../active/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md)、artifact identityは
  [model lock](../../../../../models/model-lock.md)、runtimeのdtype／quantization分離とopaque KVは
  [runtime architecture](../../../../../architecture/runtime.md)を正とする。
- toolはsupported architecture、tensor、dtype、quantization recipeだけをcapabilityとして公開する。未知metadata、欠損tensor、
  shape不一致、未対応recipeを推測で補わずfail closedにする。
- model、raw trace、raw prompt、無制限logit dump、credentialをrepositoryへ追跡しない。追跡summaryはbounded metadataと
  repository外artifactのdigestだけを持つ。
- 品質thresholdはcandidate結果を見てから決めない。FP16 baselineの反復変動と既存品質を先に測り、dataset、seed、metric、
  集約法、許容差、失敗条件をversioned policyへ固定してからcandidateを判定する。
- KV評価のscale方向はtoken内だけとし、tokenを跨ぐper-channel統計・calibrationを実装または評価候補にしない。

## 成果物

1. source／recipe／tool／output identityを固定するconverter、split/merge、LoRA conversion、layout/repack、quantization tool。
2. 共通run manifestとresult schemaを使う`sllm-bench`、perplexity、KLD／top-1、task eval、long-context eval。
3. opt-inかつsize上限付きのtoken／logit／intermediate debug dump。
4. `ci/policy/kv-cache-default-v1.json`と対応schema。Phase 53が同じpolicy digestを参照してGPU別defaultを判定する。
5. toolごとのCLI help、exit code、machine-readable output、再現手順、失敗時のpartial artifact処理。

## 作業単位

### A. 共通identity・schema・出力transaction

1. source repository/revision、使用file size/SHA-256、model lock、tokenizer、converter／runner commit、toolchain、args、recipe、
   output digestを共通manifestへ固定する。path、alias、cache directory、mtimeをidentityの代用にしない。
2. JSON schemaは`schema_version`と`struct_size`相当のadditive拡張規則を持ち、未知必須field、digest不一致、重複tensor、
   non-finite metric、0件選択を拒否する。
3. outputは一時成果物で全検査を終えてからatomic publishする。失敗・cancel・容量超過時は完成名へ公開せず、再実行時に
   stale partialを完成artifactとして再利用しない。
4. compact summaryとrepository外raw evidenceを分け、summaryからsource、binary、model、dataset、policy、raw reportを
   SHA-256で辿れるようにする。

### B. HF→GGUF、split/merge、LoRA、layout/repack

1. model plugin/capabilityがarchitecture、tensor catalog、shape、dtype、metadata mappingを宣言し、general converterは宣言済み
   combinationだけを受理する。Qwen3.5のreviewed pathを最初のintegration対象にする。
2. splitはtensor payloadを分断せず、part順、全体tensor catalog、metadata、各part digestをmanifestへ固定する。mergeは欠損、
   重複、順序違反、別model part、改変partを拒否する。
3. split→mergeはbyte identityを基本とし、container orderingを意図的に正規化する形式ではtensor/metadata semantic identityと
   canonical output digestを検証する。どちらの判定かをreportへ明記する。
4. LoRA conversionはbase model fingerprint、target tensor、A/B orientation、rank、dtype、scale、adapter digestをderived lockへ結合し、
   Phase 45のruntime binding oracleと同じ意味を維持する。
5. execution-ready layout/repackはlogical tensor digest、物理layout、padding、alignment、target semantics、recipe versionを分離し、
   元payloadとderived payloadの関係を検証可能にする。

### C. quantization・imatrix

1. sLLMが実行可能と宣言するBF16／FP8／NVFP4／MXFP4系recipeだけを対象とし、一般的なQ8_0、Q4_K、任意bit幅converterを
   このPhaseへ追加しない。実際にruntime providerがないrecipeは変換成功をproduction supportへ昇格しない。
2. calibration dataset、tokenizer、sample順、seed、context切り方、layer/tensor除外、imatrix accumulator、丸め、clamp、scale、
   non-finite処理をrecipeへ固定する。同じ入力から同じmanifestとpayload digestを再生成できることを確認する。
3. quantized artifactはbounded tensor sliceを独立decode oracleへ照合し、shape、packing、scale、tail、zero、subnormal、最大有限、
   NaN／Inf policyを検証する。
4. quality gateは単一sliceだけにせず、D/Eのmodel-level KLD、top-1、perplexity、task evalへ接続する。

### D. benchmark

1. `sllm-bench`はmodel load、warmup、measurement、E2E、TTFT、TPOT、prefill、decode、request count、parallelism、context、
   sampling、KV encoding、GPU identity、provider、fallback、cleanupを明示する。
2. wall clockとGPU timingを混同せず、中央値、個別反復、棄却反復と理由を保存する。timeout、crash、OOM、0 measured、
   CPU/backend fallbackを性能PASSにしない。
3. HBM/GTTのbefore／peak／settled、model resident、KV logical／physical bytes、workspaceを記録し、値が取れないplatformでは
   `0`で代用せずunsupported／missingを区別する。
4. Phase 53向けにはFP16、現行`fp8`／`fp8-static`／`nvfp4`、新block16候補を同じmodel/input/run boundaryで比較できる
   matrixを用意する。別artifactや別targetの速度をstrict-identical比較と表記しない。

### E. 品質・長context評価

1. tokenizer済みdataset shardと順序、license/provenance、dataset digest、seed、context長、stride、sample上限を固定し、
   perplexityをloss合計・対象token数とともに出力する。
2. FP16 baselineとcandidateのlogitを同一positionで比較し、KLD、top-1 agreement、最大／分位logit差、最初のtoken分岐位置を
   bounded sampleで記録する。token一致だけ、または平均KLDだけでtail failureを隠さない。
3. task evalはtask version、prompt renderer、few-shot、answer parser、metric実装、sample順を固定する。外部networkや更新可能な
   leaderboard値を再現性のあるlocal gateにしない。
4. KV default判定には短contextだけでなく、複数page／capacity境界を跨ぐlong-context continuation／retrieval caseを含め、
   early・middle・tail positionをsampleする。KとV、layer、KV head、block tailのcoverageをreportする。
5. FP16 baselineを複数回測り、metricごとの自然変動、deterministicでないtask、dataset汚染／空集合を先に判定する。
   `kv-cache-default-v1` policyはこのbaseline-only結果からthresholdと再測定規則をfreezeし、そのdigestをcandidate reportへ結合する。

### F. bounded debug dump

1. dumpは明示opt-in、production default offとし、出力directory、総byte、tensor数、token数、logit上位件数、layer／position filterを
   hard limitで制限する。
2. raw prompt／responseは既定で保存せず、token count/digestと必要最小限のreview済みfixtureを使う。authorization header、API key、
   model payload、pointer、device addressをdumpしない。
3. dumpのdtype、shape、layout、endianness、quantization descriptor、scale plane、source submission IDを明示し、packed KVを
   FP16 tensorとして誤読させない。
4. write失敗、disk full、cancel時に推論結果とdebug artifactの成功状態を分離し、partial dumpをvalid reportへ参照しない。

## KV default policyの必須内容

`ci/policy/kv-cache-default-v1.json`は少なくとも次を固定する。

- 対象model lock／derived lock、Qwen3.5-4B BF16を最初のpromotion scopeとすること、未検証modelはFP16を維持すること。
- exact target semantics、ROCm/HIP、code object、wave、binary、KV descriptor、provider、policy自身のdigest。
- FP16 baselineとcandidateのdataset、seed、context matrix、perplexity、KLD、top-1、task、long-context、repeat／aggregation。
- correctness、quality、performance、memory、fallback、cleanupの独立判定。quality PASSで性能FAILを隠さず、その逆も許さない。
- threshold、境界値のinclusive/exclusive、missing/non-finite/0 caseのFAIL規則、再測定回数、candidateを見ずにfreezeした根拠。
- targetごとに`adopt`／`retain-fp16`／`insufficient-evidence`を返し、一targetの不採用で別targetの判定を無効にしないこと。

## 完了条件

- supported Qwen3.5 conversionでsourceからverified GGUF／derived lockまで再現でき、digest、tensor catalog、metadata、recipeを
  fail closedに検証する。unsupported architecture／dtype／recipeのnegative caseを含む。
- split→merge、LoRA、repack、quantizationの各oracleがaligned値だけでなく1、15、16、17、255、256、257等の境界と
  malformed／truncated／duplicate inputを検証する。
- benchmarkと品質toolが共通identity、machine-readable schema、0-case rejection、atomic output、bounded raw evidenceを持つ。
- FP16 baselineだけからKV default policyをfreezeし、同一policyをPhase 53候補へ適用できる。新KV kernelやdefault selectorの
  実装完了をPhase 46の条件にしない。
- affected host tests、schema validation、CLI integration、docs/link checkと一回のintegration reviewをPASSし、採用source、
  既知制約、未対応recipeをmatching historyへ記録する。

## 対象外

- `kv-fp8-e4-block16`／`kv-fp8-e5-block16`のformat、HIP kernel、state ABI、default selector実装。
- tokenを跨ぐper-channel KV scaling、runtime calibration、要求長や空きHBMに応じて途中でKV encodingを変える処理。
- TurboQuant、一般Q8_0／Q4_K、任意architecture converter、remote benchmark service、leaderboard運用。
- model payload、raw prompt、raw profiler trace、巨大logit dumpのrepository格納。

## 停止・再計画条件

- converterが未検証metadataを推測しないと出力できない場合はそのarchitectureをunsupportedのまま残す。
- baseline反復だけで品質metricが安定せずthresholdをfreezeできない場合は、そのmetric／modelをdefault判定から外す理由を記録し、
  candidateに都合のよい閾値を後付けしない。
- tool実装が新しいmodel/runtime semantic、外部execution boundary、任意network accessを必要とする場合は本Phaseを拡大せず別計画にする。

## Phase closeout

完了時に本planをarchiveへ移し、matching historyへtool identity、schema/policy digest、test、既知制約を記録する。
Phase 53は本PhaseのKV policy milestoneを参照するが、converterやLoRA tool等の未完了部分をblockerとして継承しない。

[全体計画](../../../../main-plan.md) /
[Phase 53保存済み計画](phase53-kv-fp8-block16-default-adoption.md) /
[対応する履歴](../../../../../history/2026/08/21-31/phase46-conversion-quantization-benchmark-quality-tools.md)

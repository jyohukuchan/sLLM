# Phase 18: MTP逐次承認・target-only数値同一・最低限の高速化

> 状態: complete
> 作成日: 2026-08-16

## 目的

Phase 17で実装したQwen3.5 MTP tensor/graph/verifierを、通常のgeneration serviceで実際に高速化する内部providerへ完成させる。
MTPはmodel品質やsampling結果を変えるmodeではない。MTPなしの通常逐次target decodeを唯一の数値oracleとし、draft tokenを
先頭から一つずつ承認し、同じtarget weights、dtype、KV encoding、prompt、sampling設定、seedに対して同じ計算結果と
ユーザー可視結果を得る。

性能はcorrectnessと交換しない。target-onlyと異なる数値経路を使って見かけのacceptanceや速度を作らず、single-token decodeと
同じrow arithmeticを維持したserial-equivalent batch、device-side orchestration、prepared execution、staged KVによって
host launch/synchronizationとservice loop overheadを削減する。MTP on/offの倍率を同条件で測り、少なくとも一つのcanonical
GPU targetで反復測定のnoise envelopeを越えるfull-generation改善を得る。

## ユーザー決定と外部参照境界

- 2026-08-16のユーザー決定により、llama.cppのMTP実装を一括でcopy/adapt/portしない。
- llama.cpp issue [#25618](https://github.com/ggml-org/llama.cpp/issues/25618)は、量子化targetでdraft-model型speculationの
  greedy出力がvanillaから分岐し、同じ量子化targetのngram speculationは一致したという回帰事例である。sLLMではこのissueを
  defect classと再現matrixのsourceにだけ使い、llama.cppのMTP control flowを正しさのoracleにしない。
- MTP forward semanticsはfixed Qwen config/tensor、提供元資料、既存の独立NumPy oracleを正とする。target verify/acceptは
  sLLMの通常逐次decodeと独立oracleから設計する。
- 通常CLI/OpenAI APIへMTPの許可flag、opt-in、品質警告を追加しない。provider overrideとMTP off/onは開発・benchmark専用とし、
  通常runtimeはartifact、exact target、検証済み性能tableから内部選択する。

## 開始baselineと未達点

- fixed `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`のMTP 15 tensor、shared
  embedding/output、one-layer draft graphはPhase 17でload・real-weight実行済みである。
- Phase 17 evidence runnerはdraftごとにtarget forwardを逐次実行し、target forward数を減らさない。R9700の記録値
  `target 1968 dispatch / MTP 125 dispatch / 10.879 s`はcomponent runner全体であり、MTP off/on性能比較ではない。
- batched target verify、accepted prefixだけのgeneration-service KV publication、CLI/API MTP path、TTFT/TPOT/token/sの
  paired off/on比較は未完了である。Phase 17の「MTP完了」をこれらの証拠へ拡張解釈しない。
- 一般的なM>1 GEMM/attention providerはM=1通常decodeとkernel、reduction、rounding、KV参照順が変わり得る。
  token一致だけを見てこの差を許容せず、最初の不一致layer/opまで特定できるlockstep harnessを先に作る。

開始時にcurrent mainからfresh target-only/MTP-component baselineを取り直す。履歴値は比較対象の説明にだけ使い、
Phase 18 candidateの性能証拠として再利用しない。

## 固定対象

### Modelと数値経路

- primary model: fixed Qwen3.5-4B、text-only MTP。
- primary correctness/control: BF16 weight、FP16 KV、batch 1。
- quantized regression: 同じfixed modelの既存FP8 W8A8 target pathとFP8 KV代表case。Q4_K等の未対応formatは追加しない。
- NVFP4 KVはopaque transaction境界のfocused caseに含められるが、sLLM製Qwen NVFP4 PTQ artifactをproduction品質modelとして
  再採用しない。
- exact GPU target: R9700 `gfx1201`とV620 `gfx1030`。MI300X `gfx942`は新しい実機が利用可能な場合だけ追加し、
  過去VM evidenceをPhase 18 PASSへ流用しない。

### Generation semantics

- greedy、seeded stochastic sampling、temperature/top-k/top-p、penalty/history、EOS、stop、max token、cancel。
- CLI、OpenAI non-stream/SSE、連続request、disconnect/recovery、shutdown cleanup。
- single request、batch 1。request batching、continuous batching、multi-GPUは非対象。

## target-only同一性contract

### Canonical target step

1. MTP offの通常generation loopが、各accepted prefixに対するcanonical target logits、sampler input、selected token、KV appendを定義する。
2. MTP onでもtarget logitsだけがtoken選択の権限を持つ。draft logits/tokenは計算を先読みする候補であり、直接公開しない。
3. target verifyの各rowは、同じprefixを通常M=1 decodeした場合と同じweight bytes、provider、accumulation/reduction順、rounding、
   attention mask/position、KV encodingを使う。別のM>1 providerはrowごとの結果がこのcontractにbit-exactな場合だけ採用できる。
4. correctness harnessでは各stepのtarget logits bytes、selected token、commit後KV value/scale bytes、sampling history generationを比較する。
   storage paddingや未初期化領域は比較対象から除き、semantic payload範囲を明示する。

### 逐次承認

1. draft列 `d1..dk`に対し、targetのcanonical token `t1..tk`をprefix順に評価・sampleする。
2. `di == ti`の間だけ一つずつ承認する。最初の不一致`r`では`d1..d(r-1)`と`tr`だけを公開し、`dr..dk`および
   それ以降のspeculative stateを破棄する。後続draftを飛び越して承認しない。
3. 全draftを承認した場合だけ、同じcanonical target pathで得たbonus tokenを公開できる。bonus取得の有無でRNG順やusageを変えない。
4. target KV、MTP state、sampler/penalty history、stop matcher、usageは公開token prefixと同じ長さだけatomic commitする。
   partial failure、cancel、dropでは未commit tailを再利用しない。

### Greedyとstochasticの同一性

- greedyはMTP on/offで各target logits、visible token IDs、finish reason、usage、stop境界、commit済みKVをbit-exactにする。
- stochasticでは通常target samplerのRNG streamを唯一のpublic sampling streamとする。MTP proposal用の乱数が必要な場合は
  request sampling RNGと分離し、public RNGを進めない。
- target logitsを通常samplerへ同じ順序で渡し、同seedではMTP offと同じtarget tokenを選ぶ。draftはそのtokenと一致した場合だけ
  承認する。数学的な分布同値だけを理由にrejection/residual samplerで同seed出力の差を許容しない。
- stop/EOS/max-tokenがdraft内に現れた場合もcanonical target tokenを順に処理し、clientへrejected/unused draftを送らない。

## 受入条件

受入条件は実装開始前の本節で固定する。correctnessを満たさない高速candidateは性能値に関係なく棄却する。

### Correctness blocker

1. fixed/Unicode/code/stop/long prompt、全accept、先頭/途中/末尾reject、bonus、EOS、max token、cancelで、MTP on/offの
   canonical logits、token、finish、usage、commit済みKVがcontractどおり一致する。
2. BF16+FP16 KV primaryと、FP8 W8A8+FP8 KV quantized regressionをR9700/V620でfail-closedに実行する。CPU fallback、
   timeout、crash、zero selected caseをPASSにしない。
3. stochasticは複数固定seedでtarget samplerの入力logits、RNG counter、selected token、visible outputをMTP offと一致させる。
4. serial-equivalent batchはM=1逐次oracleとfirst/middle/last layer、attention、representative linear、final logits、KV payloadを
   段階別にbit-exact照合する。差があるproviderを「tokenは同じ」として採用しない。
5. staged KVはaccepted prefixだけをatomic publishし、FP16/FP8/NVFP4 encoding別rollbackを上位generation層へ作らない。
6. target/MTP failure、partial verify、cancel/drop後に同request stateを成功継続せず、次requestがclean stateから正常実行できる。
7. CLI/OpenAI non-stream/SSEが同じaccepted token列を返し、stop/usage/framing/disconnect/recovery/shutdown cleanupを維持する。

### 最低限の高速化と採用条件

- 比較は同一commit/binary、model lock、weight/KV encoding、GPU tuple、resident instance、prompt、sampling、出力token budgetで
  MTP off/onをcounterbalanceする。target-onlyとMTPで異なるmodel artifactやproviderを使わない。
- screening後の採用測定は3 warmup + 10 measuredを基本にし、median、MAD、p10/p90、run-order driftを記録する。
- primary性能caseは十分なdecode長を持つfixed text/code promptとし、TTFT、TPOT、decode token/s、E2E、accepted tokens/proposal、
  target logical rows/output token、target/MTP dispatch、host submission/sync、resident/peak/workspaceを測る。
- speedup倍率は`MTP on decode tokens/s ÷ MTP off decode tokens/s`、latency倍率は`MTP off TPOT ÷ MTP on TPOT`として
  同時に記録する。絶対値だけ、最良runだけ、component runner時間だけで高速化を主張しない。
- 全target共通の固定3% floorは置かない。通常providerとしてMTPをauto-selectするtarget/caseは、full-generationの改善がその
  target/case固有のnoise envelopeを越え、guard caseにnoiseを越える説明不能な退行がないことを必須とする。
- Phase 18完了には、correctness matrixを満たしたうえで少なくとも一つのcanonical targetのprimary caseでMTPをauto-selectできる
  改善を必須とする。改善しないtargetは同じユーザー操作のまま内部target-onlyを選び、未達理由を記録する。
- 他engineのMTP倍率を引用・再測定する場合はmodel、quantization、prompt/output長、GPU、engine commitを記録し、条件が異なる値を
  sLLMの合否基準にしない。「他engine同様の倍率」と主張するのは比較可能なratio evidenceがある場合だけとする。

## 実装・検証順序

### P18-A0: baseline、差分再現、reader/provenance記録

- current target-only generation、Phase 17 component runner、generation serviceのstate/dispatch/latencyをfresh profileする。
- issue #25618から「quantized target、draft-model speculation、greedy vanilla分岐、ngram control」という再現軸だけをreader記録へ固定する。
  upstream MTP source表現、control flow、関数構造をimplementation basisへ持ち込まない。
- BF16/FP8 target、FP16/FP8 KVでMTP offを二回実行し、vanilla自体のdeterminism、provider ID、logits/KV digestを確認する。
- 現行MTP verifierをgeneration pathへ仮接続する前に、target forward/dispatchが出力token当たり何回必要かを式と実測で分解する。

### P18-A1: lockstep target oracleと逐次承認state machine

- target-onlyとMTP verifyを同一accepted prefixから一stepずつ進め、layer/op logits/KV/RNG digestを採取できるtest seamを追加する。
- model-neutral speculative transactionを`proposed / verified / accepted / replacement / committed / aborted`で表し、最初のrejectより
  後をcommitできない型・generation checkを追加する。
- greedyとstochasticのcanonical target samplingを共通関数へ集約し、draft proposalがpublic sampler RNG、penalty history、stop matcherを
  進めないよう分離する。
- tiny host oracleでdraft width `0/1/2/3/7`、accept count各値、stale generation、failure/cancel境界を通してからGPUへ進む。

### P18-A2: serial-equivalent multi-token target verify

- candidate tokensを一つのverify blockとしてlowerするが、各token rowのlinear reduction/roundingをcanonical M=1と同じに保つ。
  複数rowを同時配置してもdot product内の演算順を変えない専用providerまたはgrouped M=1 dispatchを比較する。
- attentionはcandidate間のcausal dependencyを保持し、row `i`がaccepted prefixと`d1..d(i-1)`だけを見る。position/mRoPE、GDN/recurrent
  state、KV scale生成も通常step順と同じにする。
- block全体のtarget KV/stateはshadow allocationまたはversioned staged rangeへ書き、逐次承認後にprefix rangeだけpublishする。
- M=1 bit-exactを満たせないhipBLASLt/GEMM solutionやbatched attentionは棄却する。許容差を緩めて通さない。

### P18-A3: submission・MTP proposal overhead削減

- MTP resident/prepared planをmodel resident lifetimeへ固定し、request/tokenごとのupload、plan構築、allocationを除去する。
- serial-equivalent target blockとMTP proposalをprepared segment/command listへまとめ、host round trip、event query、submissionを減らす。
- draft depth `1/2/3/4/7`を測り、acceptanceと追加MTP costからtarget/case別の内部depthを選ぶ。request APIには公開しない。
- MTP graphのlinear/norm/attentionで既存provider再利用、small-M fusion、workspace reuseをprofileし、target arithmeticへ影響しない
  MTP側だけを独立に最適化する。

### P18-A4: generation service統合

- target-only loopと同じsampler/stop/usage publisherの前へ内部speculative providerを置き、non-stream/SSEで共通transactionを使う。
- accepted tokenを一つずつcanonical orderでpublisherへ渡し、SSE batchingの違いでvisible chunk/token/usage semanticsを変えない。
- provider selectionはexact target、model lock、weight/KV encoding、validated performance rowから内部決定する。未計測tuple、失敗tuple、
  speedupのないtupleはtarget-onlyを選ぶ。
- benchmark専用overrideでoff/on、draft depth、provider IDを固定できるようにするが、通常ユーザー操作やwarningは増やさない。

### P18-A5: GPU correctness integration

- R9700/V620でBF16+FP16 KV、FP8 W8A8+FP8 KVのlockstep、full generation、reject boundaryを実行する。
- fixed/Unicode/code/stop/long、greedy/seeded stochastic、CLI/OpenAI non-stream/SSE、連続request、cancel/recoveryを確認する。
- evidenceはexact target、toolchain、commit/tree、model lock、artifact digest、provider、fallback、selected case count、数値/KV/RNG digest、
  cleanupを記録する。raw logits/trace/modelはrepositoryへ入れない。

### P18-A6: paired performance、採用、closeout

- 各targetでoff/onを同一residentかつcounterbalanced順に測り、component、direct engine、service overheadを分ける。
- acceptance lengthとtarget/MTP work、launch/sync減少、TPOT/token/s倍率の因果が一致することを確認する。速い結果だけを選ばない。
- target/caseごとにMTP auto-selectまたはtarget-onlyを決定し、通常CLI/APIの操作差がないことをsmokeする。
- 1回のintegration reviewとfindingを変更した箇所だけのfocused re-reviewを行う。計画/history/main plan、runtime、model lock、
  compatibility/provenance文書のうち実際に変わった正本を同期し、planをarchiveしてPhase 19 MoEへ進む。

## 計測matrix

| lane | case | 必須比較・指標 |
| --- | --- | --- |
| host transaction | accept `0..k`、reject、bonus、stop/EOS/cancel | token、RNG、history、state generation、commit range |
| target op | M=1 vs serial-equivalent rows `2/3/4/7` | layer/op output bytes、provider、reduction、fallback |
| KV | FP16/FP8、focused NVFP4、capacity boundary | committed value/scale bytes、position、rollback/cleanup |
| full greedy | fixed/Unicode/code/stop/long | logits/token/finish/usage bit-exact、acceptance |
| full sampling | fixed seeds、temperature/top-k/top-p/penalty | sampler input、RNG counter、token/output exact |
| performance | short/code/long、decode 128以上 | TTFT、TPOT、token/s、E2E、off/on倍率、dispatch/sync、VRAM |
| service | CLI、non-stream、SSE、continuous requests/cancel | output、chunk order、usage、recovery、cleanup |

## 非対象

- llama.cpp MTP implementationの一括copy/adapt/port、Q4_K/UD-Q4等の新規量子化format。
- target-onlyと異なる出力を許すapproximate speculation、draft tokenの直接公開、品質を変えるMTP mode。
- 数学的なdistribution同値だけを保証して同seed出力差を許すrejection/residual sampling。
- Gemma 4 MTP、MoE、vision+MTP性能、request/continuous batching、multi-GPU、cross-request prefix cache。
- 他engineの絶対速度への追随をPhase 18のhard gateにすること。比較可能なMTP倍率は参考値として記録する。

## 停止・再計画条件

- canonical M=1 target path自体が同一入力でdeterministicでない場合、MTP最適化を止め、最初の非決定op/providerを修正する。
- serial-equivalent verifyがbit-exactでない場合、許容差やtoken一致へ条件を緩めず、最初の不一致layer/opへ戻る。
- staged KVでaccepted prefixだけをpublishできない場合、encoding別rollbackを上位へ追加せずtransaction/layoutを再設計する。
- correctness完了後も両canonical targetで改善がnoise envelopeを越えない場合、MTP高速化完了とはせず、profile結果と次candidateを示して
  ユーザーとPhase 18の範囲を再計画する。
- 同じwork unitの2回reject、review時間が実装時間超、1時間以上の機能進捗停止、検証/docsが30%超、見積り1.5倍超、
  acceptance変更時は追加探索・検証を止め、同じwork unitを再計画する。

## Closeoutで必要な結論

- MTP on/offは何がどこまでbit-exactか。BF16と量子化targetの双方でIssue #25618型分岐がないか。
- draftを何token提案し、平均何token承認し、target logical work、dispatch、syncをどれだけ削減したか。
- 各canonical targetのMTP前後倍率とnoise、auto-select判断。比較可能な他engine倍率がある場合は条件差。
- speedupがtarget batch arithmeticの変更ではなく、serial-equivalent row executionとorchestration削減で得られたこと。

## Closeout結論（2026-08-16）

- M=2/3/4/7/8のtarget token/hiddenはR9700/V620、BF16+FP16 KVとFP8 W8A8+static FP8 KVで逐次M=1とbit-exactだった。
  M=8では全raw BF16 logitsとaccepted-prefix K/V semantic payloadもbyte-exactだったため、Issue #25618型の量子化target分岐は
  実行matrixで観測されなかった。
- 数値target blockはdraft 1..7相当を検証し、generation transactionは一回のrecurrent-state rewindで安全なwidth 1/2に限定して
  最初の不一致より後をcommitしない。通常採用するR9700 width 1のfixed 32-token caseはproposal 16、
  accepted 15、target rows/output 0.96875、target dispatch 15,744→8,856、追加MTP dispatch 1,475だった。
- R9700の最終paired測定は中央値`1.0355x`、MAD`0.0028`、p10/p90 `1.0242/1.0448`でnoiseを越えたため、fixed
  Qwen3.5-4B BF16 text-only greedyに内部auto-selectする。V620 width 1は中央値`0.9990x`でnoise内のためtarget-onlyを維持する。
- 高速化は各rowのM=1 arithmeticを維持したsmall-M providerとblock orchestrationによる。sampled requestはpublic RNG/logits順を変えない
  target-only内部選択で、CLI/APIのflag、opt-in、警告、起動コマンド差はない。

[対応する履歴](../../../../../history/2026/08/11-20/phase18-mtp-exact-sequential-speedup.md)

# Phase 13: モデル非依存prepared execution制御

> 状態: ready（Phase 12R完了後のlocal先行実行対象）
> 作成日: 2026-08-14

[Phase 12待機中のローカル先行実行キュー](phase12-wait-local-forward-queue.md)ではPhase 12RのCI portability repairに
続いて本Phaseを実行する。Phase 12を完了扱いにせず、MI300X VMを起動しない。

## 目的

Phase 9で大きな性能改善を得たprepared semantic cache、same-stream segment owner、completion集約、
transactional state publicationを`QwenExecutionCore`固有の実装から抽出し、別model adapterが同じ高速な
実行骨格を再実装せず利用できるようにする。

Phase 13は新しいmodel architectureやkernelを追加するphaseではない。Qwen3.5固有のgraph lowering、tensor名、
attention preprocess、GDN、KV/linear state descriptorはmodel adapter側へ残し、model-neutral層は実行順序、
prepared operation、binding更新、submission owner、completion、audit、transaction境界だけを所有する。

本Phaseを旧Phase 13のGemma 4対応より前へ挿入する。旧Phase 13〜19はPhase 14〜20へ一段繰り下げる。

## 開発順序の変更

| 新Phase | 内容 | 旧Phase |
| --- | --- | --- |
| 13 | モデル非依存prepared execution制御 | 新規 |
| 14 | google/gemma-4-12B | 13 |
| 15 | Weight NVFP4 | 14 |
| 16 | KV cache FP8/NVFP4 | 15 |
| 17 | MTP、vision | 16 |
| 18 | Gemma4またはQwen3.5 MoE | 17 |
| 19 | 残りの初期version機能 | 18 |
| 20 | 人間によるREADME整備・発表 | 19 |

Phase 10〜12の内容と順序は変更しない。

## 開始時点の事実

- backend registry、semantic op registry、HIP kernel registry、versioned C ABI、opaque buffer/KV ownerは既に
  model-neutralな境界を持つ。
- public HIP completionのadaptive pollingは全model/opへ共通適用される。
- BF16 Matmul、causal attention、KV append等のnative providerはsemantic descriptorとshapeから選択され、
  Qwen graph symbolを参照しない。
- 一方、`PendingSegmentSubmission`、`flush_segment()`、`prepared_semantics` cache、terminal Argmax/KV appendでの
  completion集約は`qwen_execution.rs`内にある。新しいmodel executorを独立実装すると、この制御を複製するか、
  Phase 9以前のper-op waitへ戻る危険がある。
- Qwen attention preprocessはQ-gate packing、Q/K norm、M-RoPE、head dim 256を含み、GDNは固定head/layoutと
  transactional recurrent stateを持つ。これらを汎用演算に見せかけずmodel固有adapter/providerとして維持する。
- Qwen3.5 2B/4B/9Bは同じexecution pathを利用するが、異なるmodel architectureによるproduction実績はまだない。

## 責務境界

### model-neutral層へ移すもの

- immutableなprepared operation列と、request-local binding/dynamic parameterを分離したexecution plan。
- semantic、causal attention、linear attention等のsubmission ownerをterminal boundaryまで保持するsegment。
- same-stream順序を利用したblocking waitの集約と、boundary後の先行completion query。
- readback、KV publication、mutable model state publication、cancellation/errorを明示するboundary kind。
- success時だけstate generationをpublishし、failure/timeout/drop時はrequestをpoisonして未完了stateを公開しない
  transaction lifecycle。
- backend/target、submission/kernel count、fallback、segment/boundary countを集約する共通audit。
- descriptor、static layout、binding identity、動的fieldの区分を持つprepared cache keyと失効規則。

### model adapterへ残すもの

- model lock/configの検証、weight名、graph topology、layer type、tensor shapeの決定。
- Qwen3.5 attention preprocess、GDN、RoPE/M-RoPE、固有normalization semantics。
- KV appendまたはrecurrent state更新のどこがmodel上のpublication boundaryになるかの宣言。
- logits tensor、samplingへ渡すterminal output、model固有state descriptorの生成。
- Gemma 4、DeepSeek、MiniMax等に固有のattention、MoE routing、MTP、vision処理。

model-neutral層はmodel名、Qwen定数、weight tensor名、固定head数、固定vocabulary sizeをimportしない。
model adapterはHIP kernel symbol、raw stream/event、VMM pointer arithmeticを所有しない。

## スコープ

- `sllm-core`内のmodel-neutral prepared execution plan、transition、segment、boundary、audit契約。
- Qwen3.5 executorを薄いmodel adapterへ移行し、既存のprefill/decode/generation serviceへ接続する。
- model-neutral synthetic fixtureによる、Qwen symbolを使わないexecution controlのhost contract確認。
- 既存semantic op、KV/linear state、cancellation、readback、OpenAI serviceとの互換性確認。
- Phase 9のshort-odd profileと構造auditを使った、wait/submission/segment構造の回帰確認。
- runtime architecture、model onboarding手順、Phase 14 Gemma 4へのhandoff更新。

次はPhase 13に含めない。

- Gemma 4本体、tokenizer、weight mapping、RoPE/attention kernelの実装。
- 新しいGPU kernel、FP8/NVFP4 provider、Paged Attention、continuous batching。
- 汎用graph optimizer、JIT compiler、永続autotuning DB、multi-stream scheduler。
- public C ABIの不要な抽象化変更。既存native prepared opで表現できない場合だけ別途設計判断する。
- Qwen GDNやattention preprocessを、意味が異なるmodelにも使える汎用opとして拡張すること。

## 受入条件

1. model-neutral execution moduleがQwen module、Qwen定数、model tensor名を参照せず、prepared plan、transition、
   segment、boundary、auditを表現する。
2. Qwen executorはmodel graph/stateのadapterに縮小され、独自のprepared cache、pending submission enum、
   segment flush loop、completion wait policyを持たない。
3. cache keyと失効規則がdescriptor/static layout/binding/dynamic fieldを区別し、position、token/KV length、pointer、
   state generationの変更で古いprepared operationを誤再利用しない。
4. terminal/readback、KV publication、linear/recurrent state publication、cancel/error boundaryが明示され、
   success前のstate公開、completion前のowner解放、partial failure後のrequest再利用を許さない。
5. model-neutral fixtureが少なくともsemantic-only列、stateful boundary、terminal readback、forced failure、drop/cancelを
   Qwen symbolなしで確認する。非整列値と境界前後を含める。
6. Qwen3.5の既存graph意味、dispatch provider、submission/kernel audit、fallbackなし、output、state length、cleanupを
   移行前と同じ契約で維持する。
7. focused host testと最小modelのreal-GPU smokeを行い、Phase 9 short-oddでTTFT/TPOT/E2Eとsegment/boundary countを
   観測する。未承認の一律性能倍率はhard gateにせず、明確な退行があれば原因と採用判断を記録する。
8. CLIとOpenAI non-stream/SSEが同じshared execution pathを使い、stop/disconnect cancellation後もrequest/state/
   workspace/process cleanupを維持する。
9. Phase 14のmodel adapter実装手順が、共通層へ渡すgraph/state/boundary宣言と、model側へ残す責務を具体的に示す。
10. affected checks、1回のintegration review、plan/history/main-plan/runtime文書を同期し、完了時に本planをarchiveする。

## 実装順序

### P13-A0: 現行責務と回帰baselineの固定

- `QwenExecutionCore`のprepare、cache、submit、wait/query、state publication、readback、audit、cleanupを分類する。
- model-neutralへ移す処理とQwen adapterへ残す処理をcall graphと型依存で固定する。
- Phase 9のQwen3.5-2B focused correctnessと4B short-oddから、output、dispatch、segment boundary、性能観測値を
  移行baselineとして選ぶ。long/full matrixを通常iterationへ持ち込まない。
- forced error、timeout、drop、disconnectの既存transaction contractをhost fixtureとして先に固定する。

### P13-A1: model-neutral execution IRとadapter契約

- immutable prepared planとrequest-local transitionを分離する最小型を`sllm-core`の共通execution層へ追加する。
- operation submit、stateful submit、publication boundary、terminal readbackを明示的なnode/boundaryとして表す。
- model adapterはmodel graphから共通node列とbinding/state宣言を生成し、共通executorはmodel固有enumをmatchしない。
- dynamic parameterとcacheabilityをdescriptorで宣言し、labelやtoken countだけに依存する暗黙cache keyを廃止する。

### P13-A2: prepared cache、segment owner、transaction lifecycle

- prepared operationのownerと失効をplan/request lifetimeへ対応付ける。
- heterogeneous submissionを安全に保持できる共通ownerを導入し、同一ordered queue上のterminal eventでsegmentを閉じる。
- boundary event成功後に先行completionをqueryし、dispatch auditを収集する。pending/failureはfail-closedに扱う。
- state publication callbackは成功済みgenerationだけを公開し、error/drop/cancelではrollback/poison規則を適用する。

### P13-A3: Qwen3.5 adapter移行

- Qwen graph loweringを共通plan/transition入力へ変換し、既存semantic/KV/linear attention providerをそのまま使う。
- Qwen固有attention preprocess、GDN state、argmax/logits selectionはadapter側へ残す。
- `PendingSegmentSubmission`、`flush_segment()`、Qwen内prepared cacheとwait policyを削除し、共通executorへ一本化する。
- 2B/4B/9Bで共有するmodel-resident ownerとrequest-local stateの分離を維持する。

### P13-A4: model-neutral fixtureと失敗経路

- Qwen graph builderや定数をimportしない小さなsynthetic adapterをtest専用に作る。
- semantic-only、複数segment、KV相当のpublication boundary、terminal readbackを構成し、順序とowner lifetimeを検査する。
- prepare/submit/query failure、timeout、drop、cancel、stale dynamic binding、cache invalidationを注入し、未完了state非公開と
  resource cleanupを確認する。
- 3、17、255/256/257等の非整列・境界値を含め、単一の便利なshapeだけで抽象化を固定しない。

### P13-A5: focused GPU統合と性能回帰確認

- canonical RDNA2/RDNA4では最小Qwen modelのprefill/decode、state publication、fallbackなし、cleanupをfocusedに確認する。
- 4B short-oddでPhase 9と同じdirect engine指標を取得し、per-op blocking waitが再導入されていないことをauditで確認する。
- Phase 11/12でCDNA3 production pathが成立済みなら同じ共通executorのfocused smokeを再利用し、未配置GPUを理由に
  Phase 13のhost設計作業を増やさない。
- OpenAI non-stream/SSEとdisconnectを一回ずつ確認し、service側にmodel別token loopを増やさない。

### P13-A6: Phase 14 handoffと完了同期

- runtime文書へprepared plan、transition、segment、boundary、adapterの所有関係と失敗時遷移を反映する。
- Phase 14 Gemma 4のonboarding checklistへ、model adapter、固有op/provider、共通execution接続、focused evidenceを記載する。
- Qwen固有依存が共通moduleへ逆流していないこと、Qwen側にwait/cache実装が重複していないことをreviewする。
- affected final checks後、結果をhistoryへ記録し、本planをarchiveへ移す。

## 計測と検証lane

| lane | Phase 13での使用 |
| --- | --- |
| host-contract | model-neutral fixture、cache invalidation、boundary順序、failure/drop/cancel、audit |
| focused GPU | 最小Qwen model、短いprefill/decode、state publication、fallback、cleanup |
| performance spot | 4B short-odd、Phase 9と同じdirect指標、segment/boundary audit |
| service smoke | non-stream、SSE、disconnectをshared generation pathで各1回 |
| broad/nightly | model/kernel/dtype意味が変わらない限り通常iterationでは実行しない |

## Rollbackと再計画

- Qwen adapter移行中も現行semantic providerとtransaction contractを削除せず、共通executorへの切替単位で戻せるようにする。
- state早期公開、ownerのuse-after-free、stale prepared binding再利用、pending completionの成功扱い、silent fallbackは
  correctness blockerとする。
- 単に型を共通moduleへ移しただけでQwen固有定数やmatch分岐が残る場合はPhase 13の目的達成としない。
- 抽象化のためにmodelごとの余分なhost dispatchやper-op waitが必要になる場合は、共通IRを広げ続けず境界を再設計する。
- 同じwork unitの2回reject、review時間が実装時間超過、1時間以上の機能進捗停止、検証・文書が30%超、
  見積り1.5倍超、gate/受入条件変更のいずれかで追加review・検証を止め、ユーザーへ報告して再計画する。

[対応する履歴](../../../../../history/2026/08/11-20/phase13-model-neutral-execution-control.md)

# Phase 12待機中のローカル先行実行キュー

> 状態: ready
> 作成日: 2026-08-15

## 目的

MI300Xを管理できるまとまった時間が確保できるまでPhase 12を`ready`で保持し、GitHub-hosted CPU/compile環境、
V620 `gfx1030`、R9700 `gfx1201`で進められるPhase 12RとPhase 13以降を先行実行する。Gemma 4 Denseの完了を
`/goal`の終端にせず、共通性能再評価、Weight NVFP4、さらに後続Phaseの計画・実装準備まで、依存関係を
壊さず順に取り出せる十分に長いqueueを用意する。

これはPhase番号や承認済みの製品順序を変更しない。Phase 12を完了扱いせず、MI300X、CDNA3実機PASS、
Hot Aisle固有tupleをローカル実行から主張しない。

## 実行原則

1. 同時に実装するwork unitは一つとし、下記の上から順に進める。後段の調査や計画作成は、前段の実装へ
   変更を混ぜない範囲でだけ先行できる。
2. 各Phaseは個別のplan、受入条件、history、commit境界を持つ。前段が完了したらユーザー応答待ちで停止せず、
   次の`ready`なwork unitへ進む。
3. correctness/security blocker、実model/revisionの不在、必要hardware capabilityの不在は成功へ読み替えない。
   実装不能部分を記録した後も、独立して安全に進められるhost contract、oracle、plan作成があれば続行する。
4. 通常iterationはhost-contract、model slice、短いGPU smoke、O0/O1だけを使う。2B/4B/9B/12Bの全model、
   両GPU、全dtype、long serviceを最適化candidateごとに繰り返さない。
5. V620またはR9700のGPUを使用してよい。進行中のrunは監視し、timeout、crash、zero selection、CPU fallbackを
   GPU PASSにしない。終了時は子process、VRAM、temporary artifactを片付ける。
6. Hot Aisle VMを作成、起動、延長しない。MI300Xだけが必要なwork unitへ到達した場合はPhase 12を開始せず、
   local-onlyな次の準備へ移る。
7. model、量子化sidecar、raw trace、profile、binaryをGitへ追加しない。追跡するのはschema、runner、bounded summary、
   plan/history、source/provenance metadataだけとする。
8. 各Phaseの受入条件、検証、plan/history closeoutを完了した時点で、そのPhaseだけを必要最小限のcommitへ整理して
   current GitHub branchへpushする。次Phaseの変更を同じcommitへ混ぜず、force pushと共有履歴の書換えを行わない。

## 固定実行順序

### Q0: Phase 12R CI portability repair

> 状態: completed（2026-08-15）

正本は[Phase 12R archive](../../../../archive/2026/08/11-20/phase12r-ci-portability-repair.md)とする。

1. P12R-A0でcurrent GitHub Actions failure、link input、workflow triggerを固定する。
2. P12R-A1でH0のformat、llama reference portability、Rust dependency closureを修復する。
3. P12R-A2/A3でH3 link正本、重複workflow、GitHub-hosted/self-hosted event境界を整理する。
4. P12R-A4/A5でregistry-driven local entrypoint、H0〜H3、GitHub clean candidateを確認する。
5. P12R-A6でplan/history/testing/main planを同期してからQ1へ進む。

Phase 12RはPhase 12の完了やMI300X PASSを意味しない。runtime/model/kernelの機能変更や広いGPU再実行を混ぜない。

### Q1: Phase 13 モデル非依存prepared execution制御

正本は[Phase 13 archive](../../../../archive/2026/08/11-20/phase13-model-neutral-execution-control.md)とする。

1. P13-A0で現行Qwen責務と短い回帰baselineを固定する。
2. P13-A1/A2でmodel-neutral plan、transition、segment、boundary、cache、transactionを実装する。
3. P13-A3/A4でQwen adapter移行とQwen symbolを持たないfixtureを完成する。
4. P13-A5/A6でV620/R9700 focused smoke、short-odd、service、Gemma handoffを確認する。
5. Phase 13をarchiveし、history/main plan/runtimeを同期してからQ2へ進む。

Phase 13が未完了のままGemma executor、NVFP4 graph、MoE executorを独立実装しない。

### Q2: Phase 14 Gemma 4 12B Dense text-only

正本は[Phase 14 archive](../../../../archive/2026/08/11-20/phase14-gemma4-dense.md)とする。

1. P14-A0/A1で公式source、immutable model lock、architecture差分、adapter契約を固定する。
2. P14-A2/A3でweight mapping、graph lowering、必要な固有semantic op/providerを実装する。
3. P14-A4でPhase 13の共通executorへ接続し、Gemma側に独自wait/cache loopを作らない。
4. P14-A5/A6でmodel slice、R9700 full-model、V620 bounded evidence、CLI/OpenAI serviceを確認する。
5. P14-A7で性能summaryと共通最適化候補を渡し、Phase 14を完了してからQ3へ進む。

Gemma 4 Denseの完了はqueueのcheckpointであり、`/goal`の完了条件ではない。

### Q3: Qwen/Gemma共通のRDNA2・RDNA4性能bridge

> 状態: completed（2026-08-15）

[詳細plan](../../../../archive/2026/08/11-20/cross-model-rdna-performance-bridge.md)を正とする。

これは新しいPhase番号を挿入せず、Phase 14完了後かつPhase 15開始前のprofile-driven bridgeとする。

#### Q2-A0: fresh profile

- Qwen3.5最小modelとGemma 4の代表pathで、R9700をprimary、V620をbounded secondaryとしてO1を取得する。
- wall timeをhost/launch、M=1 matvec、MLP、attention、model固有op、transferへ分ける。
- Phase 9と同じ指標、fixed llama.cpp条件を再利用できるcaseだけ比較し、異なるmodel/tokenizerの値を同等比較と呼ばない。

#### Q2-A1: candidate選定

次の順で、両modelまたは両GPUへ適用できる上位candidateを最大二つ選ぶ。

1. model/dtype非依存のprepared command-list、graph replay、launch/completion削減。
2. model共通のM=1 matvec、gate/up projection、MLP、RMSNorm/residual fusion。
3. GPU architecture内で共通するlayout、wave、prefill provider tuning。
4. model固有またはGPU固有tuning。

full attentionが代表wall timeの支配要因へ移らない限り、RDNA4 FA3-likeを選ばない。

#### Q2-A2: bounded実装と判定

- candidateごとにmicro/oracle、O1、対象外caseの短い回帰だけを実行する。
- repeated medianで対象caseが改善し、別modelまたは別canonical caseへ明確な退行を起こさない場合だけ採用する。
- 二つ実装した、または明確な共通candidateがなくなった時点でbridgeを閉じる。全candidateの総当たりをしない。
- 採否と残差をPhase 14 history、Phase 15開始時baseline、main planへ記録してQ4へ進む。

### Q4: Phase 15 Weight NVFP4

> 状態: complete（2026-08-15）

正本は[Phase 15 archive](../../../../archive/2026/08/11-20/phase15-weight-nvfp4.md)とする。

1. P15-A0/A1でNVFP4 value、block scale、tensor scale、derived artifact、encoding contractを固定する。
2. P15-A2/A3でconverter/oracle、sidecar、loader、provider選択を実装する。
3. P15-A4でRDNA4 production candidateとRDNA2 explicit emulation/conversionを分ける。
4. P15-A5/A6でQwen/Gemma slice、full model、精度、VRAM、性能を確認する。
5. P15-A7でservice、互換性、provenance、historyを同期し、Phase 15を完了してQ5へ進む。

2026-08-15にQ4を完了した。Qwen full-modelは両exact RDNA targetでtop-1 3/3一致したがKLD budgetを超え、
Gemma sliceはtop-1 2/3だったため、NVFP4 packed-dequantは両targetともcorrectness-only opt-inとした。
当初のQ5継続条件は、同日のユーザー明示指示「Phase 15まで終わらせたらgoalを完了扱い」により本goalには適用しない。

### Q5: 枯渇防止tail queue

Q0〜Q4がすべて完了しても`/goal`を終了せず、現在のmain plan順序を保って次を行う。各Phaseでは最初に
同じdirectoryへ詳細planとhistory stubを作り、受入条件を固定してから実装へ入る。

1. **Phase 16 KV cache FP8/NVFP4**
   - FP8 KVを先に、NVFP4 KVを次に分け、KV layout/scale、append、attention consumption、accuracy、capacity、
     cancellation、VRAMを定義する。
   - vAttention/contiguous-residentのopaque契約を維持し、上位serviceへphysical layoutを漏らさない。
   - MI300X実機が必要なcaseは保留し、RDNA2/RDNA4とhost oracleで進められる範囲を実装する。
2. **Phase 17 MTP、vision**
   - MTP text-onlyを先に独立plan化し、draft/verify/accept stateとgeneration service統合を実装する。
   - visionはprocessor、image tensor、multimodal prompt、encoder/projector、cacheを別work unitにし、MTPと同時に
     debuggingしない。
3. **Phase 18 Gemma4またはQwen3.5 MoE**
   - 公式weight/revision、model size、local GPU収容性、router semantics、expert layoutを比較して一方を選ぶ。
   - router oracle、top-k/normalization、expert residency、grouped dispatch、shared expert、fallback、accuracyを先に計画する。
   - Phase 13の共通executorへMoE node/boundaryを追加し、Qwen/Gemma固有のroutingはadapter側へ残す。

Phase 16以降の詳細planを作成しただけで対応完了とはしない。実装、数値oracle、focused GPU、serviceまたは
generation統合、cleanup、history同期まで満たした時だけ次へ進む。このtailは今回の不在時間内に完遂することを
想定せず、早い進捗でqueueが空になることを防ぐためのものでもある。

## Phase 12再開時の扱い

- 帰宅後、Hot Aisle利用前にその時点の最新mainからexact `gfx942` candidateを再buildし、dry-run、artifact metadata、
  source/build identityを更新する。Phase 11時点の古いbinaryを最新mainの証拠として使わない。
- Phase 12の実行matrixはQwen3.5 4B/9B BF16/FP8、contiguous-resident KV、service、llama.cpp比較のまま固定する。
  先行実装したGemma、NVFP4、KV量子化、MoEをMI300X matrixへ自動追加しない。
- 先行変更がPhase 12対象pathへ影響する場合だけlatest candidateで確認し、別modelの追加を理由に全matrixを拡大しない。
- Phase 12を開始した時点で本queueを破棄せず、実行中work unitの状態を記録してからMI300X作業へ切り替える。

## `/goal`へ渡す完了条件

このqueue全体を短期goalの達成条件にしない。goalは上から継続し、ユーザーが戻るまで、または実行可能な
work unitが本当に尽きるまで進める。Gemma 4、共通最適化、NVFP4のいずれか一つの完了だけをgoal completeと
扱わない。Phase 18まで実装・検証・同期が完了した場合に限りqueue全体の完了を検討する。

[対応する履歴](../../../../../history/2026/08/11-20/phase12-wait-local-forward-queue.md)

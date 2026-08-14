# Phase 13 モデル非依存prepared execution制御履歴

## 2026-08-15: P13-A1〜A4 共通実行制御とQwen adapter移行

- `sllm-core`へmodel名をimportしない`prepared_execution` moduleを追加した。immutable node順序は
  `PreparedExecutionPlan<N>`、request-local値は`PreparedTransition`、boundaryは`ExecutionBoundaryKind`で表す。
- semantic prepared cacheのkeyを旧`(label, token_count)`から、`SemanticOpDescriptor`、input/output buffer ID、
  `TensorView`、access mode、token/position/期待長、binding/state generationのexact identityへ変更した。
  attention preprocess等は`Transient`、交換可能なsemantic opだけを`Reusable`とする。
- semantic、causal attention、linear attention submissionを異種ownerのままterminal boundaryまで保持する共通segment、
  terminal wait/query、exact backend/target・submission/kernel・fallback・segment/boundaryの共通auditを追加した。
- request transactionをsingle in-flightとし、成功時だけcommitする共通guardへ移した。submit/query failure、pending、
  cancel、guard dropはstate/outputを公開せずrequestをpoisonし、partial mutation後の再利用を拒否する。
- Qwen graphは`PreparedExecutionPlan<QwenGraphNode>`へlowerし、Qwen側から独自`PendingSegmentSubmission`、
  `DispatchAuditAccumulator`、`RequestLifecycle`/`TransitionGuard`、label/token prepared cache、segment flush loopを削除した。
  Qwen側にはgraph/state、attention preprocess、GDN/KV descriptor、Argmax/logits解釈とboundary宣言だけを残した。
- Qwen symbolを使わないsynthetic fixtureでsemantic列、stateful/terminal boundary、異種owner lifetime、forced failure、
  pending/query failure、drop/cancel、cache失効を確認した。値は`3/17/255/256/257`と境界前後を含む。
- host結果は`sllm-core` 106/106、`sllm-cli` 23/23、engine/G3/schema関連 89/89 PASSで、workspace checkと
  focused clippy `-D warnings`もPASSした。

## 2026-08-15: P13-A5 focused RDNA統合、性能spot、service smoke

- Qwen3.5-2B short-oddをcanonical V620 `gfx1030`とR9700 `gfx1201`で2/2 PASSした。両rowともexact HIP、
  fallbackなし、request/session cleanup 0、終了後GPU process 0である。
- Qwen3.5-4B short-oddを同じ両targetで2/2 PASSした。13 performance sample合計のauditは各targetで
  submission/kernel `103,428/108,732`、segment/boundary `1,989/1,989`であり、1 request当たりではPhase 9と同じ
  `7,956/8,364`、153/153となった。per-op waitやdispatch増加はない。

| target | TTFT median | E2E median | prefill | decode | Phase 9 TTFT / E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| V620 `gfx1030` | 0.301 s | 0.828 s | 57.50 tok/s | 30.81 tok/s | 0.306 / 0.855 s |
| R9700 `gfx1201` | 0.051 s | 0.480 s | 379.24 tok/s | 37.92 tok/s | 0.051 / 0.490 s |

- 実測開始時にCLIが既に出力していた`weight_encoding`/`fp8_provider`とperformance schemaのずれを検出した。
  これらをclosed schemaへ追加し、Phase 13の`segment_count`/`boundary_count`もCLI、direct/render、G3正規化証跡へ
  一貫して追加した。
- R9700初回のmonitorでROCm libraryが正当に遅延追加された際、runner内部は複数loader setを検証する一方、最終manifest
  validatorがfirst/last sampleへ最終setとの同一性を要求していた。全検証済みloader recordをmanifestへ保持し、各sampleを
  対応digestへ結合するよう修正した。関連engine tests 73/73と初回条件を含む再実測がPASSした。
- R9700 production serverでraw non-stream/SSE、OpenAI Python client 2.44.0、disconnect後recoveryをPASSした。
  disconnectはHIP submission/kernel `936/984`後にcancelされ、request state/workspace cleanupはいずれも0、shutdown時
  final current bytes 0、GPU process countは実行前後0であった。CLIとserverはいずれも同じQwen execution requestと
  `GenerationServiceV1`を利用し、service固有token loopは追加していない。

## 2026-08-15: P13-A6 handoff、integration review、closeout

- runtime文書へprepared plan、transition、exact cache key、segment/boundary、transaction/auditの所有関係と失敗遷移を
  反映した。Phase 14 Gemma 4 planへ具体的な共通API接続チェックリストを追加した。
- integration reviewは一回に集約し、Qwen固有symbolの共通module逆流、旧cache/wait/owner型の残存、state公開順、
  cancel/drop poison、audit overflow/fallback、loader provenanceの独立再検証を確認した。loader recordの型検証を一箇所
  強化し、再確認後にcorrectness/security blocker、release evidence欠落は残っていない。
- final affected checksはworkspace全target test、workspace clippy `-D warnings`、Rust format、engine/schema/G3 contract、
  JSON/schema/manifest validator、Markdown/link consistencyを対象とした。GPU証拠は上記P13-A5のexact target draft integration
  runを正とし、CPU/compile-onlyをGPU PASSへ読み替えない。
- host registry finalはH0 `513/513`、H1 `385/385`、H2 `36/36` PASSした。最初のH0が検出した`op.rs`のRMSNorm H3
  source-set hash driftを更新し、focused manifest validator PASS後にH0全体を再実行した結果である。
- 受入条件1〜10を満たしたためPhase 13を完了し、planをarchiveした。次のlocal forward workはPhase 14 Gemma 4である。

## 2026-08-15: P13-A0 現行責務と回帰baselineの固定

- 受入条件1〜10を実装前の固定条件とし、新しいkernel、model architecture、public C ABI変更、広範GPU matrixを
  Phase 13へ追加しない。通常iterationはhost contract、最小model focused smoke、4B short-oddに限定する。
- 現行call graphは`run_transition -> lower_graph -> execute_* -> submit_semantic/typed state submit ->
  flush_segment -> readback/state validation -> output publication`である。分離境界を次のように固定した。

| 現行責務 | model-neutral層へ移すもの | Qwen adapterへ残すもの |
| --- | --- | --- |
| `prepared_semantics` / `submit_semantic` | descriptor、static layout、binding identity、dynamic fieldを含むcache keyとprepare/submit | Qwen graph nodeからsemantic descriptorとbindingを生成する処理 |
| `PendingSegmentSubmission` / `flush_segment` | heterogeneous completion owner、same-queue terminal boundary、先行query、owner lifetime | KV append、terminal Argmax等をどのboundary kindにするかの宣言 |
| `TransitionGuard` / `RequestLifecycle` | begin、single in-flight、commit、failure/drop/cancel時poison | token、position、capacity、Qwen state lengthのadmission検証 |
| `DispatchAuditAccumulator` | backend/target、submission/kernel、fallback、segment/boundaryの集約 | Qwen公開audit型へのadapterとmodel固有memory audit |
| Argmax/logits/state publication | terminal readbackとsuccess後publicationの順序 | Qwen vocabulary、Argmax decode、KV/GDN descriptor、公開output |
- 現行cache key `(label, token_count)`はbinding pointer/owner、descriptor、position、state generationを識別しないため、
  Phase 13では再利用しない。attention preprocessを単にcache対象外にするだけでなく、共通keyがdynamic fieldを明示し、
  transient policyとexact-identity reuseを区別する。
- host baseline `cargo +1.97.1 test -p sllm-core qwen_execution -- --nocapture`は14/14 PASSした。これにはgraph順序、
  prepared再利用、terminal readback、state mutation後のforced failure、pending completion、guard drop、poison後の再利用拒否、
  dispatch audit、resident/request cleanupが含まれる。focused clippyもPASSした。
- performance baselineはPhase 9正本を再利用する。4B short-oddはV620 `gfx1030`でTTFT/E2E `0.306/0.855 s`、
  prefill/decode `56.91/29.69 tok/s`、R9700 `gfx1201`で`0.051/0.490 s`、`377.46/37.20 tok/s`である。
  submission/kernelは両targetとも`7,956/8,364`、fallbackなし、request/session cleanup 0を維持対象とする。

## 2026-08-15: Phase 12R後のqueue対象へ変更

- ユーザー指示によるCI portability repairをPhase 12Rとしてlocal forward queueの先頭へ追加した。
- 本Phaseの番号、目的、受入条件は変更せず、Phase 12R完了後の最初の製品機能Phaseとして維持した。

## 2026-08-15: Phase 12待機中の先行実行対象へ変更

- MI300Xを管理できる時間が確保できるまでPhase 12を`ready`で保持するユーザー指示に基づき、本Phaseをlocal queueの
  最初の`ready` work unitとした。
- Phase 12の完了やskipは主張せず、V620/R9700とhostだけで既存計画P13-A0〜A6を進める。
- 完了後はGemma 4 Denseで停止せず、Phase 14、cross-model RDNA性能bridge、Phase 15へ続くqueueへ接続した。

## 2026-08-14: 計画作成とPhase繰り下げ

- ユーザー指示により、Phase 9で`QwenExecutionCore`へ実装した共通化可能な実行制御を抽出する作業を
  新しいPhase 13として、MI300X実機確認とGemma 4対応の間へ挿入した。
- 旧Phase 13〜19をPhase 14〜20へ一段繰り下げた。Phase 10のFP8 W8A8、Phase 11のCDNA3移植、
  Phase 12のMI300X実機確認は変更していない。
- model-neutral層の責務をprepared plan、request transition、segment owner、completion集約、boundary、
  transactional publication、audit、cache invalidationに限定した。
- Qwen3.5固有graph、attention preprocess、GDN、tensor名、state descriptorはadapter側に残す。
  Gemma 4本体の対応は繰り下げ後のPhase 14であり、Phase 13には含めない。
- Qwen固有symbolを参照しないsynthetic adapterで共通制御を証明し、既存Qwen pathを最初のproduction adapterとして
  移行する順序を採用した。
- 通常の検証はhost contract、最小Qwen modelのfocused GPU、4B short-odd performance spot、短いservice smokeに
  限定し、model/kernel/dtype意味が変わらない広範matrixを各iterationへ追加しない。

[対応する計画](../../../../plans/archive/2026/08/11-20/phase13-model-neutral-execution-control.md)

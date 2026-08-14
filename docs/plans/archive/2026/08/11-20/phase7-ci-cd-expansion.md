# Phase 7: CI/CDの定期・互換性・性能・release拡張（完了）

## 目的

Phase 1〜6で個別に成立したhost、compile-only、GPU、full-model、性能のrunnerを、
`trusted-solo-development`向けの定期実行と明示的なrelease実行へ束ねる。Phase 7は
新しいcorrectnessまたはperformance runnerを重複実装せず、実行profile、workflow、保持期間、
結果の意味を機械検査可能にする。

## スコープ

- daily、weekly、releaseの実行profileとtriggerをversioned manifestへ固定する。
- GitHub-hosted host/compile jobと、信頼済みself-hosted GPU jobを分離する。
- dailyは代表1 tuple、weeklyは現在利用可能なcanonical 2 tuple、releaseは明示選択した
  release対象tupleを実行する。
- exact targetのcompatibility compileを`gfx1030`〜`gfx1036`、`gfx1200`、`gfx1201`、`gfx942`
  へ拡張し、compile-onlyを実機互換性へ読み替えない。
- G4 compatibility recordとP0/P1の観測結果をprofile結果へ結び付ける。
- daily/weekly artifactは30日、release evidenceは90日保持する。
- performanceは観測値であり、ユーザー承認済み閾値がない状態でhard gateにしない。

次はPhase 7本体に含めない。15:30 JSTまでにPhase 7が完了し時間が残った場合だけ、独立した
API作業単位として実施する。

- reasoning/thinking指定のAPI公開。
- `<think>`と最終回答の分離、およびSSE対応。
- strict profileを変更しないOpenWebUI向け`max_tokens`互換profile。

## 受入条件

1. `ci/matrix/phase7-ci-profiles-v1.json`がdaily、weekly、releaseのtrigger、runner境界、
   suite、tuple、性能lane、保持期間、blocking意味を一意に定義し、JSON SchemaをPASSする。
2. `ci/matrix/phase7-compatibility-v1.json`がcanonical V620 `gfx1030`とR9700 `gfx1201`の
   現行experimental tupleをexact UUID/BDF/toolchainへ固定し、実測済みtierだけを列挙する。
3. `.github/workflows/phase7-lifecycle.yml`がschedule、manual release、published releaseから
   profileを一意に解決し、public PRをself-hosted GPUへ接続せず、権限を`contents: read`へ限定する。
4. dailyはR9700の短いGPU correctness/performance観測、weeklyはcanonical 2 GPUのG4/P1、
   releaseは明示的な全対象行を選ぶ。GIMPS等のforeign workloadを性能PASSへ混ぜない。
5. compatibility compileは10 exact targetを独立rowで扱い、artifact target、ROCm root、
   compile-only claim、結果件数をfail-closedに検査する。
6. workflow actionは完全commit SHAへ固定し、profileとworkflowのtrigger、row、retention、
   timeout、runner labelのdriftをH0 contract testが拒否する。
7. focused Phase 7 contract test、JSON/schema/workflow検査、matrix/path-to-suite登録、Markdown/link
   検査をPASSする。GPU実行を行った場合はexact target、数値oracle、fallbackなし、health、cleanupを
   実測結果へ記録する。
8. plan完了時にmatching history、main plan、CI/testing正本文書を同期し、このplanをarchiveへ移す。

## 実装順序

### P7-A: profileとcompatibility契約

- profile schema、profile matrix、canonical compatibility recordsを追加する。
- trigger、runner、retention、target集合とperformance非blocking意味をvalidatorで固定する。

### P7-B: lifecycle workflowとcontroller

- GitHub-hosted contract/host/compatibility compileとtrusted self-hosted GPU jobを実装する。
- 既存Phase 5 performance、G3/A6 service、G0〜G2 runnerのうちprofileに必要なものだけを呼ぶ。
- local dry-runは選択結果だけを出力し、GPU PASSを偽装しない。

### P7-C: verificationと完了同期

- focused host contractとworkflow drift testを実行する。
- 利用可能な実GPUでdaily profileを一度実行し、scheduled pathの実行可能性を確認する。
- plan/history/main plan/CI正本文書を同期してPhase 7を完了する。

## 非blocking follow-up

- performance hard thresholdは複数のO2/O3履歴と分散が揃った後にユーザー承認を得て設定する。
- GA kernel 6.8 tuple、別driver/ROCm、CDNA3実機は未検証のためPhase 7でsupportedへ昇格しない。
- public external contribution用のephemeral GPU runnerはinactive laneでありPhase 7をblockしない。

## 完了結果

- [x] daily/weekly/release profileとcanonical compatibility recordをschema付きで固定した。
- [x] trusted GPU境界、pinned actions、retention、fail-closed aggregateをworkflowへ実装した。
- [x] 10 exact targetのcompile-only draftを実行し、target metadataを含む10/10 PASSを確認した。
- [x] R9700 `gfx1201`のdaily GPU observationを1/1 PASSし、性能非blocking意味を維持した。
- [x] H0登録、focused unit、matrix、JSON/schema/workflowと正本文書を同期した。

[対応するhistory](../../../../../history/2026/08/11-20/phase7-ci-cd-expansion.md)

## 完了後の運用更新

2026-08-14にGIMPS終了とV620利用再開が明示されたため、daily profile revision 2はcanonical
V620 `gfx1030`とR9700 `gfx1201`の両方を選択する。上記の代表1 tuple受入条件とR9700 1/1完了結果は
Phase 7完了時点の履歴であり、現在の運用契約はprofile revision 2と対応historyを正とする。

# Phase XA: CI修正・公開・push後確認

> 状態: completed（2026-08-25）
> 対象: `main`のhost-required CI、public-runtime H3 compile-only、Phase 52 closeoutを含む未公開ローカルcommit

## 目的

Phase 52完了後のローカル`main`を公開する前に、最新のGitHub Actionsで確認されたhost-requiredと
public-runtime H3 compile-onlyの失敗を修正する。関連するhost契約、format、test isolation、compile-only契約を
fail-closedのまま検証し、Phase 52の2つの未公開commitとPhase XA修正をGitHubへpushする。push後は対象workflowの
最終結果を確認し、問題が発生していないことまでをPhase XAの完了条件とする。

GPU実行、モデル全体推論、性能測定はこのPhaseの対象外である。H3はcompile-only証拠であり、GPU correctness、
numerics、performance、runtime supportのPASSへ読み替えない。

## 固定入力と既知の失敗

- 開始時の公開`origin/main`は`159bc526cb26d180161f2cd7abcc22abb7e67e84`である。
- ローカル`main`はPhase 52実装`3ed002c476b49417cc702119e37c2389cefb96bc`とcloseout
  `d7e6821382b6bf5ec8fb94a80fd6f813e68eeac5`の2 commitだけ先行し、working treeはcleanである。
- GitHub Actions run `32681109190`ではH2がPASSした一方、H0はprocfs監視のINFRA_ERROR、MSRV timeout、
  C++ format違反と後続契約FAILを記録し、H1のworkspace testは120秒でtimeoutした。
- run `32681109285`ではpublic-runtime H3の`gfx1030` compile/link/extract stepが失敗し、`gfx1201`は未実行、
  aggregateとcleanupもFAILした。
- 現行ローカルHEADではRust workspace testとPhase 52関連の登録済み複合suiteはPASSするが、C++ format検査は
  11ファイル395箇所でFAILし、`ci.tests.test_phase52_r9700_summary`単独実行はimport順序依存でFAILする。

## 受入条件

1. C++ format検査がPASSし、format以外の意味変更を混ぜない。
2. Phase 52 summary testが単独実行と登録済み複合suiteの両方でPASSする。
3. host runnerのprocfs監視がLinuxの正当な`/proc/<pid>/stat`表現を受理し、異常値はfail-closedに維持する。
4. H0/H1のcommand budgetとworkflow budgetを既存のp95 10分、厳格上限15分内で整合させ、timeoutを無効化しない。
5. `h0`、`h1`、`h2`の現行matrixを実行し、required aggregateがPASSする。zero selection、timeout、crashはPASSにしない。
6. public-runtime H3の`gfx1030`と`gfx1201`を固定ROCm image、networkなし、production public runtimeの
   compile/link/extract/inspectでPASSし、2行aggregateを得る。
7. 計画、履歴、main plan、active roadmapのPhase 52状態を実結果へ同期する。
8. 変更と既存の未公開commitを目的別の必要最小限のcommitに整理し、forceなしで`origin/main`へpushする。
9. push後のrequired host workflowと今回修正したcompile-only workflowが完了し、未解決のfailure、cancel、timeoutを残さない。

## 作業範囲

- C++ source/testのclang-format修正。
- host suiteのprocess resource監視、command timeout、row/workflow budgetの整合修正と回帰test。
- Phase 52 summary testのimport順序依存解消。
- public-runtime H3のproduction compile/link/extract/inspect失敗原因と、失敗後aggregate/cleanupの診断保持修正。
- Phase XAの計画・履歴・main plan更新、commit整理、push、push後workflow監視。

次は対象外とする。

- GPU kernel、runtime dispatch、モデル、API、性能経路の意味変更。
- GPU実行またはCPU fallbackによるGPU PASS主張。
- Phase 51の再開、Phase 46〜48の実装、release/tag/PR作成、force push。

## 検証

- 変更箇所のunit/contract testとC++ public runtime host test。
- Rust workspace test、Python host/CI contract、format/static検査。
- `run_host_suite.py`によるH0/H1/H2とaggregate。
- 固定ROCm image内のpublic-runtime H3 2 targetとaggregate。
- push後のGitHub Actions結果と最終commit identity。

## closeout

`2c28cf0811f09b9e346c6f58250289912790a83b`までをforceなしで`origin/main`へpushし、host-required、通常H3、
public-runtime H3の全workflowがPASSした。host-requiredはH0/H1/H2とaggregate、通常H3とpublic-runtime H3は
`gfx1030`/`gfx1201`とaggregateを完了した。原因、採用修正、ローカル検証、公開commit、失敗した中間run、最終成功runは
matching historyへ記録した。GPU実行、numerics、performanceのPASSは主張しない。

[全体計画](../../../../main-plan.md) / [対応する履歴](../../../../../history/2026/08/21-31/phase-xa-ci-repair-and-push.md)

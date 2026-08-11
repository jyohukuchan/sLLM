# 再出発レビュー対応計画

## 目的

`restart-review-2026-08-02.md`の指摘を事実確認し、旧版の失敗を再発させないためのgovernance、workspace、CI、credential対策を機能実装前に確定・公開する。

## 判定

| 指摘 | 判定 | 対応 |
| --- | --- | --- |
| 旧履歴が現行mainから到達可能 | 事実。ただし変更しない | 旧履歴保持はユーザーが既に現状維持と決定。orphan化・force push・履歴書換えは行わず、到達可能性とlicense境界を明記する |
| workspace・Git肥大化対策不足 | 採用 | repository hygiene方針、`.gitignore`、H0/local guardrailを追加する |
| `B-1/B/B+1`性能崖gate不足 | 採用 | P0 triplet、dispatch/fallback証拠、G3の`255/256/257`を追加する |
| GPU変更をGPU未検証でmerge可能 | 採用 | reviewed immutable SHAに結び付くG0/G1/G2/P0をGPU影響変更のmerge条件にする |
| fail-closed集約の実装条件不足 | 採用 | `always()`、`needs`、report identity/freshness、missing/duplicate/unknown/cancelを規定する |
| governance baselineが未公開 | 部分採用 | remoteとlocal commitは同期済みだが、新規governance文書は未commit。機能code前に承認・commit・pushする |
| `sLLM.md`とmain planのCI順序差 | 対応済み | ユーザー承認後、git管理外文書の実装順序をmain planへ同期した |
| 平文credential運用 | 採用 | `passwords.txt`を使用禁止とし、secret manager・短命注入・sudo規則を定義する。file自体の変更とcredential rotationは所有者が行う |

## 実施内容

- [x] Git ancestry、remote同期、backup manifest、bundleをread-onlyで確認。
- [x] 旧workspace事故の既存summaryを確認。大規模backupの再帰scanは行わない。
- [x] `.gitignore`へroot-anchoredなbuild・raw artifact・生成fixture規則を追加。
- [x] `docs/development/repository-hygiene.md`を作成。
- [x] `docs/security/credentials.md`を作成。
- [x] CI・テスト計画へperformance cliff、GPU merge gate、集約条件を追加。
- [x] main planへ再発防止方針と実装順序を同期。
- [x] AGENTS.mdのcredential、repository hygiene、CI規則についてユーザー承認を得た。
- [x] `sLLM.md`のCI実装順序をmain planへ同期した。
- [x] AGENTS.mdへ作業単位ごとのtest、適用、適用後確認、`push` skillによる公開手順を追加した。
- [ ] `passwords.txt`内に有効なcredentialがある場合、所有者がrotationし、modeを最低`0600`へ変更。
- [x] governance baselineを機能codeより先にcommit `2764e73ebc45c8bbd209a426ca93ce341ed5d860`として`origin/main`へpush。

## 確認結果

- governance baseline本体のpush後、`main`と`origin/main`は`2764e73ebc45c8bbd209a426ca93ce341ed5d860`でahead/behind 0/0。
- 現行mainは2,378 commitで、空tree reset commit `f0eefbdd`は旧commit `e146237d`を親に持つ。
- pre-reset/post-reset bundleのhashと`git bundle verify`は成功し、現行repositoryの`git fsck --full`も成功した。
- backup全fileの独立再検査は一部permission denialのため未完了だが、既存manifestにはrsync差分0とbundle検証が記録されている。
- 旧版事故summaryは、103 local commit ahead、18 tracked変更、少なくとも711,758 untracked file、約93 GBのuntracked data、約155 GBのworkspaceを記録している。
- 現在はregistered worktree 1、stale registrationなし、mainのahead/behind 0/0。
- `passwords.txt`の内容は読んでいない。監査時のmodeは`0664`であり、credential fileの最低基準を満たさない。

## 完了条件

1. AGENTS.md変更がユーザー承認済み。
2. `sLLM.md`とのCI順序差が解消または明示的に現状維持と決定済み。
3. credentialの失効・rotationとlocal file移行が所有者により完了。
4. governance baselineがremoteへpush済み。
5. H0実装時にtracked tree guardとlocal hygiene commandが同時に追加される。
6. GPU影響変更のbranch gateが、同一reviewed SHAのGPU evidenceを要求する。

[対応する履歴](../../../../../history/2026/08/1-10/restart-review-remediation.md)

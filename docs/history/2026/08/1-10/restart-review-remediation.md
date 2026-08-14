# 再出発レビュー対応履歴

## 2026-08-02

- `restart-review-2026-08-02.md`の全指摘を4つの読み取り専用監査へ分割して検証した。
- 4つのCodex sessionは全て15分以内、終了コード0で完了した。
- 旧履歴の到達可能性、remote同期、backup bundle、旧workspace事故summaryを確認した。
- 旧履歴は既存のユーザー決定どおり保持し、履歴書換えを行わないとした。
- repository hygiene、credential、performance cliff、GPU merge gate、fail-closed集約を採用した。
- `.gitignore`、CI・テスト計画、main plan、AGENTS.mdを更新し、repository hygieneとcredential文書を追加した。
- `passwords.txt`の内容・権限は変更していない。
- ユーザー承認後、`sLLM.md`のCI実装順序をmain planへ同期した。
- AGENTS.mdへ、独立した作業単位ごとのtest、本番または本番相当環境への適用、適用後確認、`push` skillによる公開手順を追加した。
- push前reviewを受け、検証・適用・push対象のimmutable identity、受入evidence、適用先がないcode変更のpush禁止、適用後失敗時のrollback規則を追加した。
- 最終reviewで検出したbuild/ROCm/target/codegen変更のGPU gate不一致を、G0/G1/G2/G4/P0必須へ統一した。
- governance baseline本体をcommit `2764e73ebc45c8bbd209a426ca93ce341ed5d860`として`origin/main`へpushした。

## 2026-08-14: worktree個数gateの修正

- ユーザー確認により、当初AIが設定したregistered worktree 4個超の停止条件には個別の明示承認記録がなく、
  validでcleanな並行worktreeまでpushを止めるのは旧workspace肥大化の防止目的に対して過剰と判定した。
- worktree個数だけのhard gateを廃止し、9個以上をadvisory warningへ変更した。missing/prunable registrationと、
  clean・unlocked・非mainで14日超のworktreeもcleanup候補として警告するが、それだけでは非zero終了しない。
- 容量、未追跡data、remote同期等の実害を伴う既存停止条件と、所有者確認なしにworktreeを削除しない規則は維持した。

[対応する計画](../../../../plans/active/2026/08/1-10/restart-review-remediation.md)

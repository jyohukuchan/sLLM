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

[対応する計画](../../../../plans/active/2026/08/1-10/restart-review-remediation.md)

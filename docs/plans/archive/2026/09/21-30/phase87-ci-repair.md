# Phase 87 CI修復

2026-09-21の依頼。公開済みCIの状況と未公開のPhase 87変更を区別し、既存CI入口で問題を再現・修正する。

受入条件:

- Phase 87に伴うRust／nativeのCIエラーを修正し、機能・数値動作を維持する。
- 新規sourceのbuild入力登録とmanifestの参照・hashを同期する。
- 影響するhost／compile-onlyチェックを実行する。警告の一括無効化やチェックの削除で通さない。
- 既存の変更を保持し、commit／pushは行わない。GPU速度・品質の再測定は動作変更が必要な場合だけ判断する。

Rust、native形式・host検証、CI manifestの調査を分担する。証拠は`.local-artifacts/phase87/ci-repair/`へ置く。
2026-09-21完了。Rust警告、HIPのsource／symbol登録、schemaとhash参照を修正した。
H0 628件・H1 1,642件・H2 38件と、両targetの直接compile/link・ELF検査がPASS。
公開CIは未再実行。commit／pushは行っていない。

履歴: [修正内容と検証](../../../../../history/2026/09/21-30/phase87-ci-repair.md)。

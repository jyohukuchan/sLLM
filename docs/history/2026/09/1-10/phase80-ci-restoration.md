# Phase 80: CI修復の履歴

> 状態: 実装・検証中。最終公開CI成功前の記録。
> 開始日: 2026-09-07

## 開始時の状態

`50208e8e921975c2e5795fcb81315f698b1b42b2`の公開後CIでは、host H0とpublic-runtime H3が失敗し、
H1/H2と基本H3は成功していた。ユーザー指示でPhase80にCI修復を先行し、既存後続を81〜83へ移した。
各Phaseの完了手順へ、push後の最終HEADのCI監視・必要な修正・再検証・再pushを追加した。

public H3は公開関数8個の未登録でcompile前に停止していた。Rust dependency closureはpackage191個が一致していたが、
edgeが固定453に対し454で失敗した。H0のsllm-tools testはRSS2,164,662,272 Bが2GiB制限を超え停止した。
これはCI内のプロセス監視による停止であり、host全体のOOMを証明しない。
Clippy、C++整形、schemaとsemantic契約の失敗も確認した。

## 修正・検証の経過

- `prepared_execution.rs`のゼロextent判定を同値な`contains(&0)`へ変更し、
  Gemma logits collectorの不要なPathBuf借用2箇所を除いた。Clippyの指摘を抑制属性で隠していない。
- C++の3ファイルを固定clang-format18で整形した。演算順、selector条件、数値基準は変更していない。
- 初回ローカルsemantic契約96件はPASS（93.075秒）。公開CIの過去1件失敗の原因はこの結果だけで解消扱いにせず、
  診断保存を修正したCIで確認する。登録H0内での再実行も96件PASS（93.599秒）。

- Rust依存の追加は`sllm-hip -> tokenizers`のnormal edgeだった。既存Phase76/78/79とGQA6のtarget登録も
  manifestへ反映し、固定件数をschema／validatorへ重複記載する構造を廃止した。正本manifestの自己整合と
  Cargo metadataの完全比較を維持し、差分を最大3件まで具体表示する。関連25 testとvalidatorはPASS。
  H0実行で検出したPhase6 A2の現行closure要約も454へ同期し、validatorと関連6 testはPASS。
- Clippy workspace/all-targets/all-features、Rust1.85 MSRV check、C++host build、
  clang-format18検査、native public-runtime host testは修正後PASS。

- H0の重複したsllm-tools Rust testをH1 workspaceへ集約した。H1は`--no-run`のbuildとtestを分離し、
  build並列2、incremental無効、dev/test debug情報0、build上限4GiB／test上限3GiBを設定した。
  network namespaceのsudo経路でも同じCargo設定を維持する。空のtargetでH1は1,479件PASS、
  build peak RSS 1,569,906,688 B、test peak RSS 498,139,136 B、全体163.866秒だった。
  H2は38件PASS。これらはdirty localの開発検証であり、最終公開HEADのCI証拠とは区別する。
- hostは失敗commandのexit、RSS、timeoutを含む64KiB上限のsummaryを生成し、失敗時stdout/stderrをuploadする。
  report検証前の暫定状態と検証済み状態を区別する。失敗fixture、ログ制限、Cargo設定伝播などの関連64 testはPASS。
- public-runtime H3は公開C ABIを117関数へ同期し、新しいruntime include 2本をsource closureへ追加した。
  CIの直接HIP buildに不足していた`SLLM_HIP_COMPILE_TARGET`を設定した。gfx1030失敗時もgfx1201を検査し、
  cleanup前にreportと制限付き診断をuploadする。workflowと失敗伝播のfocused regressionはPASS。
  固定ROCmコンテナの実compile/link後に内部kernel symbol 103件、Graph関連HIP symbol 9件、
  テンプレートcompiler stub一覧の更新漏れも検出した。成果物schemaのdirect source一覧を100本へ同期し、
  schema不一致だった契約テスト5件は修正後PASS。生成済みELFでinspectionをまとめて確認する手順へ見直した。

## 公開前の最終検証

- 固定ROCmイメージでpublic-runtime compile/link/extract/inspectはgfx1030・gfx1201ともPASS。
  作業用clean clone `b5c5b8cb`の実source/matrixを検査した結果であり、mainの最終公開CIとは区別する。
  public-runtime契約85件も同じcloneでPASS（88.110秒）。
- H0は626件PASS（239.015秒）、H1は空のtargetで1,479件、H2は38件PASS。
  公開CIは実行結果を確認後に追記する。

生ログ、生成binary、モデルは追跡しない。

[計画](../../../../plans/active/2026/09/1-10/phase80-ci-restoration.md) /
[メイン計画](../../../../plans/main-plan.md)

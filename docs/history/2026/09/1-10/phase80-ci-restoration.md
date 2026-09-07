# Phase 80: CI修復の履歴

> 状態: 完了（2026-09-07）。実装修正commitの全CI成功を確認。
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
  公開CIの経過は以下に記録する。

公開commit `7b805da4e40834ac6a39b6e9205ce77b76da8f6a`でH1/H2と基本H3は成功した。
[public-runtime run](https://github.com/jyohukuchan/sLLM/actions/runs/34109526923)は、新設の診断ログ用directoryが
未追跡と判定されclean checkout検査に停止した。専用pathだけを`.gitignore`へ追加し、厳格なclean検査を維持して再公開する。
CI H1のbuild peak RSSは1,910,018,048 B、testは1,475,383,296 B、全体207.785秒だった。

修正commit `66061db4f3a5f6c4cfaf85c8d926e0cba65d7e44`の
[基本H3](https://github.com/jyohukuchan/sLLM/actions/runs/34110212509)と
[public-runtime H3](https://github.com/jyohukuchan/sLLM/actions/runs/34110212521)は成功した。
後者は両targetのcompile/link/extract/inspectとstrict aggregateを完了した。
一方、先行[host run](https://github.com/jyohukuchan/sLLM/actions/runs/34109526892)のH0は656.718秒・RSS1,520,492,544 Bで
全40 commandを資源内で実行し、semantic契約の1件だけが失敗した。新しいstderr artifactで、
closed environmentを検査する負例がGitHubのPython実行ファイルとlocal固定pinの違いで先に止まることを確認した。
実行時のPython pinを緩めず、先行するpin rejectionもe2e負例の有効な拒否として扱う。
加えてtest専用のsealed sourceコピーへ実行Pythonのpinを注入し、`PYTHONPATH`追加、余分なtree変数、`PATH`改変が
実際の環境guardで拒否されることを検査する。production source/pinは不変で、fixtureをGPU証拠にはしない。
controller全14 testはPASS（1.781秒）。

## 完了時の公開CI

実装修正の最終commitは `bea35c9c37afda928644338434c1a42b7075cba7`。
以下はすべてこの同一HEADの成功結果であり、過去commitのPASSで置き換えていない。

| CI | 結果・範囲 |
| --- | --- |
| [host-required](https://github.com/jyohukuchan/sLLM/actions/runs/34111302411) | H0 627件、H1 1,479件、H2 38件とrequired集約がPASS。semantic契約97件を含む |
| [基本H3](https://github.com/jyohukuchan/sLLM/actions/runs/34111302373) | gfx1030、gfx1201、集約PASS |
| [public-runtime H3](https://github.com/jyohukuchan/sLLM/actions/runs/34111302300) | 両targetのcompile/link/extract/inspectとstrict集約PASS |

H0は612.461秒、peak RSS 1,522,634,752 B。H1は214.736秒、build peak RSS 1,849,626,624 B、
test peak RSS 1,524,707,328 Bで、cold runner上の明示資源予算内に収まった。
zero selection、想定外skip、timeout、RSS超過を成功へ読み替えていない。
HIP証拠はcompile-onlyであり、新しいGPU numerical correctnessや性能測定は主張しない。

受入条件を完了し、計画をarchiveへ移した。後続はPhase81 static FP8 KV／MTP／文章生成、
Phase82他精度、Phase83 NVFP4 batchingとする。以後もPhase完了時のpushとCI確認・必要な修正を継続する。
この完了記録のcommitは文書だけを変更し、実装修正commitからsource／build inputs／toolchain／モデル／artifactを変更しない。
完了記録の最終公開HEADもCIを監視し、そのcommitとrun URLは公開後の完了報告に記載する。

生ログ、生成binary、モデルは追跡しない。

[計画](../../../../plans/archive/2026/09/1-10/phase80-ci-restoration.md) /
[メイン計画](../../../../plans/main-plan.md)

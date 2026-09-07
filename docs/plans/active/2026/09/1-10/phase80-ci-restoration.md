# Phase 80: CI修復と公開後の結果確認

> 状態: 実装・検証中（2026-09-07着手）
> 作成日: 2026-09-07
> 根拠: ユーザー指示によるCI確認・必要な修正の完了手順への追加と、既存後続Phaseの繰り下げ。

## 目的と範囲

現行CIがコード・依存関係の変更を正しく検査し、失敗理由を調査でき、通常のrunner資源で完走する状態へ戻す。
Phase完了時は最終commitのpush後にCIを監視し、必要な修正と再push後の結果まで確認する。
恒久方針は[main-plan](../../../../main-plan.md)の「Phase完了時のcommit・pushとCI確認」を正本とする。

Phase 79は実装完了の履歴を維持し、公開後に判明したCI残件を本Phaseで引き継ぐ。
旧Phase 80のstatic FP8 KV／MTP／文章生成は81、旧81の他精度最適化は82、旧82のbatchingは83へ移す。
既存の数値基準、CPUによるGPU-scale検証禁止、GPU PASSの意味は変更しない。
大規模runtime分割、モデル最適化、サービス再配置、GPU性能再測定、requiredへの昇格は対象外とする。

## 着手前に確認済みの失敗

調査対象は公開commit `50208e8e921975c2e5795fcb81315f698b1b42b2`。
開始時には最新runを再確認し、既に解消した失敗を再修復しない。

| 対象 | 確認済みの事実 | 初期対応 |
| --- | --- | --- |
| public-runtime H3 | headerに検査一覧外のC ABI関数8個。gfx1030のcompile前に停止し、gfx1201はskip | 宣言・実装・binding・export／link検査と期待manifestを同期 |
| Rust依存closure | validatorをローカルで再現。packageは191個、edgeは期待453に対し454 | 追加edgeの意図を確認し、manifestと検査の件数固定を整理 |
| H0 phase46-tools-rust | RSS 2,164,662,272 Bが制限2,147,483,648 Bを超え、runner内の監視処理が停止 | H1とのテスト重複、compile並列数、debug情報、buildとtestの予算を整理 |
| H0の他失敗 | Clippy、C++整形、schema/manifest/workflow、rmsnorm semantic契約（96件中1件） | stderrを取得・再現し、実際の不具合か古い検査期待値かを判定して修正 |
| 診断保存 | hostはstdout/stderrを生成するがuploadは主にreport/hash。public H3は失敗時uploadをskip | 失敗したstepの要約とサイズ制限付きログを保存 |
| 動作中の検査 | H1、H2、基本HIP compileは成功 | 既存coverageを維持し、再構成の退行を検査 |

根拠: [host run](https://github.com/jyohukuchan/sLLM/actions/runs/34096040195)、
[public-runtime H3 run](https://github.com/jyohukuchan/sLLM/actions/runs/34096040216)。
Rust停止はCI側RSS制限によるものであり、runner全体のOOMとは断定しない。
raw log/reportの作業用コピーは`.local-artifacts/ci-audit/`にあり、追跡しない。

## 作業順と成果物

1. **診断を保存する。** `run_host_suite.py`、public-runtime runnerとworkflowを確認し、失敗step、command、exit code、
   timeout/RSS判定をjob summaryへ出す。制限付きstdout/stderrと失敗reportをcleanup前に保存する。
   秘密を含む環境変数全体や大容量binaryをartifactにしない。失敗・欠落を集約処理で成功へ変換しない。
2. **公開APIと依存manifestを同期する。** 新しい8関数の実装・宣言・binding・linkを確認する。
   依存の追加edgeとfeature、source、lock整合を確認する。個数をvalidatorやschema等へ重複固定する構造を整理し、
   正本manifestから期待値を導出して追加・削除の差分を表示する。manifest更新は意図を確認して行い、CI内で自動承認しない。
   不足symbol、署名／ABI不整合、意図しない依存変更の検出は維持する。
3. **Rustのビルド・テスト構成を修正する。** H0内`sllm-tools`とH1 workspaceのcoverageを比較し、重複を除く。
   コンパイルとtest実行の予算を分け、明示したbuild並列数とCI用debug情報設定でpeak RSSを観測する。
   予算はrunnerでの実測と余裕から決め、tiny oracleの上限をコンパイラへ一律適用しない。
   cacheはcold buildを成立させた後に必要性を判断し、cache命中を成功条件にしない。
4. **その他の失敗を直す。** Clippy、C++整形、schema、semantic契約の失敗をfocusedに再現して修正する。
   実装の不具合と正当な変更への期待値追従を区別し、古いPhaseの固定値だけを維持するテストは契約に即して見直す。
5. **CIと同じ入口で確認して公開する。** 影響するhost rowと両targetのpublic-runtime compile-onlyを実行する。
   main-plan／計画／履歴を更新してcommit・pushし、最終HEADの期待workflow/job終了まで監視する。
   失敗した場合は診断から修正し、再検証・commit・push後のCIを確認する。

実装対象の入口: `.github/workflows/host-required.yml`、`h3-public-runtime-compile.yml`、
`ci/tools/run_host_suite.py`、`run_h3_public_runtime_compile.py`、`validate_rust_dependencies.py`、
`ci/matrix/host-v1.json`、`suites-v1.json`と関連schema／manifest／tests。
検査の無効化、全エラーのsoft-fail、timeoutの一律延長だけで終了しない。

## 2026-09-07の実行順の見直し

public H3はcompile前の公開ABI確認を直した後、compile/link後の内部symbol／compiler stubと成果物schemaにも
古い期待一覧が残っていると分かった。同じcompile単位で検査段階ごとの再実行を続けず、生成済みELFを使って
host/device inspectionの全段階と関連schemaをまとめて同期し、その後に両targetの最終compileを1回行う。
H0再実行中のrunner更新でsource hashの一時的不一致も出たため、最終H0はsource manifest更新が揃ってから行う。
受入条件・数値基準・scopeは変更せず、既に成功した独立検査はその変更範囲で再利用する。

## 完了条件と検証

以下を実装開始時の受入範囲とする。ユーザーのCI修復依頼を具体化した条件であり、後続機能の検証範囲は拡大しない。

- [x] 公開APIと依存closureが現行コードに一致し、不足・意図しない差分のnegative testも機能する。
- [x] Rust host build/testが明示した資源予算で完走し、重複削除前の必要coverageを維持する。
- [x] 確認済みH0失敗を解消し、H0/H1/H2が成功する。
- [ ] 基本H3とpublic-runtime H3が成功し、public-runtimeのgfx1030/gfx1201の両rowでcompile/link/inspectが実行される。
- [x] 失敗fixtureで診断・制限付きログが残り、失敗／欠落／想定外skip／cancelを成功として集約しない。
- [ ] 最終公開HEADの対象CI成功、commitとrun URL、資源観測値と適用範囲を記録する。

変更中はfocused test、統合時に影響する登録済みrow、公開後に実際のGitHub CIを確認する。
CI復旧のために過去Phaseの全GPU測定を再実行しない。runtime/kernelの意味を変更する修正が必要と判明した場合は、
その変更に限り既存方針に従ったGPU検証を追加する。外部障害等は原因を記録して確認待ちとし、PASSにしない。

## 記録と引継ぎ

完了時に本計画をarchiveへ移し、`docs/history/2026/09/1-10/phase80-ci-restoration.md`へ
原因・変更・検証・最終公開commit/runを記録して計画と相互リンクする。現在は未完了なので履歴の完了記録は作らない。
後続のPhase 81はCI修復完了後に開始する。

[メイン計画](../../../../main-plan.md) /
[後続ロードマップ](phase76-qwen38-27b-nvfp4-priority-roadmap.md) /
[CI・テスト方針](../../08/1-10/ci-test-strategy.md)

# Phase XA履歴: CI修正・公開・push後確認

## 2026-08-25: 計画作成

- ユーザー指示により、CI修正、GitHubへのpush、push後に問題が発生しなかったことの確認をPhase XAへ割り当てた。
- 開始時の`origin/main`は`159bc526cb26d180161f2cd7abcc22abb7e67e84`、ローカル`main`はPhase 52の
  実装とcloseoutの2 commitだけ先行し、working treeはcleanだった。
- 最新公開commitのhost-required run `32681109190`はH0/H1がFAIL、H2がPASSだった。H0ではprocfs監視の
  INFRA_ERROR、MSRV timeout、C++ formatと後続contract failure、H1ではworkspace testの120秒timeoutを確認した。
- public-runtime H3 run `32681109285`は`gfx1030` compile/link/extractで失敗し、`gfx1201`を実行できず、
  aggregateとcleanupもFAILした。
- ローカル確認ではRust workspace testとC++ public runtime host testはPASSした一方、C++ formatは11ファイル
  395箇所でFAILし、Phase 52 summary test単独実行は`run_phase50_r9700_sllm`のimport順序依存でFAILした。
- Phase XAの受入条件を、format、test isolation、host resource/timeout、public H3 2 target、required aggregate、
  文書同期、commit/push、push後workflowの全完了へ固定した。GPU実行やPhase 51再開は範囲外とした。

## 2026-08-25: 原因と採用修正

- C++ format違反11ファイルをclang-formatへ同期し、Phase 52 summary aggregatorが単独importでも`ci/tools`を解決するようにした。
- procfs監視は終了競合で一時的に不完全な`/proc/<pid>/stat`を最大3回再読込し、永続的な不正値は`EPROTO`のまま
  fail-closedにした。H0/H1のcommand budgetを300秒、H1 jobを15分へ整合し、H1 Rust testは4 jobs/4 threadsへ固定した。
- public-runtime H3はproduction direct closureをMoEとtoken selectorまで拡張し、public ABI 98件、kernel symbol 53件、
  causal-attention compiler stub 18件、HIP undefined symbol 52件をexact setとして検査するようにした。統合レビューで検出した
  public ABI 72件対header 98件の不一致も修正し、focused re-reviewでmissing/extraなしを確認した。
- 最初の公開候補`1f9fa6c941c9c55be18244d395d4aec862fa0740`ではH0の600秒行上限不足が判明したため、
  `382c26a27da2b95e76482b666add07091cc93cf5`で行上限を720秒へ更新した。次の公開runで、先行MSRV gateと
  dependency validatorが異なるCargo command/environmentを使い、コールドビルドを2回行うことを特定した。
- `2c28cf0811f09b9e346c6f58250289912790a83b`では両者を
  `cargo +1.85.0 check --jobs 1 --workspace --all-targets --locked --offline --target x86_64-unknown-linux-gnu`と
  同一のB0 sanitized environmentへ統一した。standalone dependency checkは省略せず、同一入力のCargo cacheだけを再利用する。

## 2026-08-25: ローカル検証

- 固定Python 3.12.10で最終候補`2c28cf08`のH0 strictを実行し、585/585 PASS、244.781秒、
  `immutable=true`、reviewed/tested/workflow SHA完全一致を確認した。
- 空の専用`CARGO_TARGET_DIR`による連続実測は、先行MSRV gateが184.79秒、後段dependency validatorが0.69秒で
  いずれもPASSした。後段のstandalone cargo checkが同じbuild cacheを再利用することを確認した。
- H0 draftは585/585 PASS、関連Rust contract/dependency unitは32/32 PASS。最終修正前の全体候補でもH1は
  969/1004 selected、H2は38/47 selectedでPASSし、固定ROCm image内の通常H3とpublic-runtime H3は
  `gfx1030`/`gfx1201`およびaggregateがPASSした。

## 2026-08-25: commit、push、公開後確認

- `3ed002c476b49417cc702119e37c2389cefb96bc` `fix: make long RDNA KV allocation fail closed`
- `d7e6821382b6bf5ec8fb94a80fd6f813e68eeac5` `docs: close Phase 52 with R9700 100k evidence`
- `1f9fa6c941c9c55be18244d395d4aec862fa0740` `fix: repair Phase XA CI contracts`
- `382c26a27da2b95e76482b666add07091cc93cf5` `fix: extend H0 cold-build row budget`
- `2c28cf0811f09b9e346c6f58250289912790a83b` `fix: reuse H0 MSRV build evidence`
- すべて通常pushで`origin/main`へ公開し、force pushや共有履歴の書換えは行っていない。
- 中間のhost-required run
  [32808395269](https://github.com/jyohukuchan/sLLM/actions/runs/32808395269)はH0 600秒上限、
  [32809540571](https://github.com/jyohukuchan/sLLM/actions/runs/32809540571)はH0 720秒内の重複コールドビルドでFAILした。
  いずれもartifactを確認して原因を修正し、失敗を無視または再実行だけで済ませなかった。
- 最終候補`2c28cf08`の
  [host-required run 32811462527](https://github.com/jyohukuchan/sLLM/actions/runs/32811462527)はPASSした。
  H0は585/585 PASS・587.830秒、H1は969/1004 selected PASS・160.522秒、H2は38/47 selected PASS・
  1.115秒で、required aggregateもPASSした。
- [通常H3 run 32811462546](https://github.com/jyohukuchan/sLLM/actions/runs/32811462546)は`gfx1030`、`gfx1201`、
  aggregateがPASSした。
- [public-runtime H3 run 32811462536](https://github.com/jyohukuchan/sLLM/actions/runs/32811462536)は固定image、
  networkなしで両targetのcompile/link/extract/inspect、exact needs aggregate、artifact upload、cleanupがPASSした。
- Phase XAの全受入条件を満たしたため完了とした。H3結果はcompile-only証拠であり、GPU runtime、numerics、
  performance、モデル推論のPASSには読み替えない。

[対応する計画](../../../../plans/archive/2026/08/21-31/phase-xa-ci-repair-and-push.md)

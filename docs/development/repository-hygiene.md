# Repository hygiene policy

## 目的

旧版で発生した、巨大な未追跡workspace、過剰なtracked artifact、多数のworktree、長期間未pushのcommitを再発させない。Gitにはsource、設定、文書、小さなtest fixture、再現用manifest、集約結果だけを保存し、model、binary、raw trace、profile、生成物は保存しない。

この方針はデータを自動削除するためのものではない。上限超過時は処理を停止して対象を報告し、所有者の確認なしにdirty worktree、未追跡file、artifactを削除しない。

## Gitへ保存するもの

- source code、build script、schema、workflow。
- 人がreviewできる文書と計画。
- 小さく決定的なtest fixture。
- model lock、artifact manifest、hash、外部保存先、集約metric。
- 再生成手順と、その手順を固定するtool revision・引数。

## Gitへ保存しないもの

- model weight、model cache、tokenizer cacheの複製。
- build directory、compiler output、実行binary、object、package。
- raw profiler output、trace、memory dump、core dump、長大なlog。
- benchmarkの全raw sampleや生成済みgraph用中間data。
- large model slice、生成fixture、変換途中のtensor。
- secret、token、password、private key。

ローカル生成物の標準配置は`.local-artifacts/`配下とし、種類別のignored directoryへ置く。Gitには必要なsummary、manifest、hashだけを別pathへ保存する。

## H0のtracked tree検査

H0はGit treeとPR差分だけを対象にし、filesystem全体を再帰scanしない。

- 新規・変更tracked blobが1 MiBを超えた場合はwarning、10 MiBを超えた場合はfailure。
- 1変更で追加するtracked contentが合計50 MiBを超えた場合はfailure。
- 新規tracked pathが200件を超えた場合はwarning、500件を超えた場合はfailure。
- tracked pathの純増が500件を超えた場合はfailure。
- `.local-artifacts/`、`tests/fixtures/generated/`、およびpolicyで禁止したraw artifact pathがindexへ入っていた場合はfailure。
- `.gitignore`は強制追加を防げないため、ignoreとH0 index検査の両方を使う。

閾値を超える正当なsourceまたはfixtureは、内容、必要性、代替案、license、owner、削除条件を記録した明示的なallowlist entryとreviewを必要とする。単にignore patternを広げて隠さない。

## ローカルworkspace検査

untracked、ignored data、worktree、remote同期はGitHub-hosted H0ではなく、local hygiene commandとpush前検査で確認する。

| 対象 | warning | 作業停止 |
| --- | ---: | ---: |
| non-ignored untracked data | 256 MiB | 1 GiB |
| checkout内のignored/generated data | 10 GiB | 20 GiB |
| `.git`を除くcheckout全体 | 20 GiB | 30 GiB |
| registered worktree | 3 | 4超 |

- model weightはcheckout外の検証済みcacheへ置く。
- G2等のmodel sliceは検証済みread-only cacheから実行時に抽出し、`.local-artifacts/model-slices/`等のlocal-only領域へ一時保存する。raw sliceをGit、GitHub Actions artifact、JSON reportへ埋め込まず、lock fingerprint、tensor、byte range、recipe、size、SHA-256、短いsummaryだけを追跡する。
- `reference/`のsource mirrorは登録したrepositoryと固定commitをmanifestへ記録し、生成物やmodelを混在させない。
- 上限超過時はsize、file count、上位directoryをread-onlyで報告し、自動削除しない。
- `git worktree prune --dry-run --verbose`でstale registrationを検出する。
- missing/invalid worktree pathは即時報告する。
- clean、unlocked、非mainで14日間活動のないworktreeはstale候補として報告する。dirty worktreeは自動削除しない。

## commitとremote同期

- `main`はmilestone完了、handoff、push作業完了時に`origin/main`よりahead 0、behind 0とする。
- feature branchはupstreamを設定し、ahead数と最終活動日時をlocal hygiene reportへ含める。
- feature branchが20 commit超ahead、7日超未push、またはowner不明のupstream未設定状態になった場合はhandoffを停止する。
- 通常開発中の短期的な未commit・未push状態は許容するが、governance、license、test policy等の開発前baselineを長期間localだけに置かない。
- force pushや共有済み履歴の書換えによって同期問題を解消しない。

## 実装済みの検査入口

Phase 1で次を追加した。

- `ci/tools/tracked_tree.py`: Git index/treeとbase revisionの差分を検査するH0 script。
- `ci/tools/local_hygiene.py`: filesystemをread-onlyで集計するlocal hygiene command。
- `ci/policy/hygiene-allowlist-v1.json`と`ci/schema/hygiene-allowlist-v1.schema.json`: machine-readableなallowlistとschema。
- 両commandはfile size、file count、worktree、branch activity、ahead/behindを含むJSON summaryを出力する。
- 恒久的なcommandと出力先は[host build and test entry points](testing.md)を正とする。

これらのcommandは上限超過を報告して非zero終了するが、dirty worktree、未追跡file、ignored artifactを削除しない。

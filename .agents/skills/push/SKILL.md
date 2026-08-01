---

name: push
description: uLLM-project全体の変更を確認し、未コミット変更と未公開のローカルコミットを必要最小限のコミットに整理して、現在のブランチをGitHubへpushする。ユーザーが明示的にcommit、push、変更の公開、またはpush前のコミット整理を依頼した場合に使用する。共有済み履歴の書き換えやforce pushには使用しない。
---

# 目的

このSkillを含むuLLM-projectのGitリポジトリ全体を対象として、変更内容を安全にコミットし、現在のブランチをGitHubへpushする。

原則として、同じ目的の変更は1つのコミットにまとめる。独立した変更としてレビュー、取り消し、またはcherry-pickする必要がある場合のみ、複数のコミットに分ける。

# 安全上の制約

* ユーザーが作成した変更を破棄しない。
* `git reset --hard`、`git clean -f`、`git checkout -- <path>`を使用しない。
* `git push --force`および`git push --force-with-lease`を使用しない。
* remoteに存在するコミットの履歴を書き換えない。
* rebase、merge、cherry-pickの途中、detached HEAD、または未解決のconflictがある場合は処理を停止して状況を報告する。
* Gitのユーザー名、メールアドレス、remote URLなどの設定を変更しない。
* tagの作成、releaseの作成、Pull Requestの作成は行わない。
* `.env`、秘密鍵、認証情報、token、個人用設定など、秘密情報の可能性があるファイルをコミットしない。
* push先がGitHubでない場合はpushせず、検出したremoteを報告する。

# 手順

## 1. Gitの状態を確認する

以下を確認する。

* Gitリポジトリのルート
* 現在のブランチ
* working treeとstaging areaの状態
* untracked files
* remoteとupstream branch
* upstreamに対するahead／behind
* merge、rebase、cherry-pickの進行状態

リポジトリのルートが、このSkillを含むuLLM-projectであることを確認する。

remoteの最新状態を確認するため、適切なremoteに対して`git fetch --prune`を実行する。

## 2. すべての変更をレビューする

次の内容を確認する。

* staged changes
* unstaged changes
* untracked files
* 削除されたファイル
* renameされたファイル
* upstreamにまだ存在しないローカルコミット

変更内容を読まずにコミットしない。

すべての変更が今回pushすべき内容か確認する。生成物、一時ファイル、デバッグ用変更、秘密情報の疑いがあるファイルは除外する。

判断できない変更がある場合は、勝手に破棄したりコミットしたりせず、対象ファイルと理由を報告する。

## 3. 必要な検証を実行する

`AGENTS.md`、package scripts、Makefile、既存CI設定などを確認し、変更に必要なlint、typecheck、test、buildを実行する。

検証に失敗した場合はpushしない。失敗したコマンドと主要なエラーを報告する。

検証を実行できない場合は、その理由を明示する。

## 4. コミット構成を決める

コミット数は必要最小限にする。

基本方針:

* 同じ目的の変更は1コミットにまとめる。
* 実装と、その実装に対応するテストは原則として同じコミットに含める。
* formattingだけの変更を不必要に別コミットへ分けない。
* 独立した変更や、個別に取り消す必要がある変更だけ分割する。
* 意味の異なる無関係な変更を、コミット数削減だけを目的に混在させない。

既存コミットをまとめる場合:

* upstreamに存在しないローカル専用コミットだけを対象にする。
* remoteから到達可能なコミットは書き換えない。
* ローカル専用コミットがすべて同じ目的なら、原則1コミットにまとめる。
* 複数の独立した目的がある場合は、目的ごとの最小コミット数に整理する。
* 安全な基準点を特定できない場合は履歴を変更しない。

## 5. 変更をstageしてcommitする

変更内容を確認した後で、意図したファイルだけをstageする。

すべての変更を対象にする場合でも、対象一覧を確認せずに機械的にstageしない。

コミットメッセージは、実際の変更目的を簡潔に表す。リポジトリに既存のコミット規約がある場合は、その規約に従う。

`update`、`fix stuff`、`changes`など、内容が分からないメッセージを使用しない。

コミット後、working treeが意図した状態になっていることを再確認する。

## 6. upstreamとの差分を確認する

push前に再度fetchし、現在のブランチとupstreamの関係を確認する。

upstreamが先行している場合:

* working treeがcleanであることを確認する。
* ローカル専用コミットをupstream上へrebaseする。
* conflictが発生した場合は自動的に推測して解決せず、rebaseを中断または安全に停止して状況を報告する。
* merge commitは、リポジトリの既存方針で要求されない限り作成しない。

## 7. GitHubへpushする

remote URLがGitHubを指していることを確認する。

upstreamが設定済みの場合は、現在のブランチをそのupstreamへpushする。

upstreamが未設定の場合は、適切なGitHub remoteへ現在のブランチをpushし、upstreamを設定する。

現在のブランチ以外をpushしない。

## 8. 結果を報告する

完了時に、以下を簡潔に報告する。

* pushしたブランチ
* push先remote
* 作成または整理したコミットのhashとmessage
* 実行した検証と結果
* push後のworking treeの状態
* 除外した変更や未解決事項

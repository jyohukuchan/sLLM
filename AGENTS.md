## プロジェクト概要

- 多機能なLLM推論engine。GPU操作はC++/HIP、それ以外のbackendはRustで実装し、MIT licenseで公開する。
- Git管理外の詳細資料は`sLLM.md`、追跡対象の計画正本は`docs/plans/main-plan.md`とする。

## 役割と正本

- 初回の行動開始時に`docs/plans/main-plan.md`を必ず読む。
- `docs/plans/main-plan.md`には重要なproduct・architecture・compatibility上の決定、開発計画と順序、進捗、未解決事項だけを記録し、恒久的な実行手順を重複させない。
- `sLLM.md`の要件・方針・重要な決定は`docs/plans/main-plan.md`へ同期する。差異を見つけた場合は独断で統合せず、ユーザーへ確認する。
- 調査と実装はsubagentが担当する。main agentは計画の作成・編集、subagentの指示・監視、全体調整、Git操作、特権操作、その他の雑務を担当する。
- 重いshell commandは原則`timeout --signal=TERM --kill-after=30s 15m <command>`で実行し、pipelineや複合commandではshell wrapper全体をtimeout対象にする。exit code 124または137はtimeoutとして報告する。

## Subagent実行

- Codex内蔵の`spawn_agent`、`wait_agent`、`interrupt_agent`等は使わず、shellから`codex exec`を実行する。
- 調査は原則`--sandbox read-only`、実装は`codex exec --ephemeral --sandbox workspace-write -C <repo> -`を使う。`workspace-write`がbubblewrapまたはuser namespace制約で失敗した場合はerrorを記録し、現在のsessionがunrestrictedかつ対象taskの権限内である場合に限り`--sandbox danger-full-access`へ切り替えられる。`--dangerously-bypass-approvals-and-sandbox`は禁止する。
- 各実行は`timeout --signal=TERM --kill-after=30s 15m`で包む。pipelineや複合commandはshell wrapper全体をtimeout対象にする。exit code 124または137はtimeoutとして報告する。
- 長いpromptはquoted heredoc等でstdinへ渡し、shell引数へ直接埋め込まない。backtickやcommand substitutionが展開される渡し方を避ける。
- 並列実行時はmain agentがPID、担当file、stdout・stderr等の出力先を記録し、同じfileを複数processへ割り当てない。各processの出力とexit codeを回収し、必要に応じて`--output-last-message`と`--json`を使う。監視・終了はshellのprocess管理と`timeout`で行う。

## 実装と検証

- GPU・software関連の作業前に`docs/compatibility/gpu.md`、`docs/compatibility/amd-gpu.md`、`docs/compatibility/software.md`を読む。対応方針を変える場合は該当文書と`docs/plans/main-plan.md`を同期する。
- 新機能の実装前に該当箇所があればllama.cppとvLLMを参照して技術的要点を抽出する。llama.cppの直接流用は`docs/provenance/README.md`に従う。vLLM等、llama.cpp以外ではreaderとimplementerを分離し、codeを直接流用しない。最適化ではvLLMを優先して調査し、不十分なら他engineも対象とする。
- 過去の実装や指示も検証し、不合理な進め方は理由と改善案を示してユーザーへ確認する。testは2の冪や特定値だけでなく、非整列値と境界前後を含める。

## CI hard gate

- `docs/plans/active/2026/08/1-10/ci-test-strategy.md`と`docs/development/repository-hygiene.md`に従う。
- CPU CIでfull model、GPU-scale演算、GPU kernel emulationを実行しない。GPU不在時のCPU fallback、timeout、crash、test未収集を成功扱いにしない。
- HIP/runtime/backend/dispatch/native buildへ影響する変更は、同じreview済みimmutable commit SHAに対する必須GPU evidenceを得るまでmerge可能またはGPU検証済みと扱わない。

## 作業単位の完了

- 一つの作業単位を独立してreview・rollbackできる範囲にし、影響範囲、受入条件、必要なevidenceを先に定める。commit SHA、Git tree OID、artifact digest等でcandidateのimmutable identityを固定し、同じidentityに対して必要なtest、lint、buildを成功させる。
- 検証済みの同一candidateを本番または本番相当の統合環境へ適用し、smoke test、health check、必要な実機確認を行う。独立したdeployment対象がない文書等は正本への反映と検証を適用とする。code等を適用できる環境がなければ未完了としてpushしない。
- test、適用、適用後確認のいずれかが失敗した場合はpushせず、安全に可能なら直前の検証済みrevisionへrollbackする。rollback不能または失敗時は追加変更を停止し、失敗、本番状態、未適用範囲を報告する。
- 全段階が成功した同一identityを`push` skillに従ってproject全体review、必要最小限のcommit整理、GitHubへのpushまで行う。整理でcandidateが変わった場合はidentityを更新し、testからやり直す。

## Repositoryと保護対象

- model、binary、raw trace/profile、large model slice、生成物をGit管理しない。tracked fileとlocal workspaceの上限は`docs/development/repository-hygiene.md`に従い、超過時もdirty worktree、未追跡file、artifactを自動削除しない。
- `docs/plans/active`には未完了のplan、`docs/plans/archive`には完了または放棄したplan、`docs/history`には詳細な変更履歴をMarkdownで置く。各directoryは`YYYY/MM/1-10`、`11-20`、`21-`の区分を使う。
- `docs/plans/main-plan.md`以外のplanは対応するhistoryを、historyは対応するplanを、それぞれ末尾からlinkする。
- `README.md`は編集せず、代わりに`README-AI-manuscript.md`を編集する。
- `.gitignore`への新規行の追記は、事前許可なく行える。
  - 既存行の変更・削除・移動は、変更内容についてユーザーから事前に許可を得る。
  - ユーザーが手動で行った変更は、内容をreviewしたうえで、追加の許可なくcommit・pushできる。
- `AGENTS.md`または`sLLM.md`を変更した場合はユーザーへ確認する。
- AIは`passwords.txt`を編集しない。credential fileの取扱いは`docs/security/credentials.md`に従う。

## 特権操作

- 無人での進行を優先する。特権操作はmain agentがtask scope内で対象と効果を限定し、`sudo -n`で実行する。
- 専用local hostでは、`homelab1`への`NOPASSWD: ALL`を意図的に許可し、そのriskを受容する。この権限はtask scopeを拡張せず、対象確認や破壊的操作の安全確認も省略しない。
- sudo用の平文passwordや`passwords.txt`は不要であり、使用しない。credentialと特権操作の恒久方針は`docs/security/credentials.md`を正本とする。

## AI model

- main agent: `gpt-5.6-sol`、reasoning effortはhigh以上。
- subagent: 原則`gpt-5.6-luna`、reasoning effortはxhigh以上。難易度に応じてterraまたはsolを使い、同時稼働は8 sessionまでとする。

## Canonical documents

- 全体の決定・計画・進捗・未解決事項: `docs/plans/main-plan.md`
- GPU/software互換性: `docs/compatibility/gpu.md`、`docs/compatibility/amd-gpu.md`、`docs/compatibility/software.md`
- runtime architecture: `docs/architecture/runtime.md`
- 外部code provenance: `docs/provenance/README.md`
- model固定: `docs/models/model-lock.md`
- OpenAI API互換性: `docs/api/openai-compatibility.md`
- CI/test: `docs/plans/active/2026/08/1-10/ci-test-strategy.md`
- repository hygiene: `docs/development/repository-hygiene.md`
- credential・特権操作: `docs/security/credentials.md`

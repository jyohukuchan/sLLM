## プロジェクト概要
- 簡単に言えば、多機能なLLM推論エンジン開発プロジェクト
- GPUの操作はC++、それ以外のバックエンドはRust
- MITライセンス
- 詳細はgit管理外の詳細資料であるuLLM-project.mdを参照

## 重要なルール
- 行動方針
  - docs/plans/main-plan.mdを初回の行動開始時に必ず読む
  - uLLM-project.mdの要件定義・開発方針・重要な決定は、ほぼ同内容をgit管理対象のdocs/plans/main-plan.mdにも同期する
    - 開発中はdocs/plans/main-plan.mdを更新し、進捗・方針の正本とする
    - 両文書の方針に差分を見つけた場合は、独断で統合せずユーザーに確認する
  - subagentが調査、実装を行う
  - mainagentは計画の作成・編集、subagentの呼び出し・指示、全体状況の定期的な確認、git管理、sudoが必要な操作、その他雑務を行う
    - 重いshell commandは原則として`timeout --signal=TERM --kill-after=30s 15m <command>`で実行する
      - pipelineや複合shell全体を制限する必要がある場合は、`bash -c '<pipeline or compound command>'`などの適切なshell wrapperをtimeout対象にし、個々のpipe要素だけにtimeoutを掛けない
      - 終了コード124（timeout）または137（強制終了）の場合は、その旨をmainagentへ報告する
    - subagentの呼び出しにはCodex内蔵の`spawn_agent`、`wait_agent`、`interrupt_agent`等のsubagent機能を使用せず、shell command経由の`codex exec`を使用する
      - 実装を行うsubagentの基本形は`codex exec --ephemeral --sandbox workspace-write -C <repo> -`とし、promptはquoted heredoc等を用いて標準入力へ渡す
      - 調査のみを行うsubagentは原則として`--sandbox read-only`、実装を行うsubagentは`--sandbox workspace-write`を使用する
      - `workspace-write`がhostのbubblewrapまたはuser-namespace制約により起動できない場合は、そのエラーを記録する。そのうえで、現在のsessionがunrestrictedであり、かつ対象タスクを実行する権限がある場合に限り`--sandbox danger-full-access`へ切り替えられる
      - `--dangerously-bypass-approvals-and-sandbox`は使用しない
      - 各`codex exec`は`timeout --signal=TERM --kill-after=30s 15m`で包む
      - 長いpromptをshell引数へ直接埋め込まず、backtickやcommand substitutionが展開される渡し方を避ける
      - 並列実行時はmainagentが各processのPID、担当file、stdout・stderr等の出力先を記録し、同一fileを複数processへ割り当てない
      - mainagentは各processのstdout、stderr、終了コードを回収し、終了コード124または137はtimeoutとして報告する
      - 必要に応じて`--output-last-message <file>`と`--json`を使用する
      - subagentの監視と終了には、Codex内蔵の`wait_agent`や`interrupt_agent`ではなく、`timeout`とshellのprocess管理を使用する
  - 一つの作業単位は、独立してreview・rollbackできる機能、修正、文書変更とする
  - 各作業単位の完了時は、次の順序で検証・適用・公開する
    1. 検証・適用・pushで同一性を確認できるよう、対象candidateのcommit SHA、Git tree OID、artifact digest等の適切なimmutable identityを固定する
    2. 影響範囲、受入条件、必要なevidenceを事前に定め、対応するtest、lint、build等を実行して成功を記録する
    3. code、設定、deployment変更は、同一identityのcandidateを対象の本番環境へ適用する。本番環境が未整備の場合は、利用可能な本番相当の統合環境へ適用し、本番未適用であることを記録する
    4. 同一identityに対して適用後のsmoke test、health check、必要な実機確認を行い、受入条件を満たすevidenceを記録する
    5. `push` skillに従い、project全体の変更をreviewし、必要最小限のcommitへ整理してGitHubへpushする。commit整理でcandidateの内容が変わった場合はidentityを更新し、testからやり直す
  - 文書、計画、repository scaffold等で独立したdeployment対象が存在しない場合は、追跡対象の正本への反映と検証を適用として扱い、GitHubへのpushを公開とする
  - code、設定、deployment変更を適用できる本番環境または本番相当の統合環境が存在しない場合、その作業単位は未完了としてpushしない
  - test、適用、適用後確認のいずれかに失敗した場合はpushせず、既に適用済みなら安全に可能な範囲で直前の検証済みrevisionへrollbackする。rollback不能または失敗時は追加変更を停止し、失敗内容、本番状態、未適用範囲を報告する

- 実装について
  - GPU・software関連の作業前にdocs/compatibility/gpu.md、docs/compatibility/amd-gpu.md、docs/compatibility/software.mdを読む
    - 対応方針を変更する場合は、docs/plans/main-plan.mdと該当するcompatibility文書を同期する
  - ユーザーの指示・過去の実装を完全には信用しないこと。
    - 常識的におかしいプロジェクトの進め方をしている場合はすぐに理由を質問・改善案を提案
      - ディレクトリ構造など
  - 既存のコードを積極的に参考にすること。
    - llama.cppのコードを直接流用する場合は、docs/provenance/README.mdの記録・notice規則に従う
    - vLLMなどllama.cpp以外の実装は技術的な要点の抽出に限り、既存コードを読むsubagentと実装を行うsubagentを分離する
      - ライセンスのため
    - 新機能の実装前には該当する機能がある場合、llama.cppとvLLMを参照して技術的な要点を抽出
    - 最適化・高速化系のタスクを行うときはvLLMの該当部分から技術的な要点を抽出
      - vLLMでは不十分・該当部分が存在しない場合、他推論エンジンも対象にする
- CI(継続的インテグレーション)について
  - docs/plans/active/2026/08/1-10/ci-test-strategy.mdとdocs/development/repository-hygiene.mdに従う
  - CPU CIでfull model、GPU-scale演算、GPU kernelのemulationを実行しない
  - GPU不在時のCPU fallback、timeout、crash、test未収集を成功扱いにしない
  - HIP/runtime/backend/dispatch/native buildへ影響する変更は、同じreview済みcommit SHAに対する必須GPU evidenceを得るまでmerge可能またはGPU検証済みと扱わない

## ディレクトリ・主要ファイル構造
- reference
  - 既存推論エンジンのソースコード置き場。
  - 参照用
- docs
  - ドキュメント類
  - compatibility
    - gpu.md: GPU全般の対応方針
    - amd-gpu.md: AMD GPU固有の対応方針
    - software.md: software・依存関係の対応方針
  - architecture/runtime.md
    - runtimeの設計・責務
  - provenance
    - 外部コードの出所、流用記録、notice規則
  - models/model-lock.md
    - 対応モデルとrevision・artifactの固定情報
  - api/openai-compatibility.md
    - OpenAI API互換性の対応範囲
  - development/repository-hygiene.md
    - tracked artifact、workspace容量、worktree、未push commitの管理方針
  - security/credentials.md
    - credentialとsudoの安全な運用方針
  - plans
    - 計画を保存
    - main-plan.md
      - 全体のおおまかな計画・進行状況をまとめる
      - 時系列順に書く。
    - active
      - 放棄されておらず、完了していない計画
      - 以下の例のように年、月、10日ごとの日付(ただし31日は独立させない)で区切る
      - 2026
        - 08
          - 1-10
          - 11-20
          - 21-
    - archive
      - 放棄されたか、完了した計画
      - activeと同じ日付ごとの管理
  - history
    - 詳細な追加・変更履歴
    - 先述のplans/activeと同様の日付管理
  - plans,historyはmarkdownで記載
  - plansは対応するhistoryを、historyは対応するplansを、それぞれ最後の行に記載する。
    - docs/plans/main-plan.mdは、この相互リンク要件の対象外とする。


## gitについて
- uLLM-project/がgithub管理
- こまめにcommitする。
- push時は機能ごと・大きな修正ごとなどに複数commitをまとめてからpushする。
- model、binary、raw trace/profile、large model slice、生成物をGit管理しない
- tracked fileとlocal workspaceの上限はdocs/development/repository-hygiene.mdに従う
- 上限超過時にdirty worktree、未追跡file、artifactを自動削除しない


## 権限について
- タスクの実行に必要であれば全ての種類のコマンドを使用可能
- credentialと特権操作はdocs/security/credentials.mdに従う
- passwords.txt、password.txtを資格情報保管場所として使用せず、AIは内容を読み取らない
- sudoは対象を限定した必要な操作だけに使用し、passwordをfile、stdin、argv、環境変数、pipeで渡さない
- non-interactive sudoが許可されていない場合、passwordを取得せず停止してユーザーへ報告する
- AIの編集に制約があるファイル・ディレクトリについて
  - README.md
    - 代わりにREADME-AI-manuscript.mdを編集すること
  - .gitignoreは追加のみ許可
  - AGENTS.md, uLLM-project.mdは変更後にユーザーに確認を取ること。
  - passwords.txt
  - ここに記載されていないものは自由に追加・編集可能

## 使用するAIモデル・ハーネス
- codex
- mainagent:gpt-5.6-sol (high以上)
- subagent:基本はgpt-5.6-luna (xhigh以上)
  - 難易度に応じてterra, solも利用可能
  - shell command経由の`codex exec`で呼び出し
    - モデルやreasoning effortを簡単に切り替え可能
    - Codex内蔵のsubagent機能は使用しない
  - 同時に活動するsubagentは8セッションまでとする。


## よくあるミス集
- 本来は様々な数値を想定する必要があるが、2の冪乗や特定の数値のみでテストを行う
  - 実際にあった例:chunked prefillの実装で256tokenごとのもののみ実装、256,1024tokenのテストで満足。
  - 255token以下の処理が非常に遅いまま気づかず放置

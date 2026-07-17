# Codex personal terra/luna agents

- 個人用カスタムエージェント `~/.codex/agents/terra.toml` を追加した。
  - モデル: `gpt-5.6-terra`
  - 推論強度: `xhigh`
  - 用途: 調査、探索、外部資料確認、ログ分析
  - 権限: 読み取り専用
- 個人用カスタムエージェント `~/.codex/agents/luna.toml` を追加した。
  - モデル: `gpt-5.6-luna`
  - 推論強度: `xhigh`
  - 用途: コーディング、要約、定型検証、反復作業
- ルート `AGENTS.md` に、作業内容に応じて `terra` と `luna` を自動選択する規則を追加した。
- 小さく密結合な作業や同一ファイルの並行編集は、自動委任の対象外とした。
- Python 標準の TOML パーサーで両ファイルを検証し、`TOML_OK` を確認した。
- `codex --strict-config` は `features` と `mcp` サブコマンドでは未対応だったため、この経路の検証には使用していない。
- 新しい個人用カスタムエージェントの読み込みには、Codex の再起動または新規セッションが必要。

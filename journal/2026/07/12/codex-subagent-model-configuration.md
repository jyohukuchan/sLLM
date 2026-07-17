# Codex subagent model configuration

## 結論

Codex の subagent モデルは、個人用 `~/.codex/agents/*.toml` またはプロジェクト用 `.codex/agents/*.toml` のカスタムエージェントファイルで指定する。

各ファイルに必須の `name`、`description`、`developer_instructions` と、任意の `model`、`model_reasoning_effort` を書く。モデルを省略すると親セッションから継承されるか、Codex がタスクに応じて選択する。

## Terra の例

```toml
name = "terra_researcher"
description = "調査、コード探索、大きなファイルの確認を高速に行う。"
model = "gpt-5.6-terra"
model_reasoning_effort = "xhigh"
sandbox_mode = "read-only"
developer_instructions = """
調査と根拠収集に集中し、簡潔な要約を親エージェントへ返す。
明示的に依頼されない限り、ファイルを変更しない。
"""
```

## Luna の例

```toml
name = "luna_worker"
description = "定型的な要約、軽量なコード変更、反復作業を低コストで行う。"
model = "gpt-5.6-luna"
model_reasoning_effort = "xhigh"
developer_instructions = """
範囲を限定した作業を実行し、変更内容と検証結果を親エージェントへ返す。
"""
```

## 全体設定

`~/.codex/config.toml` または信頼済みプロジェクトの `.codex/config.toml`:

```toml
[agents]
max_threads = 4
max_depth = 1

[features.multi_agent_v2]
hide_spawn_agent_metadata = false
tool_namespace = "agents"
```

利用時は「`terra_researcher` で調査し、`luna_worker` で実装して」のようにカスタムエージェント名を指定する。

GPT-5.6 Sol はモデルメタデータから Multi Agent V2 を選択する。Codex 0.144.1 では
`hide_spawn_agent_metadata` の既定値により `agent_type`、`model`、`reasoning_effort` が
`spawn_agent` の定義から隠れるため、上記設定で公開する。`features.multi_agent_v2` の
機能フラグ自体を明示的に有効化する必要はない。

## 確認事項

- 公式 subagent ドキュメントは `model` と `model_reasoning_effort` の直接指定を案内している。
- `gpt-5.6-terra` と `gpt-5.6-luna` は公式モデルID。
- 両モデルは `none`, `low`, `medium`, `high`, `xhigh`, `max` をサポートする。
- モデル利用可否はアカウントまたはワークスペースの権限に依存する。
- ローカル環境は `codex-cli 0.144.1`。
- 再起動後は `spawn_agent` に `agent_type` が現れることと、子セッションの
  `turn_context.model` が指定した Terra/Luna になることを確認する。
- 公式 Docs MCP `openaiDeveloperDocs` をグローバル登録した。現在のセッションへの反映には Codex の再起動または新規セッションが必要。

## 公式資料

- https://learn.chatgpt.com/docs/agent-configuration/subagents
- https://learn.chatgpt.com/docs/config-file/config-reference
- https://developers.openai.com/api/docs/models

# Codex custom subagent registration

- `~/.codex/config.toml` に `luna` と `terra` の agent role を登録した。
- role config は `agents/luna.toml` と `agents/terra.toml` を参照する。
- 同時スレッド数は 4、subagent の最大深度は 1 とした。
- `codex doctor --json` で設定の読み込みと全体状態が `ok` であることを確認した。
- 現在の会話には起動時のツール定義が残るため、利用開始には新しい Codex セッションが必要。

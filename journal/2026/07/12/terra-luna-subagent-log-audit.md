# terra / luna subagent ログ監査

## 結論

2026-07-12 の待機テストでは、`terra` と `luna` がそれぞれ指定どおり起動した。
これは子エージェントの自己申告ではなく、親ログの `spawn_agent` 呼び出し、子ログの `session_meta`、`turn_context`、`task_complete` を突き合わせて確認した。

## コピーと同一性確認

- 指定された luna ログを `/tmp/rollout-2026-07-12T18-48-35-019f55ba-86ed-77f3-81dc-734af5a8d8b9.analysis-copy.jsonl` にコピーした。
- 元ログとコピーの SHA-256 はともに `ca39eb37e56fc25146cb96049109981e6a60f681ade533f3b1faf34341b3a6ab` だった。
- terra の兄弟ログを `/tmp/rollout-2026-07-12T18-48-31-019f55ba-7904-74c1-b383-6b87715793be.analysis-copy.jsonl` にコピーした。
- 元ログとコピーの SHA-256 はともに `11f835e58b232ecc76ef1788ff37447fec7d4d3197d19b0828edf9b2982e2a31` だった。
- 親ログは分析時点のスナップショットとして `/tmp/rollout-2026-07-12T18-47-15-019f55b9-4dcc-7da2-86a6-6201b19e542f.analysis-snapshot.jsonl` にコピーした。

## 実行基盤側の証拠

### 親ログ

- 29 行目: `spawn_agent` に `agent_type: terra`, `fork_turns: none`, `task_name: terra_wait_test` を指定。
- 31 行目: `/root/terra_wait_test`、nickname `Lorentz` の起動応答。
- 33 行目: `spawn_agent` に `agent_type: luna`, `fork_turns: none`, `task_name: luna_wait_test` を指定。
- 35 行目: `/root/luna_wait_test`、nickname `Mencius` の起動応答。
- 23 行目の `terra` + `fork_turns: all` は実行基盤に拒否されており、実際の起動には数えない。

### terra 子ログ

- 1 行目 `session_meta`: `thread_source: subagent`, `agent_path: /root/terra_wait_test`, `agent_role: terra`, 親スレッドID `019f55b9-4dcc-7da2-86a6-6201b19e542f`。
- 8 行目 `turn_context`: `model: gpt-5.6-terra`, `effort: xhigh`。collaboration settings も同じ。
- 33 行目 `task_complete`: `duration_ms: 135060`。

### luna 子ログ

- 1 行目 `session_meta`: `thread_source: subagent`, `agent_path: /root/luna_wait_test`, `agent_role: luna`, 親スレッドID `019f55b9-4dcc-7da2-86a6-6201b19e542f`。
- 8 行目 `turn_context`: `model: gpt-5.6-luna`, `effort: xhigh`。collaboration settings も同じ。
- 25 行目 `task_complete`: `duration_ms: 129102`。

## 判断

親の起動指定、実行基盤が生成した子セッションの役割、実行モデル、推論強度、共通の親スレッドID、完了イベントが一貫している。そのため、terra と luna が実際にそれぞれのカスタムエージェント設定で呼び出されたと判断できる。

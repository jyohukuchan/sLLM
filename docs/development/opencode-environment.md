# opencode の実行環境

2026-09-23時点の設定。`opencode` v2.0.12、設定は `~/.config/opencode/opencode.jsonc`。
Codexの設定（`~/.codex/config.toml` と `.codex/config.toml`）と役割を揃えてあるが、
利用できるアカウントが異なるため既定モデルは同一ではない。Git管理外の設定なので、
変更したときはこの文書を更新する。

## モデルとアカウント

`opencode auth list` で使えるのはOpenCode Go、OpenCode（既定）、OpenRouter、DeepSeek（`DEEPSEEK_API_KEY`）。
このうち実際に応答するのは**OpenCode Go、DeepSeek、OpenCodeの無料枠（`-free`）だけ**である。

- `opencode/gpt-6-astra`（Codexの既定モデルと同じもの）は `Insufficient account funds` で失敗する。
- `openrouter/~openai/gpt-astra-latest` はcredit不足で失敗する。
- `opencode/gpt-5.6-terra`／`opencode/gpt-5.6-sol` も同じ理由で使えない。

現在の設定は次のとおり（2026-09-23更新）。

| 項目 | 値 |
| --- | --- |
| `model` | `opencode/mimo-v2.6-flash-free`（期間限定の無料枠） |
| `small_model` | `opencode/mimo-v2.6-flash-free` |
| subagent | `worker`（mode `subagent`、**model指定なし**＝呼び出したmain agentのモデルを継承） |
| `compaction` | `auto: true` |
| `tool_output` | `max_lines: 400`、`max_bytes: 20000`（既定は2000行／51,200 byte） |

無料枠を使うには`opencode/mimo-v2.6-flash-free`を指定する。`opencode-go/mimo-v2.6-flash`はGoプランの課金になる。
組み込みの`explore`／`general`もmodel指定がないため、main agentのモデルを継承する。
2026-09-22の実績では、`explore`はmainと同じモデルで動作し、孫subagentは0件だった。

v2.0.12では`subagent_depth`と`compaction.prune`が`unsupported`として無視されるため設定していない
（公開schemaには存在する）。subagentの入れ子は既定で1段に制限されている。

[AGENTS.md](../../AGENTS.md)のとおり、subagentは実行中のmain modelと同じproviderに従う。
OpenAI固定の `luna`／`terra`／`sol` は**定義していない**。`opencode` providerの有料モデルを使う場合にだけ、
Codexと同じ役割・モデルで再追加する。戻す先は `opencode/gpt-6-astra`（main）と
`opencode/gpt-5.6-luna`／`-terra`／`-sol`（subagent）である。

代替として動作を確認済みの funded モデルは `opencode-go/gpt-5.6-luna`、`deepseek/deepseek-v4-flash`。

## MCPサーバー

| 名前 | 種別 | 接続先 | Codex側 |
| --- | --- | --- | --- |
| `firecrawl` | local | `npx -y firecrawl-mcp@3.24.0`、`FIRECRAWL_API_KEY=local-dev`、`FIRECRAWL_API_URL=http://127.0.0.1:3002` | `~/.codex/config.toml` と同じ |
| `huggingface` | remote | `https://huggingface.co/mcp` | Codexでは `.codex/config.toml`（project単位） |
| `openai-docs` | remote | `https://developers.openai.com/mcp` | `~/.codex/config.toml` の `openaiDeveloperDocs` |

注意点。

- Firecrawlの**MCP終端は3000番ではない**。127.0.0.1:3000はFirecrawlのWebアプリで、`/mcp`へPOSTすると
  `Method Not Allowed` を返す。API本体は127.0.0.1:3002で、Codexと同じくlocalのnpxブリッジ経由で使う。
- `timeout` の単位は**ミリ秒**。既定5000で、リモート2件には60000、firecrawlには120000を設定している。
  秒のつもりで60と書くと接続前にタイムアウトする。
- 状態は `opencode mcp list` で確認する。3件ともconnectedであることを確認済み
  （firecrawl 27 tools、huggingface 4 tools、openai-docs 5 tools）。

## 権限

Codexの `approval_policy = "never"`／`sandbox_mode = "danger-full-access"` に合わせ、
`permission` の read／edit／glob／grep／list／bash／task／external_directory を `allow` にしている。
都度確認したい場合は該当項目を `ask` にする。CLIには `--auto` もある。

## 既知の不安定性（2026-09-22、MiMo V2.6 Flash）

長いセッションで2回停止した。どちらもファイルは壊れていない。

1. **HTTP 400**（15:06）: ツール出力を多く抱えた23メッセージ目で`provider.invalid-request`となった。
   直前の応答は空で、リクエストが大きくなりすぎた類の失敗と見られる。
2. **圧縮直後の反復ループ**（23:27〜23:41）: 自動compactionの直後の1応答に、ツール呼び出しが858個並んだ。
   中身は25種類しかなく、同じ`sed -n`範囲の読み出しが最大209回繰り返された。約13.5分生成して出力長上限
   （`finish: length`）で打ち切られ、858個はいずれも実行されていない。TUIには「writing command...」が残り、
   動いているように見えたが、セッションは23:41からidleだった。

確認方法: `~/.local/share/opencode/opencode.db`の`session_message`を見る。assistantメッセージの
`finish`（`length`、`error`等）、`time.completed`、toolパートの`state.status`（`streaming`のまま残る）で判別できる。
ログファイル`~/.local/share/opencode/log/opencode.log`にはモデル名が出ない。

対策として、`tool_output`の上限を下げて文脈の増加を抑えた。運用では次を守る。

- 1セッション＝1ステップに絞り、ステップごとに新しいセッションを始める。長いセッションを続けない。
- 指示文に「同じファイル範囲を読み直さない」「読んだ内容を要約してから編集へ進む」を書く。
- TUIの表示ではなくDBの最終更新時刻で進捗を判断する。

## 運用上の注意

- 実行はCodexより遅い。単純な質問でも数分かかることがあり、リポジトリを読ませる指示では
  10分で打ち切られた例がある。長い作業はバックグラウンドで実行し、完了を待つ。
- `opencode` はリポジトリの `AGENTS.md` を読む。`opencode run --auto "AGENTS.mdの最初の見出し行を出力して"`
  で `# AGENTS.md` を返すことを確認した。
- flash系モデルを使うときは、AGENTS.mdの既存規則（受入条件の事前固定、1作業単位＝1仮説、
  main agentによる編集内容の検査、commit前のformat・clippy・CI hash更新）を指示文へ明示的に書く。
  2026-09-22の段階7では、対象の取り違え・計測なし・不採用コードの残留が同時に起きた
  （[記録](../history/2026/09/21-30/phase87-stage7.md)）。

## 変更履歴

- 2026-09-22: 初版。MCP 3件を設定し接続を確認。既定モデルをMiMo V2.6 Flashにし、
  OpenAI固定のsubagentは定義しない構成とした。既存のfirecrawl設定（3000番のremote）の誤りを修正した。
- 2026-09-23: 無料枠の`opencode/mimo-v2.6-flash-free`へ切替。`worker`のmodel指定を外して継承にした。
  未対応の`subagent_depth`を削除し、`tool_output`の上限を下げた。既知の不安定性を追記した。

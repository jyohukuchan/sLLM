# reasoning・thinking・OpenWebUI互換拡張（完了）

## 目的

Phase 7完了後の残り時間で、OpenAI-compatible profile v1のstrictな既存動作を変えずに、
Qwen3.5のthinkingをopt-inでAPI公開し、`<think>` reasoningと最終回答をnon-stream/SSEで
分離する。OpenWebUIが送信するlegacy `max_tokens`は明示的な別profileだけで受理する。

## 受入条件

1. 既存`ServerConfigV1::new`はstrict profileのままで、`max_tokens`を
   `unsupported_parameter`として拒否する。
2. `--compatibility-profile openwebui`だけが`max_tokens`を`max_completion_tokens`と同じ
   1〜4,096の上限と意味で受理し、両方の同時指定を拒否する。
3. requestの`sllm.thinking=enabled|disabled`と`sllm.separate_reasoning`をclosed schemaとし、
   未指定とdisabledはPhase 6のthinking-disabled動作を維持する。
4. production Qwen backendはtyped requestのthinking modeを固定chat rendererへ渡す。
5. separation有効時はchunk境界をまたぐ`<think>`/`</think>`をstatefulに処理し、
   non-streamは`message.reasoning_content`/`message.content`、SSEは
   `delta.reasoning_content`/`delta.content`を返す。tag自体は返さない。assistant履歴の
   string `reasoning_content`は次のrequestへround-tripでき、system/user上では拒否する。
6. role先頭chunk、final usage、finish reason、`[DONE]`、post-header error、disconnect cancellationの
   strict回帰を維持する。
7. Rust unit/HTTP contract、fmt、clippy、H0と文書linkをPASSし、実GPUを使った場合は
   exact targetとcleanupをhistoryへ記録する。

## 非blocking follow-up

- OpenAI Responses APIやstandardのreasoning fieldへの追従は、pinned schemaを更新する別profileとする。
- reasoning token内訳のusage拡張は、tokenizer/accounting contractを定義するまで追加しない。

## 完了結果

- [x] strictとOpenWebUI互換profileをruntime config/CLIで分離した。
- [x] thinking control、reasoning/final separation、assistant履歴round-tripをtyped requestへ実装した。
- [x] non-stream/SSEのchunk境界、usage、finish、error、disconnect回帰をhost testでPASSした。
- [x] canonical R9700 `gfx1201`のproduction backendでnon-stream/SSEと2 requestをPASSした。
- [x] workspace test、fmt/clippy、H0 504/504、文書整合性をPASSした。

[対応するhistory](../../../../../history/2026/08/11-20/api-reasoning-openwebui-extension.md)

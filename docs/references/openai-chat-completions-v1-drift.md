# OpenAI Chat Completions profile v1仕様差分（Phase 6 A2）

観測日は2026-08-13。実装の正本は
[`docs/api/openai-compatibility.md`](../api/openai-compatibility.md)と、そこで固定したOpenAPI commit
`117ce5680e4269f6656a4fd70d28f9755630d938`である。current観測は正本を更新しない。

## 固定値と観測値

| 区分 | commit | OpenAPI SHA-256 | `info.version` |
| --- | --- | --- | --- |
| normative pin | `117ce5680e4269f6656a4fd70d28f9755630d938` | `e9cfcc3a325093a640af9e3b289dd4fa69f0c03e3a9af425fda47a5fe1238361` | `2.3.0` |
| current observation | `11854aef674352d3f9cd5c0a7038f079a7bbac06` | `63028c4d3916a53d9252c8b665b9f220b57051c8030b7eb23603048c75691bb2` | `2.3.0` |

currentは[OpenAI OpenAPI repository](https://github.com/openai/openai-openapi)の`main`を観測した。
公式API referenceは引き続き
[`POST /v1/chat/completions`](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)と
[`GET /v1/models`](https://developers.openai.com/api/reference/resources/models/methods/list)を定義する。

## request / response / stream / error差分

参照closureを再帰的に比較した。`POST /v1/chat/completions`は両側71 schema、`GET /v1/models`は
両側2 schemaだった。

| 面 | currentとの差分 | profile v1への影響 |
| --- | --- | --- |
| request | なし | なし |
| response | なし | なし |
| SSE stream | なし | なし。profileは`data: ...\n\n`とexact `[DONE]`を維持する |
| error | なし | なし |
| その他 | `ModelIdsShared` enumへ`gpt-5.5`追加 | なし。sLLMはserved aliasをmodel lockへ結合する |

OpenAIの[streaming guide](https://developers.openai.com/api/docs/guides/streaming-responses)はSSEのincremental
event処理を説明しているが、sLLMのwire contractは固定profileを正本とする。

## supported / rejected境界

supportedは`GET /v1/models`、`POST /v1/chat/completions`、system/user/assistantの文字列message、
temperature、top_p、max_completion_tokens、stop、presence/frequency penalty、stream、`n=1`である。

tools/function calling、multimodal、developer/tool role、logprobs、response_format、seed、`n != 1`はrejectする。
current schemaに存在することをsupportedと読み替えない。pin更新とResponses API追加は別decisionとする。

機械可読な観測値は
[`ci/contracts/phase6-a2-v1.json`](../../ci/contracts/phase6-a2-v1.json)に固定する。

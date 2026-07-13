# uLLM OpenAI Chat Completions Subset v0.2

Status: proposed implementation contract

This document extends the frozen v0.1 contract with model-independent reasoning
and thinking-budget fields. v0.1 parsing and behavior remain unchanged when a
v1 manifest or v1 worker is selected.

## 1. Request fields

The root request may contain exactly one of these fields:

| field | accepted values | meaning |
| --- | --- | --- |
| `reasoning_effort` | `none`, `low`, `medium`, `high` | selects a budget declared by the served-model dialect |
| `thinking_budget_tokens` | `-1` or an integer `0..max_budget_tokens` | selects an exact hard reasoning-body budget |

`reasoning_effort` and `thinking_budget_tokens` together are rejected with
`400 unsupported_parameter`. `none` disables reasoning. `-1` enables reasoning
without a hard budget. When both fields are omitted, the served-model default
is used; the initial AQ4 profile defaults to reasoning disabled.

The budget counts only reasoning-body token IDs. Natural or forced end-sequence
token IDs do not count toward the budget, but they do count toward generated
`completion_tokens`. The gateway MUST reject a request when the selected budget,
forced close sequence, and the dialect's minimum reserved answer tokens cannot
fit within `max_completion_tokens` and the model context. It MUST NOT silently
clamp the request.

## 2. Message history

Assistant messages may contain `reasoning_content` in addition to `content`.
The served-model dialect controls whether that field is omitted or preserved
when rendering the next prompt. The default policy is `omit`.

## 3. Response

Non-stream responses add `message.reasoning_content` when reasoning is enabled.
The ordinary answer remains in `message.content`.

`usage.completion_tokens` equals the number of all generated and committed token
IDs, including natural or forced delimiter tokens. The optional
`usage.completion_tokens_details.reasoning_tokens` counts reasoning-body token
IDs only.

Streaming order is:

1. assistant role delta;
2. zero or more `delta.reasoning_content` fields;
3. zero or more `delta.content` fields;
4. the finish chunk;
5. an optional usage chunk when requested;
6. `[DONE]`.

Delimiter token IDs and their decoded text MUST appear in neither output field.
The stream's field-wise concatenation MUST equal the corresponding non-stream
fields. `finish_reason` describes the final answer termination and MUST NOT be
`reasoning_budget` merely because a budget caused the phase transition.

## 4. Compatibility

Unknown v0.2 fields remain strict errors. A v1 manifest, v1 worker, or request
without reasoning fields continues to use the v0.1 path and its response shape.

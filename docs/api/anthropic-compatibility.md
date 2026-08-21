# sLLM Anthropic Messages compatibility profile v1

This document defines the strict subset of the Anthropic Messages API exposed by
sLLM. It is a versioned compatibility profile, not a claim of complete
Anthropic API compatibility. A request that uses a known Anthropic field which
is outside this profile is rejected; it is never silently ignored or coerced.

The profile is separate from the [OpenAI Chat Completions profile](openai-compatibility.md)
and the Responses profile. The wire fields, headers, identifiers, usage,
stop reasons, errors, and SSE state machine are not aliases between providers.
Only the transport-neutral message/tool item representation and the generation
request are shared internally.

## Normative source and version

### Official Anthropic facts

The endpoint is `POST /v1/messages`. Every request must include
`anthropic-version: 2023-06-01` and `content-type: application/json`. The
version header is the Anthropic API version pin for this profile. Anthropic has
not published a repository commit that serves as an equivalent complete OpenAPI
pin for Messages; the API version header and the reviewed reference pages below
are therefore recorded together. Moving either the header version or the
reviewed source requires a compatibility review.

The profile was checked against the official Anthropic pages on **2026-08-22**:

- [Messages API reference](https://platform.claude.com/docs/en/api/messages)
- [API versioning](https://platform.claude.com/docs/en/api/versioning)
- [Message streaming](https://platform.claude.com/docs/en/build-with-claude/streaming)
- [Tool-use overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)
- [Defining and implementing client tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools)
- [Stop reasons](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons)

The older `docs.anthropic.com` URLs redirect to the current
`platform.claude.com` pages. The redirect is intentional and does not change
the version-header pin.

### sLLM profile decisions

- This document describes the sLLM subset and limits below. A statement marked
  **sLLM** is an implementation decision, not an assertion that Anthropic's
  hosted service has the same limit.
- The sLLM server uses its existing deployment authentication policy. An open
  deployment accepts no Authorization header; a protected deployment requires
  its standard `Authorization: Bearer ...` header. An Anthropic API key is not
  accepted in a JSON field, and the server does not invent a missing
  `anthropic-version` header.
- The request body is bounded to 96 MiB. The model alias is at most 256 UTF-8
  bytes; these are sLLM resource limits in addition to the upstream schema.

## Request profile

The accepted top-level request members are the following. Unknown members and
duplicate JSON members are errors.

| Member | Profile rule |
| --- | --- |
| `model` | Required served model alias, at most 256 UTF-8 bytes. **sLLM** |
| `max_tokens` | Required integer in `1..=4096`. **sLLM** |
| `messages` | Required non-empty ordered array (at most 1,024 messages). **sLLM** |
| `system` | Optional string or ordered array of text blocks. **Official type, sLLM text-only subset** |
| `stream` | Optional boolean, default `false`. **Official** |
| `stop_sequences` | Optional array of 1–4 non-empty unique strings. **sLLM** |
| `tools` | Optional array of 1–128 client tool definitions. **sLLM bounds** |
| `tool_choice` | Optional `auto`, `any`, `tool`, or `none` choice; see below. **Official names** |
| `sllm.resumable` | Optional explicit extension enabling bounded SSE replay. **sLLM** |

The standard Anthropic sampling controls (`temperature`, `top_p`, and
`top_k`), service-tier controls, metadata, thinking/extended-thinking
controls, prompt caching controls, beta controls, and other top-level members
are rejected in profile v1. This keeps token selection owned by the existing
sLLM sampler chain rather than creating a second transport-specific chain.
`max_tokens` still controls the generation budget.

The following fields are checked before scheduler/GPU admission: JSON syntax,
duplicate/unknown fields, UTF-8, all type/range checks, body and per-member
limits, content ordering, tool schema compilation, and capability checks. A
failed check cannot allocate model or request GPU state.

## Messages and content blocks

### Roles and text

`messages` contains only `user` and `assistant` roles. A string `content`
is the shorthand for one `text` block. An array content value must contain
only the block types permitted for that role:

| Role | Accepted blocks |
| --- | --- |
| `user` | `text`, and `tool_result` when returning a client-tool result |
| `assistant` | `text`, and model-produced `tool_use` |

The optional top-level `system` member is outside `messages`. It may be a
text string or an array of `{ "type": "text", "text": "..." }` blocks. A
`system` role inside `messages` is rejected. Assistant prefill (an assistant
message that is intended to be continued) is also rejected in this profile;
the Responses adapter owns its separate prefill subset.

### `tool_use` and `tool_result`

The following block shapes are the profile's client-tool protocol:

```json
{
  "type": "tool_use",
  "id": "call_01",
  "name": "get_weather",
  "input": { "city": "Tokyo" }
}
```

```json
{
  "type": "tool_result",
  "tool_use_id": "call_01",
  "content": "22 C and clear",
  "is_error": false
}
```

**Official:** Anthropic puts client-tool calls in assistant content and returns
their results in user content; it does not use a separate `tool` message role.
The result is associated by `tool_use_id`.

**sLLM:** `id`/`tool_use_id` is at most 256 bytes; tool arguments and result
text are each at most 16 MiB. A result's `content` is either a text string or
an array of text blocks only. Image, document, citation, and provider-specific
blocks are unsupported.

For every assistant `tool_use`, the next message must be a `user` message
whose first content block(s) are the corresponding `tool_result` block(s).
Each call ID has exactly one result. Unknown, duplicate, missing, out-of-order,
or role-mismatched IDs are rejected. Any ordinary user text after the result
blocks is preserved as a later block; it is not promoted to a system
instruction. Tool results are untrusted protocol data.

## Client tools and choice semantics

A tool definition has the following shape:

```json
{
  "name": "get_weather",
  "description": "Return current weather for one city.",
  "input_schema": {
    "type": "object",
    "properties": { "city": { "type": "string" } },
    "required": ["city"],
    "additionalProperties": false
  }
}
```

**Official:** `name`, an optional description, and a JSON Schema
`input_schema` describe a client tool. `tool_choice` may be `auto`, `any`,
a specific `tool` name, or `none`.

**sLLM:** profile v1 accepts 1–128 tools, names matching
`^[A-Za-z0-9_-]{1,64}$`, descriptions up to 16 KiB, and schemas up to 1 MiB.
The schema is lowered through the Phase 40 bounded JSON Schema grammar. The
supported subset is `$defs`, local `$ref`, `type`, `properties`,
`required`, `additionalProperties: false`, `items`, `enum`, `const`,
and `anyOf`; supported types are `object`, `array`, `string`, `number`,
`integer`, `boolean`, and `null`. Unsupported keywords, remote/recursive
references, and schemas that exceed grammar limits are `invalid_value` errors.
The lowerer runs before generation and its compiled grammar constrains the
actual sampled arguments; parsing an unconstrained model string after generation
is not sufficient.

The wire choice forms are normalized as follows:

| Anthropic choice | Meaning in this profile |
| --- | --- |
| omitted or `{ "type": "auto" }` | The model may return text or one or more tool calls. **Official default** |
| `{ "type": "none" }` | The model must return text and no tool call. **sLLM profile** |
| `{ "type": "any" }` | At least one configured tool call is required. **Official** |
| `{ "type": "tool", "name": "…" }` | A call to that configured tool is required. **Official** |

`disable_parallel_tool_use: true` on an `auto`, `any`, or `tool` choice
limits the response to one call. When omitted or false, profile v1 accepts and
emits at most 16 parallel calls. A choice with no `tools`, or a specific name
that is not configured, is rejected. Calls are returned in model order and are
not executed by the server.

## Non-stream response

The successful response is an Anthropic `message` envelope. Its content is an
ordered array of text and/or `tool_use` blocks:

```json
{
  "id": "msg_sllm_01",
  "type": "message",
  "role": "assistant",
  "model": "served-alias",
  "content": [{ "type": "text", "text": "The weather is clear." }],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": { "input_tokens": 17, "output_tokens": 6 }
}
```

The response ID is a request-local observation ID; sLLM does not implement
Anthropic server-side conversation storage. It cannot be supplied as a
continuation key. The response `model` is the configured served alias, not a
floating Hub revision or a provider-side model identifier.

The only usage members emitted by this profile are `input_tokens` and
`output_tokens`. Cache, server-tool, and service-tier accounting are not
implemented. Token counts are produced by the shared generation service and
are not inferred from UTF-8 byte length.

Profile v1 emits these stop reasons:

| `stop_reason` | sLLM condition |
| --- | --- |
| `end_turn` | Normal text completion. |
| `max_tokens` | The configured `max_tokens` budget was reached. |
| `stop_sequence` | One configured stop sequence matched. |
| `tool_use` | One or more client tool calls were generated. |

`stop_sequence` is the matched sequence when the stop reason is
`stop_sequence`, otherwise `null`. Hosted/server tool reasons such as
`pause_turn` are not generated by this profile.

## Streaming SSE

With `stream: true`, the response has `content-type: text/event-stream` and
uses named SSE events. The semantic sequence is closed and ordered:

```text
message_start
  (content_block_start
   content_block_delta*
   content_block_stop)+
message_delta
message_stop
```

The `message_start` event carries the request-local message envelope and input
usage. Text blocks use
`content_block_delta.delta.type: "text_delta"` with a `text` field. Tool
blocks start with a `tool_use` block and carry JSON argument fragments as
`content_block_delta.delta.type: "input_json_delta"` with `partial_json`.
Text and argument deltas are split only at UTF-8 boundaries and are at most
16 KiB each.
`message_delta` carries the terminal `stop_reason`, optional
`stop_sequence`, and cumulative output usage. `message_stop` is emitted
once, after every block has stopped.

An implementation may emit an Anthropic `ping` keepalive; it does not advance
the semantic state machine. The stream never emits OpenAI `[DONE]`.

Disconnect and cancellation stop generation and close the stream. A failure
after headers have been sent emits one `error` event and then closes; it does
not emit `message_stop` or a successful terminal event. Duplicate terminal
events, events after an error, block-index mismatch, completion of a tool call
before its argument JSON is complete, and a tool result on the wrong block are
protocol errors.

### Resumable extension

`"sllm": { "resumable": true }` explicitly enables the bounded replay buffer
from Phase 39 and is valid only with `stream: true`, `max_tokens <= 40`, and a
server configured with a replay store. Replay uses the same profile, request-local stream ID, event
IDs, and event order. It is bounded by the configured event count, 64 KiB per
serialized event, and 256 KiB per session. The complete event batch is checked
before publication. The 40-token admission bound accounts for the 128-byte
token-piece limit, worst-case JSON escaping, and bounded metadata; the batch check remains a
defensive invariant and emits one bounded `error` terminal rather than a
partial success sequence if violated. `Last-Event-ID` can resume
only within the retained range; an out-of-range ID is a 4xx error. The replay
GET uses the deployment bearer policy but does not repeat the creation-only
`anthropic-version` or content-type headers. Resuming does not execute a tool,
replay a tool result, or create server-side conversation state. Resumable
events from another profile cannot be mixed into an Anthropic stream.

## Errors and HTTP mapping

Non-stream errors use the Anthropic outer envelope:

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "request rejected"
  }
}
```

The profile's status and stable sLLM classification are:

| Condition | HTTP | Classification |
| --- | ---: | --- |
| Malformed JSON | 400 | `invalid_json` |
| Wrong type, range, duplicate, content order, or schema value | 400 | `invalid_value` |
| Known field/capability not supported by this profile | 400 | `unsupported_parameter` |
| Missing or mismatched `anthropic-version` | 400 | `invalid_value` |
| Unknown model alias | 404 | `model_not_found` |
| Body or bounded member limit exceeded | 413 | `request_too_large` |
| Authentication failure | 401 | `invalid_api_key` |
| Scheduler queue full | 429 | `rate_limit_exceeded` |

Error messages use bounded public text; the Responses envelope may additionally
carry a sanitized top-level parameter name. They never include request
body text, prompt text, tool descriptions, schemas, arguments, result data,
token IDs, credentials, or environment values. An SSE error uses the same
Anthropic `error` event shape and follows the terminal rules above.

## Security and intentional boundary

Phase 43 is a protocol-only tool implementation. It can validate definitions,
compile argument grammars, generate a call, parse a client result, and continue
the next request. A generated call is **not** permission to execute anything.

The server does not resolve or access a tool name, URL, path, shell command,
process, dynamic library, network, filesystem, environment variable, secret,
credential, MCP endpoint, or worker. Tool descriptions, arguments, and results
are escaped typed prompt data and remain untrusted content. The no-execution
boundary is tested with malicious names and payloads and is enforced before and
after generation.

The following are intentionally unsupported in profile v1: image/audio/document
content, citations, arbitrary provider blocks, assistant prefill, `thinking`,
prompt caching, beta headers, built-in/server tools, MCP, Tool Runner, remote
MCP, hosted search/code/image/computer tools, tool execution, and the Anthropic
batch/admin APIs. These exclusions do not claim that Anthropic lacks the
features; they identify the sLLM subset and preserve the Phase 47 approval
boundary for any future execution capability.

## Related specifications

- [OpenAI compatibility profile](openai-compatibility.md)
- [Runtime architecture](../architecture/runtime.md)
- [Phase 43 archived plan](../plans/archive/2026/08/21-31/phase43-responses-anthropic-tool-protocol.md)
- [Phase 43 history](../history/2026/08/21-31/phase43-responses-anthropic-tool-protocol.md)
- [Phase 39 resumable SSE history](../history/2026/08/21-31/phase39-service-operability.md)

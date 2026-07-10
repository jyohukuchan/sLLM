# uLLM OpenAI Chat Completions Subset v0.1

Status: frozen for P8 implementation

Date: 2026-07-10

This document is the normative HTTP contract between the v0.1 uLLM gateway and
OpenWebUI. It defines a deliberately small OpenAI-compatible subset. OpenAI API
behavior not stated here is not implemented.

The words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative. JSON
member names and string values are case-sensitive unless stated otherwise.

## 1. Fixed Product Identity and Limits

| item | fixed value |
| --- | --- |
| public model ID | `ullm-qwen3-14b-sq8` |
| model | `Qwen/Qwen3-14B-FP8` |
| context length | 4096 tokens |
| default maximum completion | 256 tokens |
| hard maximum completion | 512 tokens |
| active requests | 1 |
| waiting requests | 0 |
| request body limit | 2 MiB (2,097,152 bytes) |
| SSE client queue | 32 token events |
| gateway processes | one Uvicorn process, `--workers 1`, reload disabled |

The API is stateless. Every completion request MUST contain the complete chat
history to use. There is no request-to-request KV reuse, automatic history
truncation, waiting queue, request batching, tool calling, or multimodal input.

## 2. HTTP Surface

| method and path | authentication | result |
| --- | --- | --- |
| `GET /healthz` | none | gateway process liveness |
| `GET /readyz` | none | resident model and worker health |
| `GET /v1/models` | Bearer key | the one fixed model |
| `POST /v1/chat/completions` | Bearer key | non-stream or SSE chat completion |

Query parameters are not defined for these endpoints. A request containing any
query parameter MUST be rejected with `400 invalid_request_error`.

`POST /v1/chat/completions` MUST use `Content-Type: application/json`; a media
type parameter such as `charset=utf-8` is allowed. The body MUST be valid UTF-8
JSON and no larger than the fixed body limit. Invalid media type, invalid UTF-8,
invalid JSON, a non-object root, duplicate JSON object keys, `NaN`, or infinity
MUST return `400 invalid_request_error`. The gateway MUST reject an oversized
body while reading it and MUST NOT buffer more than the configured limit plus
the minimum framing required to detect overflow.

Successful JSON responses use `Content-Type: application/json`. JSON output
MUST be valid UTF-8 and MUST NOT contain non-finite numbers.

### 2.1 Authentication

Both `/v1/models` and `/v1/chat/completions` require the same configured API
key. `/healthz` and `/readyz` are deliberately unauthenticated and MUST NOT
return the key, filesystem paths, prompt text, model source paths, or internal
error details.

The gateway MUST:

- accept the key only from one `Authorization: Bearer TOKEN` header;
- treat the `Bearer` scheme case-insensitively and the token bytes exactly;
- reject a missing, empty, duplicated, malformed, or incorrect header with
  `401 invalid_api_key`;
- return `WWW-Authenticate: Bearer` on that 401 response;
- compare the token without a data-dependent early exit;
- never accept the key from a query parameter, body field, command line, or
  cookie;
- never log or include the token in fixtures or errors.

Authentication is checked before parsing the chat request body. Therefore an
unauthenticated malformed request returns 401, not a schema error.

The service MUST fail startup if the configured key file is absent, unreadable,
empty after removal of one terminal `LF` or `CRLF`, or contains another line.

## 3. Health and Model Discovery

### 3.1 Liveness

While the gateway can serve HTTP, `GET /healthz` returns 200:

```json
{"status":"ok"}
```

Liveness does not assert that the model is loaded.

### 3.2 Readiness

`GET /readyz` returns 200 while the worker and model are healthy:

```json
{"status":"ready"}
```

It MUST remain 200 during healthy `Prefilling`, `Decoding`, `Finishing`, and
`Resetting` states. Busy is not unready.

During `Loading`, `Restarting`, `Failed`, or `Shutdown`, it returns 503:

```json
{"status":"not_ready"}
```

The readiness response intentionally does not expose whether the single request
slot is occupied.

### 3.3 Model List

After successful authentication, `GET /v1/models` returns 200 with exactly:

```json
{
  "object": "list",
  "data": [
    {
      "id": "ullm-qwen3-14b-sq8",
      "object": "model",
      "owned_by": "ullm"
    }
  ]
}
```

This endpoint MUST NOT trigger model loading or mutate worker state. It returns
the configured product identity even while `/readyz` is 503.

## 4. Chat Completion Request

The root request is a JSON object. `model` and `messages` are required. The
complete allowed field set is defined below.

### 4.1 Root Fields

| field | accepted value | omission or `null` behavior |
| --- | --- | --- |
| `model` | exact fixed model ID string | required; `null` is invalid |
| `messages` | non-empty message array | required; `null` is invalid |
| `stream` | boolean | defaults to `false` |
| `stream_options` | object described below | defaults to no usage chunk |
| `max_tokens` | integer 1 through 512 | omitted |
| `max_completion_tokens` | integer 1 through 512 | omitted |
| `temperature` | finite number from 0 through 2 | defaults to `0.6` |
| `top_p` | finite number greater than 0 and at most 1 | defaults to `0.95` |
| `seed` | signed 64-bit integer | gateway selects an OS-random seed |
| `n` | integer exactly equal to 1 | defaults to 1 |
| `stop` | empty string or empty array | no stop-string matching |
| `frequency_penalty` | finite number exactly equal to 0 | defaults to 0 |
| `presence_penalty` | finite number exactly equal to 0 | defaults to 0 |
| `logit_bias` | empty object | no bias |
| `logprobs` | boolean `false` | no log probabilities |
| `top_logprobs` | no non-null value | omitted |
| `user` | string | accepted metadata; does not affect inference |

JSON booleans are not integers or numbers for validation. For example,
`"seed": true` and `"temperature": false` are invalid.

For every optional field in this table, JSON `null` has the same effect as
omission. This rule does not apply to required fields. When both maximum-token
fields are non-null, the request MUST be rejected with
`400 unsupported_parameter`, even if the values are equal. When both are null
or omitted, the effective maximum is 256.

`stream_options` permits only this member:

| field | accepted value | omission or `null` behavior |
| --- | --- | --- |
| `include_usage` | boolean | defaults to `false` |

`stream_options.include_usage=true` requires `stream=true`; otherwise the
request is `400 invalid_request_error`. An empty object or an object whose
effective `include_usage` is false is accepted with `stream=false`. Unknown
members follow the unknown-field policy in section 4.5.

The effective inference parameters are:

```json
{
  "temperature": 0.6,
  "top_p": 0.95,
  "top_k": 20,
  "max_completion_tokens": 256,
  "seed": "OS-random signed 64-bit integer",
  "eos_token_ids": [151645, 151643]
}
```

`top_k` and EOS IDs are product configuration, not request fields. Any non-null
request `top_k` is unsupported. `temperature=0` selects greedy sampling; in that
case `top_p`, `top_k`, and `seed` remain recorded effective values but do not
change the selected token.

The selected effective seed MUST be included in structured internal request
metadata for reproducibility, but MUST NOT expose the API key or raw message
text. `user` MUST NOT be sent to the Rust worker and MUST NOT be logged in clear
text.

### 4.2 Messages

Each `messages` element is an object with exactly these effective members:

| field | rule |
| --- | --- |
| `role` | required string: `system`, `user`, or `assistant` |
| `content` | required string or non-empty array of text parts |

The allowed role grammar is:

```text
[system]? user (assistant user)*
```

The optional `system` message can occur only at index 0. After it, roles
alternate starting with `user` and ending with `user`. A system-only request,
assistant-first request, repeated role, second system message, or request ending
in assistant is `400 invalid_request_error`.

String content is used exactly as received. An array content value is normalized
by concatenating its parts in array order with no inserted separator. Each part
MUST be an object of this form:

```json
{"type":"text","text":"content"}
```

`type` and `text` are required and non-null. `type` MUST equal `text`, and
`text` MUST be a string. Image, image URL, audio, input audio, file, refusal,
tool-call, and any other content part is `400 unsupported_parameter`.

Empty strings are allowed in message content. The resulting templated prompt,
not the source character count, is subject to the minimum one-token rule.

### 4.3 Chat Template and Token Accounting

The gateway MUST load the tokenizer and template only from the frozen local
Qwen3 tokenizer directory. It MUST apply the complete normalized message list
with:

```text
add_generation_prompt=true
enable_thinking=false
```

It MUST NOT inject a system message, remove a message, trim content, summarize
history, or add application-specific separators. Special tokens come only from
the frozen tokenizer/chat-template behavior.

`prompt_tokens` is the exact number of token IDs produced by that template.
After request normalization:

```text
1 <= prompt_tokens
1 <= effective_max_completion_tokens <= 512
prompt_tokens + effective_max_completion_tokens <= 4096
```

Violation of the final inequality returns `400 context_length_exceeded`. There
is no automatic truncation. The gateway MUST complete this check before it sends
`generate` or otherwise mutates worker/GPU state.

`completion_tokens` counts sampled token IDs reported by the worker, including
an EOS token even when special-token removal makes it contribute no visible
text. `total_tokens` is the integer sum of prompt and completion counts.

### 4.4 Explicitly Unsupported Fields

The following non-null root fields are recognized but unsupported in v0.1 and
MUST return `400 unsupported_parameter`:

- `top_k`;
- `tools`, `tool_choice`, `parallel_tool_calls`;
- `functions`, `function_call`;
- `response_format` and all JSON schema or structured-output controls;
- `modalities`, `audio`;
- `reasoning_effort` and any request control that enables thinking;
- `store`, `metadata`, `service_tier`;
- `prediction`;
- non-null `top_logprobs`;
- `logprobs=true`;
- non-empty `stop` string or array;
- non-empty `logit_bias`;
- `n` other than 1;
- nonzero `frequency_penalty` or `presence_penalty`.

An empty `tools` array or empty `response_format` object is still non-null and
therefore unsupported. Unsupported behavior MUST NOT be silently ignored or
reported as implemented.

### 4.5 Null and Unknown-Field Policy

At the request root, message object, content-part object, and `stream_options`
object:

- a listed required member cannot be null;
- a listed optional member handles null as defined above;
- an unknown member whose value is null is ignored;
- an unknown member whose value is not null returns
  `400 unsupported_parameter` with that member path in `error.param`.

This policy permits clients that serialize unused optional values as null while
preventing a misspelled or silently unsupported active control. Duplicate keys
are always invalid, including duplicate unknown-null keys.

Members such as `name`, `tool_calls`, `tool_call_id`, `function_call`, and
`refusal` are not part of a v0.1 message. Their non-null presence is
unsupported.

### 4.6 Validation and Admission Order

The gateway MUST use this externally observable order:

1. match method and path;
2. authenticate protected endpoints;
3. validate query absence, content type, body size, UTF-8, and JSON framing;
4. validate root field types and the requested model ID;
5. validate parameters, message members, content parts, and role order;
6. apply the frozen chat template and validate token/context limits;
7. check worker/model readiness;
8. atomically acquire the one active-request slot while rechecking readiness;
9. send exactly one worker `generate` command.

Consequences of this order are normative:

- an unknown model returns 404 even if the worker is loading;
- an invalid/context-overflow request returns 400 even while another request is
  active;
- a valid request returns 503 if the worker is not ready;
- a valid, ready request returns 429 immediately if the slot is occupied;
- no validation failure creates waiting work or mutates GPU state.

The active slot has no queue. A collision returns `Retry-After: 1`; the gateway
MUST NOT retain, retry, or later admit that request. Once acquired, the slot
remains owned until the matching worker `released(reset_complete=true)` event
or until the gateway process exits after a fatal failure. HTTP transport close
alone does not release it.

For a streaming request, status and SSE headers MUST NOT be committed until the
request is admitted and the matching worker `started` event has been validated.
A failure before that point uses a normal JSON error response.

## 5. Successful Non-Streaming Response

For `stream=false`, status is 200 and the response has this exact member shape:

```json
{
  "id": "chatcmpl-0123456789abcdef0123456789abcdef",
  "object": "chat.completion",
  "created": 1783641600,
  "model": "ullm-qwen3-14b-sq8",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "generated text"
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 17,
    "total_tokens": 59
  }
}
```

Rules:

- `id` is `chatcmpl-` followed by 32 lowercase hexadecimal characters and is
  unique for the gateway process lifetime;
- `created` is an integer Unix timestamp in seconds, chosen once at admission;
- `model` is always the fixed public model ID;
- `choices` contains exactly one item with `index=0`;
- assistant `content` is the complete stable detokenized text and may be empty;
- `logprobs` is always null;
- no tool, refusal, reasoning, or system-fingerprint fields are emitted;
- usage uses the accounting in section 4.3.

`finish_reason` is `stop` when a configured EOS ID ends generation and `length`
when the effective completion limit is reached. A cancellation or worker failure
does not produce a successful non-stream response.

## 6. Successful Streaming Response

For `stream=true`, status is 200. Required response headers are:

```text
Content-Type: text/event-stream
Cache-Control: no-cache
X-Accel-Buffering: no
```

The `Content-Type` MAY include `charset=utf-8`. HTTP transfer framing is chosen
by the server. The stream MUST NOT be compressed or buffered by the gateway.

Each SSE record contains one UTF-8 `data:` line followed by a blank line. JSON
records use compact JSON, and no `event:`, `id:`, or `retry:` SSE fields are
emitted:

```text
data: {JSON}\n\n
```

All completion chunks share the same `id`, `created`, `model`, and `object`.
`id` and `created` follow the non-stream rules. The sequence is exact.

### 6.1 Role Chunk

Exactly one role chunk is first:

```json
{
  "id": "chatcmpl-0123456789abcdef0123456789abcdef",
  "object": "chat.completion.chunk",
  "created": 1783641600,
  "model": "ullm-qwen3-14b-sq8",
  "choices": [
    {
      "index": 0,
      "delta": {"role": "assistant", "content": ""},
      "logprobs": null,
      "finish_reason": null
    }
  ]
}
```

It is emitted immediately after the validated worker `started` event and before
any content chunk.

### 6.2 Content Chunks

Zero or more content chunks follow:

```json
{
  "id": "chatcmpl-0123456789abcdef0123456789abcdef",
  "object": "chat.completion.chunk",
  "created": 1783641600,
  "model": "ullm-qwen3-14b-sq8",
  "choices": [
    {
      "index": 0,
      "delta": {"content": "stable suffix"},
      "logprobs": null,
      "finish_reason": null
    }
  ]
}
```

Each content string MUST be non-empty. Concatenating all content delta strings
in order MUST exactly equal the non-stream `message.content` produced from the
same worker token sequence and seed.

Incremental decoding uses `skip_special_tokens=true` and disables cleanup. The
gateway MUST emit only a stable decoded suffix. It MUST NOT emit a Unicode
replacement character merely because a token sequence is incomplete, repeat a
prefix, omit a final stable suffix, or split invalid UTF-8 bytes onto the wire.

### 6.3 Final Choice Chunk

After the worker's matching successful `released` event, exactly one final
choice chunk is emitted:

```json
{
  "id": "chatcmpl-0123456789abcdef0123456789abcdef",
  "object": "chat.completion.chunk",
  "created": 1783641600,
  "model": "ullm-qwen3-14b-sq8",
  "choices": [
    {
      "index": 0,
      "delta": {},
      "logprobs": null,
      "finish_reason": "stop"
    }
  ]
}
```

The finish reason is `stop` or `length` under the same rules as non-streaming.
It appears in no earlier chunk.

### 6.4 Optional Usage Chunk

When `stream_options.include_usage=true`, exactly one usage chunk follows the
final choice chunk:

```json
{
  "id": "chatcmpl-0123456789abcdef0123456789abcdef",
  "object": "chat.completion.chunk",
  "created": 1783641600,
  "model": "ullm-qwen3-14b-sq8",
  "choices": [],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 17,
    "total_tokens": 59
  }
}
```

When usage was not requested, no chunk contains a `usage` member.

### 6.5 Terminal Marker

After the final choice and optional usage chunk, the gateway emits exactly:

```text
data: [DONE]\n\n
```

`[DONE]` occurs once, is the last SSE record, and is followed by transport close.
SSE comments and heartbeat records are not part of v0.1.

## 7. Errors

### 7.1 JSON Error Envelope

Every error sent before streaming headers has this exact member shape:

```json
{
  "error": {
    "message": "human-readable stable message",
    "type": "invalid_request_error",
    "param": "messages[0].content",
    "code": "invalid_request_error"
  }
}
```

`message`, `type`, and `code` are strings. `param` is a dotted/indexed request
path or null. Errors MUST NOT contain prompts, generated text, API keys,
filesystem paths, backtraces, raw worker messages, or HIP addresses.

| condition | HTTP | `type` | `code` | `param` |
| --- | ---: | --- | --- | --- |
| malformed body, schema, or role order | 400 | `invalid_request_error` | `invalid_request_error` | offending path or null |
| context reservation exceeds 4096 | 400 | `invalid_request_error` | `context_length_exceeded` | `messages` |
| unsupported non-null/non-neutral parameter | 400 | `invalid_request_error` | `unsupported_parameter` | field path |
| missing or invalid API key | 401 | `invalid_request_error` | `invalid_api_key` | null |
| unknown model | 404 | `invalid_request_error` | `model_not_found` | `model` |
| active slot occupied | 429 | `rate_limit_error` | `request_busy` | null |
| worker/model not ready | 503 | `server_error` | `model_not_ready` | null |
| unexpected internal failure before headers | 500 | `server_error` | `internal_error` | null |

The 429 response MUST include `Retry-After: 1`. It is advice to the client, not a
gateway retry or queued reservation.

Framework-native 422 response bodies are forbidden on these endpoints. Request
validation failures are mapped to the 400 envelope above.

### 7.2 Failure After Streaming Headers

When an internal or worker failure occurs after SSE headers are committed and
the transport remains writable, the gateway attempts exactly one record:

```text
data: {"error":{"message":"The generation failed.","type":"server_error","param":null,"code":"internal_error"}}\n\n
```

It then closes the transport. It MUST NOT emit a final choice chunk, usage
chunk, or `[DONE]`. This error record is not a successful completion chunk.

On a watchdog breach, delivery is best effort and bounded to 250 ms. A blocked
or disconnected client MUST NOT delay transport close, worker termination, or
gateway exit; consequently the record can be absent on an unusable transport.

A client disconnect emits no further bytes. The gateway still cancels the
worker and drains worker stdout and stderr until matching release or process
exit.

## 8. Request Lifecycle, Cancellation, and Backpressure

The gateway owns one atomic active slot and no waiting storage. Worker stdout
and stderr pumps MUST never await the client-facing queue or socket.

The successful lifecycle is:

```text
validated -> admitted -> worker started -> token/progress events
          -> worker released(reset_complete=true) -> HTTP terminal output
          -> active slot free
```

The gateway MUST NOT clear the slot on EOS detection, maximum-token detection,
client disconnect, cancel transmission, HTTP close, or a worker `error`. Only a
matching `released(reset_complete=true)` permits reuse. Fatal worker failure is
resolved by gateway process exit and systemd restart, not by local slot reuse.

### 8.1 Cancellation

Client disconnect and the OpenWebUI Stop action both cause a matching worker
`cancel` command when generation is active. After cancellation:

- new token events are not sent to the client;
- both worker pipes continue to be drained;
- the gateway waits for `released(outcome="cancelled", reset_complete=true)`;
- a writable stream closes without a finish chunk, usage chunk, or `[DONE]`;
- the next request is admitted only after release.

There is no false `finish_reason="stop"` for cancellation.

### 8.2 Slow Client

The client queue contains at most 32 token events. The first failed nonblocking
enqueue because that queue is full MUST immediately request cancellation with a
stable slow-client reason. Later token events are discarded while worker stdout
and stderr continue to drain. After terminal release, the gateway closes the
HTTP stream without a final choice, usage chunk, error record, or `[DONE]`.

No five-second grace period is allowed before initiating slow-client
cancellation. The five-second deadline in section 8.3 applies after any cancel
command has been sent.

### 8.3 Hard Deadlines and Fatal Ordering

| deadline | value | start and reset rule |
| --- | ---: | --- |
| worker Ready | 600 s | worker process spawn to validated `ready` |
| complete request | 180 s | active-slot acquisition to matching `released` |
| protocol progress | 30 s | generate send; reset by matching `started`, `progress`, `token`, or `released` |
| cancel to release | 5 s | matching cancel send to matching `released` |

Any deadline breach, unexpected worker EOF, protocol corruption, or fatal
worker error is fatal to the gateway. It MUST perform this bounded ordering:

1. make `/readyz` return 503;
2. attempt the post-header error record without blocking for more than 250 ms;
3. close the active client transport;
4. terminate the worker, allowing at most 2 seconds before kill;
5. exit the gateway nonzero within 5 seconds of detecting the fatal condition.

Client backpressure cannot extend any step. A partially streamed request is
never retried. Recovery belongs to systemd restarting the complete control
group.

## 9. Fixed OpenWebUI Deployment Topology

The release topology is part of this API contract:

- the gateway is a systemd system service running as `homelab1`;
- OpenWebUI is attached to Docker network `open-webui-network`;
- the observed v0.1 subnet is `172.20.0.0/16` and host gateway is
  `172.20.0.1`;
- the gateway binds only `172.20.0.1:8000`;
- OpenWebUI Base URL is `http://172.20.0.1:8000/v1`;
- the OpenWebUI connection supplies the mandatory Bearer key;
- title, follow-up, and tag background generation are disabled or use another
  model while active-request capacity is one.

Before enabling the service, deployment MUST revalidate the Docker network ID,
subnet, gateway, bridge interface, host route, and firewall rule. A mismatch is
a deployment failure. The service MUST NOT compensate by binding to
`0.0.0.0`, a LAN address, or every interface. v0.1 does not provide TLS and
relies on the restricted local bridge topology.

Forwarded headers MUST NOT be trusted to widen authentication or network scope.

## 10. OpenWebUI Interoperability Fixtures

P8-A MUST capture the actual upstream HTTP emitted by the installed OpenWebUI,
not a hand-written approximation and not only the browser-to-OpenWebUI request.
Capture occurs at a recording endpoint reached through the fixed Docker network.

The fixture set MUST contain at least:

```text
tests/fixtures/sq8-serving-v0.1/openwebui/
  manifest.json
  models-request.json
  chat-nonstream-request.json
  chat-stream-request.json
  chat-stream-response.sse
```

`manifest.json` records:

- OpenWebUI version and immutable image digest;
- Docker network ID, subnet, gateway, and recording endpoint address;
- capture time in UTC;
- the exact Base URL excluding credentials;
- SHA-256 for every fixture file;
- whether title, follow-up, and tag generation were disabled;
- the capture tool/source commit and redaction procedure.

Each request fixture records method, path, ordered header names with secret
values redacted, raw body SHA-256 after deterministic redaction, and the parsed
JSON value. It MUST preserve the observed root field set, nulls, types, message
content representation, and `stream_options` behavior. It specifically records
whether OpenWebUI emitted:

- `max_tokens`, `max_completion_tokens`, or both;
- `temperature`, `top_p`, `seed`, `n`, and `stop`;
- penalties, `logit_bias`, or log-probability fields;
- `stream_options.include_usage`;
- `user` or any unexpected metadata/control field;
- string message content or `type=text` content parts.

The non-stream and stream captures MUST each result from an actual OpenWebUI
provider call. They MUST use fixed test messages after redaction and MUST NOT
contain real conversation text, cookies, API keys, authorization tokens, user
identifiers, or host filesystem paths. The authorization header is represented
only as `Bearer <REDACTED>`.

Redaction may replace string values and update dependent byte lengths and hashes;
it MUST NOT add, delete, reorder, or change the type of JSON members. The final
redacted fixture, rather than secret raw capture, is the committed and hashed
artifact.

If the actual OpenWebUI payload contains a non-null field outside the subset in
section 4, P8-A does not silently relax the gateway. The field is reviewed and
this specification is deliberately revised, or the OpenWebUI setting is changed
and the fixture recaptured. Runtime-specific workarounds are forbidden.

## 11. Minimum Conformance Matrix

Contract tests MUST cover at least:

| case | required result |
| --- | --- |
| missing/wrong Bearer key | 401 and `WWW-Authenticate: Bearer` |
| fixed model list | exact one-model response |
| unknown model | 404 `model_not_found` |
| string content | accepted |
| text-part content | accepted and concatenated in order |
| image/audio/file part | 400 `unsupported_parameter` |
| valid system plus user | accepted |
| two-turn history ending in user | accepted |
| repeated role or assistant-final history | 400 `invalid_request_error` |
| unknown null root/nested field | ignored |
| unknown non-null root/nested field | 400 `unsupported_parameter` |
| both non-null maximum-token fields | 400 `unsupported_parameter` |
| default maximum | 256 |
| maximum 512 | accepted when context reservation fits |
| maximum 0 or 513 | 400 `invalid_request_error` |
| exact context reservation 4096 | accepted |
| reservation 4097 | 400 `context_length_exceeded`, no worker mutation |
| `temperature=0` | greedy |
| invalid/non-finite sampling value | 400 `invalid_request_error` |
| neutral penalties and empty bias | accepted |
| non-neutral penalty/bias/stop/tool | 400 `unsupported_parameter` |
| `stream=false` | exact non-stream shape and usage |
| stream without usage | role, content, finish, `[DONE]` |
| stream with usage | finish, usage, `[DONE]` ordering |
| EOS | finish reason `stop` exactly once |
| token limit | finish reason `length` exactly once |
| second valid concurrent request | immediate 429 and `Retry-After: 1` |
| healthy active request | `/readyz` remains 200 |
| disconnect before/mid generation | cancel, release, no terminal SSE success |
| queue event 33 while full | immediate cancel and bounded memory |
| worker failure before headers | JSON error, gateway fatal exit |
| worker failure after headers | one best-effort error event, no `[DONE]`, fatal exit |
| startup/request/progress/cancel hang | fixed watchdog and fatal ordering |
| Japanese, emoji, combining text, code fence | stream concatenation equals non-stream text |
| actual OpenWebUI stream/non-stream fixtures | accepted without hidden coercion |

Tests that reach worker/GPU state run serially. Pure HTTP schema tests MAY run in
parallel. Every rejected validation/context case MUST assert that no worker
`generate` command was emitted.

## 12. Non-Goals

The following require a later versioned contract:

- Responses API, Completions API, embeddings, audio, and image endpoints;
- request batching, parallel active requests, or a waiting queue;
- more than one model or tenant/key policy;
- tool/function calls and structured output guarantees;
- request stop-string matching;
- multimodal message content;
- automatic chat-history truncation;
- thinking/reasoning controls;
- prompt caching or cross-request KV reuse;
- full 40960-token context;
- model hot reload and automatic request retry;
- public/LAN binding or TLS termination.

## 13. Change Control

Any change to an accepted field, default, status code, error code, response
member, SSE order, token-accounting rule, authentication scope, topology,
deadline, or fixture payload requires a versioned specification change and an
updated contract fixture. Implementation convenience is not a reason to drift
from this document.

This specification is subordinate only to a later explicitly versioned product
contract. It is the P8-A source of truth for P8-D, P8-E, and P8-F HTTP behavior.

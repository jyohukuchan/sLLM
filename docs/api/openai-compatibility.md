# sLLM OpenAI-compatible Chat Completions profile v1

This document defines the initial public compatibility profile. “Compatible”
means compatible with the explicitly listed subset; it does not claim complete
OpenAI API compatibility.

## Normative source and version

Within this profile, request and response names, JSON shapes, and semantics are
defined by the official [OpenAI OpenAPI specification](https://github.com/openai/openai-openapi)
pinned at commit
[`117ce5680e4269f6656a4fd70d28f9755630d938`](https://github.com/openai/openai-openapi/tree/117ce5680e4269f6656a4fd70d28f9755630d938).
The exact schema file is
[`openapi.yaml`](https://github.com/openai/openai-openapi/blob/117ce5680e4269f6656a4fd70d28f9755630d938/openapi.yaml).
Moving this pin is a versioned compatibility decision and requires review of the
resulting request, response, and error differences.

This document selects the supported subset and imposes explicit rejection rules.
The pin does not make every endpoint or field in the OpenAI schema supported.
This profile determines which pinned-schema operations and fields sLLM accepts;
for that subset, the pinned schema governs JSON names, types, ranges, and standard
response shapes. Explicit rejection and status-code rules in this profile govern
sLLM behavior for requests outside that subset. The
[llama.cpp server API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
is an implementation reference and a peer for differential comparison; it is not
the sLLM API specification.

## Initial endpoints and multimodal revision

### `GET /v1/models`

Returns an OpenAI-style list object whose `data` entries identify models currently
available to requests. Initial entries expose the standard `id`, `object`,
`created`, and `owned_by` fields. A served `id` is a configured alias bound to a
model-lock fingerprint as specified in
[`../models/model-lock.md`](../models/model-lock.md), not a floating Hub revision.

### `POST /v1/chat/completions`

The original profile accepts one text chat and produces one text completion.
Phase 17 adds the versioned multimodal input subset below without changing the
response or SSE shape for text-only clients.

Required request fields:

- `model`: an `id` returned by `GET /v1/models`;
- `messages`: a non-empty ordered array of text messages.

Supported message roles are `system`, `user`, and `assistant`. For each message,
`content` must be a JSON string. Multipart content and the `developer`, `tool`, or
`function` roles are outside profile v1. This initial Qwen profile additionally
requires at least one `user` message and permits at most one `system` message,
which must be the first message.

Supported generation fields are:

- `temperature`;
- `top_p`;
- `max_completion_tokens`;
- `stop` as a string or array of strings;
- `presence_penalty`;
- `frequency_penalty`;
- `seed` as an optional signed 64-bit integer, matching the pinned OpenAPI `int64` schema;
- `stream`;
- `n`, only when its value is `1`.

The server validates the ranges and types defined by the pinned OpenAI schema. It
must reject any request containing an unsupported field or value even when the
rest of the request is valid; it must not silently coerce or discard it.
Phase 17 resource limits cap the JSON body at 96 MiB so two bounded Base64 images fit, the model alias at
256 UTF-8 bytes, messages at 1,024 entries, and `max_completion_tokens` at
1–4,096. A `stop` array contains 1–4 nonempty, unique strings; the total stop
payload is also bounded by the request-body limit.

For `stream: false` or an omitted `stream`, the response is the standard
`chat.completion` object with one choice. The choice contains an assistant text
message and a `finish_reason`; token accounting is returned in the standard
`usage` object when available under the pinned schema.

For `stream: true`, the response uses Server-Sent Events with content type
`text/event-stream`. Each event is framed as `data: <JSON>\n\n` and contains a
standard `chat.completion.chunk`. After the final JSON chunk, the stream terminates
with exactly `data: [DONE]\n\n`. Disconnect and generation failures must terminate
generation promptly; an error before response headers uses the normal JSON error
form below. After response headers have been sent, profile v1 emits one SSE event
whose `data` is the same standard error envelope, closes the stream immediately,
and does not emit a final completion chunk or `[DONE]`. This is an explicit sLLM
terminal-error convention, not a claim that OpenAI specifies an equivalent
mid-stream error event. Clients must treat close-without-`[DONE]` as failure and
must not retain the partial text as a successful completion.

## Errors and unsupported features

All non-streaming API errors use the standard envelope:

```json
{
  "error": {
    "message": "human-readable description",
    "type": "invalid_request_error",
    "param": "field_name_or_null",
    "code": "stable_machine_readable_code_or_null"
  }
}
```

The initial HTTP mapping is:

| Condition | HTTP status | `error.type` | `error.code` |
| --- | ---: | --- | --- |
| Malformed JSON | 400 | `invalid_request_error` | `invalid_json` |
| Invalid field/value | 400 | `invalid_request_error` | `invalid_value` |
| Unsupported input | 400 | `invalid_request_error` | `unsupported_parameter` |
| Invalid credential | 401 | `invalid_request_error` | `invalid_api_key` |
| Unknown model alias | 404 | `invalid_request_error` | `model_not_found` |
| Request too large | 413 | `invalid_request_error` | `request_too_large` |
| Rate limit exceeded | 429 | `rate_limit_error` | `rate_limit_exceeded` |

The table is the profile-v1 mapping for these conditions. Fields in the envelope
retain the pinned schema's types; `param` identifies the offending field when one
can be identified and is `null` otherwise.

Malformed JSON, an invalid type or range, an unknown model, an unknown standard
field, and any recognized-but-unsupported field or value return an appropriate
4xx response. Unsupported capability must never be silently ignored. In
particular, profile v1 rejects:

- tools, function calling, tool choice, and tool messages;
- image content outside the Phase 17 subset below, and all video or audio content;
- `logprobs` and `top_logprobs`;
- structured output and `response_format`;
- reproducibility claims involving `system_fingerprint`（同一model artifact、runtime、target、
  request parameter内では`seed`をsampling RNGへ固定するが、異なるtuple間のbitwise再現性は保証しない）;
- `n` with any value other than `1`;
- multipart message content and non-text output; and
- other pinned-schema fields not explicitly listed as supported above.

### Phase 17 image-content subset

For Qwen3.5 BF16, a `user` message may use an ordered `content` array containing
`{"type":"text","text":"..."}` and
`{"type":"image_url","image_url":{"url":"data:image/...;base64,..."}}` parts.
Part order is preserved. One or two unique images are accepted; image parts on
system/assistant messages, unknown part types, empty text, duplicate images,
HTTP(S) URLs, Files API IDs, and `detail` values other than omitted or `auto`
are errors. The server never performs an outbound image fetch.

PNG, JPEG, WebP, and a single-frame GIF are selected from magic bytes and must
match the data-URL MIME. Encoded bytes, decoded pixel area, aspect ratio, image
count, and total visual tokens are bounded before execution. Existing string
`content`, response objects, error envelopes, and SSE framing are unchanged.
CLI users may supply up to two trusted local files with repeated `--image PATH`;
the images are placed before the final user message text. This local-file form
is not an HTTP compatibility field.

Authentication and deployment policy are outside this payload-compatibility
profile. A deployment that requires authentication should use the standard
`Authorization: Bearer ...` header rather than adding credentials to JSON bodies.

## Initial production runtime

The initial `sllm-server` runtime serves one model-resident Qwen backend on one
GPU. On a host with multiple GPUs, deployments must make exactly the selected
GPU visible by its stable UUID and pass logical device index `0` to the server:

```console
ROCR_VISIBLE_DEVICES=GPU-76a08c022586fed6 sllm-server \
  --gguf /models/model.gguf --derived-lock /models/model.lock.json \
  --device-index 0 --target gfx1030 [server options]
```

The corresponding canonical R9700 invocation uses UUID
`GPU-a8e9ddefa2d60f55` and target `gfx1201`. A global physical device index in a
multi-visible-GPU process is not a supported initial deployment: HIP current
device state is thread-local, while the bounded scheduler executes generation on
a worker thread. UUID isolation makes logical device `0` stable on every server
thread and prevents a target-specific code object from being submitted to a
different visible GPU. Multi-GPU serving remains outside profile v1.

This deployment condition does not change the JSON or SSE compatibility claim.
The production backend uses the same transport-independent generation service as
the CLI, keeps the model resident between requests, and creates and releases
request-local KV/linear state for each request. An omitted extension continues to
use the locked Qwen chat template with thinking disabled.

The server context capacity is selected with `--context-length TOKENS`. When the
option is omitted, the exact model artifact's declared native/recommended context
is used. A larger value is accepted without an opt-in or override flag; startup
emits one `context_length_exceeds_recommended` warning on stderr containing both
values, and the ready event reports them. This warning is advisory about model
quality, not a request rejection. Each request still requires prompt tokens plus
`max_completion_tokens` to fit the configured server context, and allocation,
position-representation, and kernel resource failures remain ordinary runtime
errors.

## sLLM extensions

sLLM-specific request or response fields must be isolated under a clearly named
top-level `sllm` object, or exposed through a separately documented non-OpenAI
endpoint. They must not reuse a standard OpenAI field with different semantics.
Standard clients can therefore select this profile without accidentally enabling
engine-specific behavior.

An extension is opt-in. An unrecognized member inside `sllm` is also an error; it
is not silently ignored. Extension fields are not part of the compatibility claim
and must be documented and versioned independently.

### Thinking and separated reasoning extension

Qwen thinking is enabled per request with this closed extension:

```json
{
  "sllm": {
    "thinking": "enabled",
    "separate_reasoning": true
  }
}
```

`thinking` is `enabled` or `disabled` and defaults to `disabled` when the `sllm`
object or member is absent. `separate_reasoning` defaults to `false` and is valid
only with `thinking: "enabled"`. The enabled mode is passed to the verified fixed
Qwen renderer; it does not execute an arbitrary client-supplied template.

With separation enabled, a non-stream response adds `reasoning_content` to the
assistant message and keeps only the final answer in `content`. An SSE response
uses `delta.reasoning_content` for thinking and `delta.content` for the final
answer. The stateful separator recognizes `<think>` and `</think>` even when a tag
crosses backend delta boundaries, and does not expose the tags themselves. If
generation finishes before `</think>`, all generated text is reasoning and final
`content` is empty. Usage continues to report total completion tokens; no
reasoning-token sub-count is claimed.

The direct `reasoning_content` response member is an opt-in sLLM wire extension
chosen for existing client interoperability. It is not part of the pinned OpenAI
profile-v1 compatibility claim. Requests keep all sLLM-specific controls under the
top-level `sllm` object. For multi-turn round trips, an assistant input message may
carry a string `reasoning_content` beside its string `content`; the verified Qwen
renderer normalizes that history. The field is rejected on system and user
messages and remains outside the pinned compatibility claim.

### OpenWebUI compatibility profile

The server defaults to the strict profile and continues to reject legacy
`max_tokens`. A deployment that needs OpenWebUI's legacy request shape can opt in:

```console
sllm-server [required model and GPU options] \
  --compatibility-profile openwebui
```

Only that profile accepts `max_tokens` as an alias for
`max_completion_tokens`, with the same integer range of 1–4,096. Sending both
names is an error. All other strict field, role, content, sampling, error, and SSE
rules remain unchanged. The ready event reports the selected compatibility
profile so deployments can audit which behavior is active.

## Deferred API surface

The Responses API (`POST /v1/responses`) is planned for a future profile. It is not
an alias for Chat Completions and must not be exposed until its request items,
streaming events, tool behavior, multimodal behavior, and errors are implemented
against a separately pinned official schema.

Other OpenAI endpoints are unsupported unless a later compatibility document
explicitly adds them.

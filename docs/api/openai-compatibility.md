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

## Initial endpoints

### `GET /v1/models`

Returns an OpenAI-style list object whose `data` entries identify models currently
available to requests. Initial entries expose the standard `id`, `object`,
`created`, and `owned_by` fields. A served `id` is a configured alias bound to a
model-lock fingerprint as specified in
[`../models/model-lock.md`](../models/model-lock.md), not a floating Hub revision.

### `POST /v1/chat/completions`

The initial profile accepts one text chat and produces one text completion.

Required request fields:

- `model`: an `id` returned by `GET /v1/models`;
- `messages`: a non-empty ordered array of text messages.

Supported message roles are `system`, `user`, and `assistant`. For each message,
`content` must be a JSON string. Multipart content and the `developer`, `tool`, or
`function` roles are outside profile v1.

Supported generation fields are:

- `temperature`;
- `top_p`;
- `max_completion_tokens`;
- `stop` as a string or array of strings;
- `presence_penalty`;
- `frequency_penalty`;
- `stream`;
- `n`, only when its value is `1`.

The server validates the ranges and types defined by the pinned OpenAI schema. It
must reject any request containing an unsupported field or value even when the
rest of the request is valid; it must not silently coerce or discard it.

For `stream: false` or an omitted `stream`, the response is the standard
`chat.completion` object with one choice. The choice contains an assistant text
message and a `finish_reason`; token accounting is returned in the standard
`usage` object when available under the pinned schema.

For `stream: true`, the response uses Server-Sent Events with content type
`text/event-stream`. Each event is framed as `data: <JSON>\n\n` and contains a
standard `chat.completion.chunk`. After the final JSON chunk, the stream terminates
with exactly `data: [DONE]\n\n`. Disconnect and generation failures must terminate
generation promptly; an error before response headers uses the normal JSON error
form below. Mid-stream failure behavior must be documented before it is claimed as
compatible.

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
- image, video, or audio content;
- `logprobs` and `top_logprobs`;
- structured output and `response_format`;
- `seed` and reproducibility claims involving `system_fingerprint`;
- `n` with any value other than `1`;
- multipart message content and non-text output; and
- other pinned-schema fields not explicitly listed as supported above.

Authentication and deployment policy are outside this payload-compatibility
profile. A deployment that requires authentication should use the standard
`Authorization: Bearer ...` header rather than adding credentials to JSON bodies.

## sLLM extensions

sLLM-specific request or response fields must be isolated under a clearly named
top-level `sllm` object, or exposed through a separately documented non-OpenAI
endpoint. They must not reuse a standard OpenAI field with different semantics.
Standard clients can therefore select this profile without accidentally enabling
engine-specific behavior.

An extension is opt-in. An unrecognized member inside `sllm` is also an error; it
is not silently ignored. Extension fields are not part of the compatibility claim
and must be documented and versioned independently.

## Deferred API surface

The Responses API (`POST /v1/responses`) is planned for a future profile. It is not
an alias for Chat Completions and must not be exposed until its request items,
streaming events, tool behavior, multimodal behavior, and errors are implemented
against a separately pinned official schema.

Other OpenAI endpoints are unsupported unless a later compatibility document
explicitly adds them.

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

The original profile accepts one text chat and produces one or more bounded text
completions (`n` up to 8).
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

- `temperature`, which defaults to and only accepts `1.0`;
- `top_p`, which defaults to and only accepts `0.95`;
- `max_completion_tokens`;
- `stop` as a string or array of strings;
- `presence_penalty: 0.0` and `frequency_penalty: 0.0` (the only accepted values);
- `seed` as an optional signed 64-bit integer, matching the pinned OpenAPI `int64` schema;
- `stream`;
- `n`, an integer in the inclusive range `1..=8`;
- `logprobs: false` and `top_logprobs: 0` may be sent explicitly; logit bias,
  enabled logprobs, and nonzero top logprobs are rejected;
- `response_format` with the `text`, `json_object`, or bounded `json_schema`
  variants described below; and
- the opt-in `sllm` extension object described in [Sampler-chain extension](#sampler-chain-extension).

The server validates the ranges and types defined by the pinned OpenAI schema. It
must reject any request containing an unsupported field or value even when the
rest of the request is valid; it must not silently coerce or discard it. Seed,
stop, tool/JSON grammar, and reasoning controls remain request-selectable. The
fixed model profile resolves `top_k` to 20 for the reviewed Qwen coding
profile, 64 for Gemma 4, and 0 (disabled) for Ministral 3;
explicit sampler values must match that model value.

The fixed GPU selector uses numerical contract version 2. After the internal
valid-token mask, candidates are ordered by descending score, then ascending
token ID for ties. Top-k (unless zero) precedes top-p; the first candidate that
reaches the cumulative probability threshold is included. Sampling uses the
renormalized retained candidates. Signed API seeds map to the existing unsigned
64-bit seed representation; GPU draws use SplitMix64 with a request-local
counter and wrapping 64-bit arithmetic. A repeated seed is reproducible within
the same model, runtime, target, and request conditions. Switching from the old
CPU sampler to this GPU contract can change the generated sequence for the same
seed; CPU/GPU or cross-version sequence equality is not promised.

Phase 17 resource limits cap the JSON body at 96 MiB so two bounded Base64 images fit, the model alias at
256 UTF-8 bytes, messages at 1,024 entries, and `max_completion_tokens` at
1–4,096. A `stop` array contains 1–4 nonempty, unique strings; the total stop
payload is also bounded by the request-body limit.

For `stream: false` or an omitted `stream`, the response is the standard
`chat.completion` object with `n` choices. Each choice has a stable zero-based
`index`, an assistant text message, and a `finish_reason`; token accounting is
returned in the standard `usage` object when available under the pinned schema.
Prompt tokens are counted once while completion tokens are summed across
choices. Choice zero retains the requested `seed`; later choices use a
versioned deterministic derivation when a seed is present.

Logprob output is disabled in the fixed profile. Grammar, tool, and reasoning
masks still apply to candidate selection and continue to preserve their forced
token and stop behavior.

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

### Structured output

`response_format: {"type":"text"}` leaves ordinary unconstrained text
generation unchanged. `json_object` enables the bounded JSON grammar and also
requires at least one message to mention “JSON” (case-insensitive), matching the
OpenAI request convention. This generic mode permits depth 1, up to four
members/items per object/array container, string/number length up to 64, and
whitespace up to 16 bytes. `json_schema` accepts a schema envelope with a
non-empty name (at most 256 bytes), an optional description (at most 4,096
bytes), an optional boolean `strict`, and a schema no larger than 64 KiB.

The schema lowerer is deliberately fail-closed. It supports only `$ref` to a
local `$defs` entry, `$defs`, `type`, `properties`, `required`,
`additionalProperties: false`, `items`, `enum`, `const`, and `anyOf`. Supported
types are `object`, `array`, `string`, `number`, `integer`, `boolean`, and
`null`; property, enum, nesting, and grammar-state limits are bounded. Keywords
such as `pattern`, `format`, numeric/string/array size bounds, `allOf`, `not`,
conditional/dependent keywords, `oneOf`, remote references, and recursive
references are rejected with `invalid_value` instead of being ignored. The
lowerer does not silently widen this subset when a schema contains one of these
keywords.

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
- reproducibility claims involving `system_fingerprint`（同一model artifact、runtime、target、
  request parameter内では`seed`をsampling RNGへ固定するが、異なるtuple間のbitwise再現性は保証しない）;
- multipart content on system or assistant messages and non-text output; and
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

Source-tree startup also launches the loopback WebUI at `http://localhost:65457`
by default. Use `--webui false` for a headless/API-only deployment, or
`--webui-port PORT` to select a port independently from `--listen`. When WebUI
is enabled, omitted metrics default to enabled and the two local WebUI origins
are added to the exact CORS allowlist.

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

### Sampler-chain extension

`sllm.sampling` selects sampler-chain version `1` and keeps the stage order
transport-independent. The fixed profile accepts only neutral controls:
`top_k` must match the model profile (20 for the Qwen coding profile, 64 for
Gemma 4, or 0 for Ministral 3),
`min_p: 0`, `typical_p: 1`, `repeat_penalty: 1`, `repeat_last_n: 0`, and
`ignore_eos: false`. DRY and XTC may be omitted or supplied with zero
probability/multiplier; dynamic temperature and Mirostat are disabled.

```json
{
  "sllm": {
    "sampling": {
      "chain_version": 1,
      "top_k": 20,
      "min_p": 0,
      "typical_p": 1,
      "repeat_penalty": 1,
      "repeat_last_n": 0
    }
  }
}
```

Unknown members and non-neutral values return `unsupported_parameter`; the
server never silently rewrites them or falls back to host sampling.

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

## Phase 39 service operations extension

Phase 39 leaves Chat Completions profile v1 unchanged when its new deployment
features are disabled. `GET /healthz` is unauthenticated process liveness and
does not invoke generation. `GET /readyz` is unauthenticated readiness and
returns success only when the explicit server lifecycle is `ready` and the
scheduler still accepts work. It never substitutes CPU execution or another
backend for an unavailable model.

The following operational endpoints are sLLM extensions, not OpenAI API
compatibility claims:

| endpoint | authorization | contract |
| --- | --- | --- |
| `GET /metrics` | user or admin key | opt-in bounded Prometheus exposition; absent deployments return 404 |
| `GET /props` | user or admin key | model identity, fixed runtime-memory categories, lifecycle, scheduler and enabled features |
| `GET /slots` | admin key | redacted bounded queued/active slot snapshot |
| `POST /admin/slots/{id}/cancel` | admin key | cancel one known queued or active slot |
| `POST /admin/keys/reload` | admin key | atomically reload the configured key file |
| `GET /v1/chat/completions/{id}/events` | user or admin key | resume an explicitly resumable SSE response |

`props`, slots and metrics contain no prompt text, token IDs, credential,
request ID or backend error strings. Metric labels use only the startup model
aliases and fixed enums. Runtime memory values are sLLM-tracked device
allocation current/high-water values; they are not a claim about all
driver-visible VRAM. A busy backend returns a nonblocking zero snapshot rather
than delaying generation for a scrape.

Resumable streaming is a closed opt-in request extension:

```json
{
  "stream": true,
  "sllm": {
    "resumable": true
  }
}
```

It is rejected when `stream` is false or the server was not started with the
feature enabled. Each buffered SSE event has a monotonic event ID. A reconnect
supplies one decimal `Last-Event-ID`; the server emits only later events.
Unknown streams return 404 and cursors older than the bounded replay window
return 416. Replay is process-local and bounded by configured event count plus
64 KiB/event and 256 KiB/session. Ordinary SSE remains
non-resumable and keeps its existing disconnect-cancels-generation behavior.

CORS is disabled by default and, when enabled, accepts only configured exact
HTTP(S) origins—never `*`, paths, queries or userinfo. TLS is disabled by
default; certificate and private-key paths must be configured together and are
validated before model/backend startup.

## Phase 41 state extensions

Phase 41 does not add an OpenAI wire field or a new HTTP endpoint. Prefix cache,
context shifting, stateless prompt checkpointing, and draft-provider selection
are explicit server-startup controls. Their audit output uses fixed enums and
numeric counters and never includes checkpoint paths, token IDs, prompt text,
conversation bytes, KV bytes, seeds, or grammar payloads.

Checkpointing is not an OpenAI conversation/session API. A configured load is
validated once at backend startup and can continue only a request whose fully
rendered tokens have the stored token history as an exact prefix with a
non-empty suffix. Save occurs after fresh prompt prefill and before the first
visible delta. Load and save names are mutually exclusive. Unsupported
model/provider combinations and combinations with prefix cache, context shift,
or draft execution are rejected rather than silently falling back to a fresh
request. Mid-generation HTTP/SSE resume remains outside this contract.

Assistant history/prefill continues to use the existing typed assistant message
path. Prefilled assistant bytes initialize decoding and stop matching but are
not emitted again as completion content and are counted as prompt input.

## Phase 43 Responses profile

Phase 43 exposes `POST /v1/responses` as a separate strict profile. Its normative
OpenAI OpenAPI `2.3.0` pin is commit
[`010421dcbd0475277ea8c3e6c1e1cbca4659c4bd`](https://github.com/openai/openai-openapi/tree/010421dcbd0475277ea8c3e6c1e1cbca4659c4bd).
It is not an alias for Chat Completions. The checked-in machine contract is
[`phase43_protocol_profiles_v1.json`](../../tests/fixtures/phase43_protocol_profiles_v1.json).

The request accepts a served `model`, string or ordered typed `input`, optional
`instructions`, `max_output_tokens`, `temperature`, `top_p`, `stream`, bounded
metadata, `store: false`, client function definitions, `tool_choice`,
`parallel_tool_calls`, `reasoning.effort`, and `sllm.resumable`. Typed input is
limited to text messages, `function_call`, and `function_call_output`. Leading
system/developer messages retain their order and are combined for the fixed Qwen
renderer; a system/developer item after an ordinary message is rejected rather
than silently reordered. Images, audio, files, hosted tools, MCP, stateful
`previous_response_id`, and `store: true` are unsupported.

Non-stream output has `object: "response"`, a request-local `resp_` ID,
nonzero Unix `created_at`, selected `model`, status, typed output items,
`output_text`, incomplete/error fields, and input/output/total usage. Streaming
uses named Responses events, including content-part and function-argument
added/delta/done events, and ends once with `response.completed`; a failure uses
one `error` event. It never emits `[DONE]`. Visible deltas are split at UTF-8
boundaries to at most 16 KiB.

Client functions are protocol-only. Names, descriptions, schemas, calls and
client-owned results are validated and rendered as untrusted data. The server
does not execute a call or resolve a URL, path, process, network, filesystem,
environment, credential, MCP endpoint, hosted tool, worker, or sandbox. Tool
definitions are limited to 128, names to 64 ASCII bytes, descriptions to 16 KiB,
schemas to 1 MiB, call IDs to 256 bytes, arguments/results to 16 MiB, and one or
16 generated calls according to the parallel policy. The generated envelope is
JSON-Schema grammar constrained before scheduler/GPU admission.

`sllm.resumable` requires `stream: true`, `max_output_tokens <= 40`, and an
enabled Phase 39 replay store.
Replay retains exact named events within its configured event count, 64 KiB per
serialized event, and 256 KiB per session. The 40-token admission bound combines
the 128-byte token-piece cap with worst-case JSON escaping and bounded metadata,
so every admitted snapshot event fits. A defensive batch preflight still emits
one bounded error terminal instead of publishing a
partial success sequence if an invariant is violated. The replay GET endpoint
uses the deployment's normal bearer policy and does not require a content-type
header.

Authentication is deployment policy: an open server accepts no Authorization
header; a protected server requires its standard bearer header. Other OpenAI
endpoints remain unsupported unless a later compatibility document adds them.

## Phase 44 template・reasoning・interactive extensions

Phase 44 does not turn arbitrary client template source into an OpenAI wire field. The reviewed Qwen renderer remains the default for Chat
Completions/Responses, and the explicit generic provider is currently a transport-independent/frontend and CLI opt-in. Its source must be a bounded
regular non-symlink file with caller-supplied lowercase SHA-256; JSON-only messages/special tokens/flags/kwargs are rendered by the exact MiniJinja
`2.24.0` sandbox. Paths, prompt payloads, filesystem/environment/network/process access, and unbounded object callbacks are never exposed. Generic
raw-text and Gemma input are rejected rather than silently converted or routed to a reviewed template.

Reasoning remains a typed sLLM extension lowered through the same frontend controller used by Chat/Responses/CLI. `disabled`, `enabled`, and
`template-default` map to the existing thinking modes; an optional budget is 1..=4,096 generated reasoning tokens and includes any forced close
marker sequence. Admission accounts for `max_output_tokens`, grammar/stop/sampling/cancellation masks, and rejects an empty candidate or forced-token
mismatch before generation. Non-stream and SSE adapters expose reasoning/visible content split from the same generated token history; usage counts
forced and visible tokens normally and does not claim a separate reasoning-token counter. Anthropic thinking remains unsupported in its profile.

The new `chat` CLI is not an HTTP session API. It owns a closed prompt source matrix (`--prompt`, `--message`, `--prompt-file`, or interactive stdin),
bounded typed transcript, reverse-prompt turn boundary, and versioned JSONL events. Successful turns alone are published to the conversation. Save/resume
uses Phase 41's opaque stateless prompt checkpoint owner and exact model/template/tokenizer/target/KV identity; mid-generation HTTP/SSE resume and WebUI
remain outside this profile. MI300X real correctness/performance is deferred until a fresh exact runtime is available.

## Phase 45 adapter・dynamic model lifecycle extensions

Phase 45 keeps the existing Chat Completions, Completions, Responses, and Anthropic wire profiles unchanged. The only inference request extension is
the sLLM namespace: `sllm.adapters` and `sllm.control_vectors` are ordered selections of preloaded, verified artifact names. Each list is bounded to
four entries; an optional finite f32 scale is bounded to `[-16,16]`; duplicate names, non-canonical order, unknown artifacts, wrong base lock,
shape/dtype/rank/range mismatch, and cross-list duplicate names are rejected before model or GPU work.

Lifecycle management is an admin surface, not an OpenAI request field. With the same strict offline `--models` manifest used by the server, alias-only
actions are `load`, `preload`, `unload`, `clear-quarantine`, and `evict-idle` under the admin role. HTTP admin routes are
`/admin/models/{alias}/load`, `/preload`, `/unload`, `/clear-quarantine`, and `/admin/models/evict-idle`; no model path, URL, artifact payload, or
credential is accepted in a path or JSON body. Unknown aliases return 404, loading/draining models return 503, and a bounded queue returns 429.
The `sllm models` CLI uses loopback cleartext HTTP only and reads credentials from its environment/file policy; remote/HTTPS targets and direct token
arguments are rejected.

The machine-readable contract is [`phase45_adapter_lifecycle_v1.json`](../../tests/fixtures/phase45_adapter_lifecycle_v1.json), with the closed schema
[`phase45-adapter-lifecycle-v1.schema.json`](../../ci/schema/phase45-adapter-lifecycle-v1.schema.json) and dependency-free validator
[`validate_phase45_profiles.py`](../../ci/tools/validate_phase45_profiles.py). The compact [GPU summary](../../ci/matrix/phase45-adapter-lifecycle-gpu-summary-v1.json), schema,
validator, and mutation tests record the bounded V620 `gfx1030`/R9700 `gfx1201` release-build evidence: disabled/LoRA/control/combined cases are
bitwise-identical across two runs, HIP-only with fallback false, and cleanup/baseline restored. `gfx942`/MI300X runtime remains deferred.

## Phase 55 Gemma 4 MoE profile

The reviewed `gemma4moe` GGUF uses the existing Chat Completions, Completions,
Responses, SSE, model-list, metrics, cancellation, and dynamic model-lifecycle
transports without an architecture-specific wire field. Phase81 connects its
resident terminal row to the common device selector with the locked Gemma K=64
profile; while that connection is unavailable, Chat, Completions, and Responses
requests are rejected before request-state allocation and never fall back to CPU
or another model topology. Grammar, tools, logprobs, logit bias, random
sampling, reasoning mode, adapters, embeddings, rerank, infill, image input,
and draft execution retain their existing capability checks.

The artifact fixes KV storage to implicit-unit static E4M3. Process/model
configuration may omit the KV option or name `fp8-static`; FP16, retired
block-16, MXFP8, and NVFP4 KV options are rejected for this architecture.
Prefix-cache and checkpoint startup options use the same 30 opaque KV states;
an exact hit reuses its terminal Argmax, a partial hit executes only the new
suffix as M=1 transitions, and a divergent token history fails closed.
The bundled WebUI sends the fixed-profile request and continues to expose only
the output-token limit in its chat settings. It uses the canonical strict-profile
`max_completion_tokens` field; the legacy `max_tokens` alias remains confined to
the explicitly selected OpenWebUI compatibility profile.

## Phase 56 Gemma 4 MTP profile

Gemma 4 MTP adds no architecture-specific JSON field. Chat Completions,
Completions, model listing, non-stream responses, SSE, usage, stop exclusion,
cancellation, metrics, and lifecycle admin routes keep their existing wire
contracts. The initial assistant path is exact `gfx1201`, Argmax-only,
draft width 1, context at most 2,048, and the reviewed target/assistant pair.
It cannot satisfy the fixed stochastic profile; unsupported sampling, logprobs,
or structured generation fails before request execution.

Static startup supplies `--mtp-assistant-gguf PATH`,
`--mtp-assistant-derived-lock PATH`, and `--draft mtp-auto` beside the ordinary
target GGUF options. Dynamic WebUI startup instead selects a server-side model
folder; the library exposes the assistant as a disabled companion row and
attaches it to the compatible target alias. No assistant path or draft control
is accepted from an inference request. Target-only aliases remain target-only.

The server audit and Prometheus memory gauges include both target and assistant
resident allocations. Draft proposed/accepted/rejected counts remain bounded
backend audit data and do not change OpenAI response usage, which counts prompt
and visible completion tokens under the existing profile.

## Phase 42 inference profiles

Phase 42 adds separate, versioned wire contracts for inference modes. The
machine-readable source is [`phase42_profiles_v1.json`](../../tests/fixtures/phase42_profiles_v1.json)
and its closed Draft 2020-12 schema is
[`phase42-profile-v1.schema.json`](../../ci/schema/phase42-profile-v1.schema.json).
The dependency-free identity and boundary validator is
[`validate_phase42_profiles.py`](../../ci/tools/validate_phase42_profiles.py).
These artifacts describe the completed host contract and exact V620/R9700 GPU
acceptance. The feature-pinned gfx942 build passes, while MI300X runtime
execution remains deferred and is not implied by compile-only evidence.

The official OpenAI OpenAPI `2.3.0` operation subset remains pinned to commit
[`117ce5680e4269f6656a4fd70d28f9755630d938`](https://github.com/openai/openai-openapi/tree/117ce5680e4269f6656a4fd70d28f9755630d938).
The implementation reference is the detached llama.cpp `b10453` commit
[`3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`](https://github.com/ggml-org/llama.cpp/tree/3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70).
llama.cpp is a technical reference, not the sLLM API specification. Moving
either pin requires a new profile/schema version and an explicit field and
error-diff review.

| profile | endpoint | compatibility claim | fixed semantics |
| --- | --- | --- | --- |
| `openai-completions-v1` | `POST /v1/completions` | OpenAI subset only | Four prompt shapes (string, string array, token array, token-array array); `max_tokens` 1–4,096; fixed temperature `1.0`, top-p `0.95`, zero penalties, and logprobs `0`; `n` 1–8; stop strings 1–4, unique and nonempty; strict SSE/usage/error framing shared with the transport adapter. |
| `openai-embeddings-v1` | `POST /v1/embeddings` | OpenAI subset only | Four input shapes; `float` or `base64` output; arithmetic mean over final hidden rows, L2 normalization, finite F32 output, model-lock dimension and input ordering. Pooling and normalization are not client-selectable. |
| `sllm-rerank-v1` | `POST /v1/rerank` | sLLM-native, not OpenAI | L2-normalized query/document dot product; higher score wins; ties retain original document order; 1–256 documents; `top_n` is 1 through document count and is never silently clamped. |
| `sllm-token-utilities-v1` | `POST /v1/tokenize`, `/v1/detokenize`, `/v1/apply-template`, `/v1/input-tokens` | sLLM-native | Shared frontend tokenizer/renderer; model-default special-token policy; lossless byte fallback; verified template digest; no model execution or GPU execution. |
| `sllm-infill-v1` | `POST /v1/infill` | sLLM-native, not OpenAI | Prefix/suffix FIM mode with the common generation subset; model-lock capability and verified template are mandatory; unsupported capability rejects and never falls back to generic completion. |

All profiles use the existing bounded body limit (96 MiB), 256-byte model
alias limit, fail-closed unknown fields, and no silent type coercion. Invalid
JSON, wrong type/range/non-finite/empty or mixed inputs, unsupported model
capability, and oversize inputs map to the versioned `ApiErrorV1` matrix. The
standard status/code pairs are 400 `invalid_json`, 400 `invalid_value`, 400
`unsupported_parameter`, and 413 `request_too_large`; the parameter path is
included whenever one exists. Token and template utility requests are checked
before model or GPU execution. No prompt, token, vector, or secret is emitted
to logs, metrics, or props.

The embedding and rerank contracts deliberately pin hidden semantics rather
than inheriting llama.cpp's future-changing aliases. The initial production
FIM route is unsupported until a reviewed model lock records FIM prefix,
suffix, middle token IDs, context limit, and verified template digest. Exact
`gfx942`/MI300X execution evidence is deferred until a fresh runtime is
available; compile-only or host evidence cannot promote that capability.

Phase 42 itself does not expose Responses, Anthropic Messages, tools/MCP, arbitrary
Jinja or template kwargs, multimodal embedding/rerank/infill, wire session
resume, or llama.cpp endpoint aliases. Its Completions and infill generation
fields use the same fixed temperature `1.0`, top-p `0.95`, and neutral penalty
profile described above; explicit nonfixed values are rejected before execution.

## Phase 60 Ministral 3 API status

Ministral 3 text execution is wired through the existing model-list, Chat
Completions, non-stream response, SSE, usage, cancellation, and dynamic model
library contracts. Its verified graph now has the common device-selector seam,
but the pinned official generation config supplies no top-k recommendation. The
explicit Ministral profile therefore disables top-k (`top_k=0`) while retaining
fixed temperature `1.0` and top-p `0.95`; the common selector must support this
tuple before execution and there is no host-sampling fallback. Image input,
vision execution, tools, logprobs, and unsupported sampling options fail closed;
JSON grammar uses the same internal selector mask when requested.

This is an integration-status statement, not a production-quality claim. The
same official GGUF executes on exact `gfx1030` and `gfx1201`, but its greedy
output currently diverges from the fixed llama.cpp oracle. The alias must not
be presented as quality-accepted until that numerical mismatch is resolved.

## Qwen3.8 NVFP4 R9700 text server (2026-09-06)

The dedicated `--qwen38-nvfp4 ABSOLUTE_DIRECTORY` profile serves the verified
Unsloth Qwen3.8-27B-NVFP4 artifact on exact gfx1201, logical device 0, using
FP16 or standard OCP MXFP8 E4 KV (`--kv-cache-encoding`; omitted selects MXFP8 E4).
The current local deployment explicitly selects FP16 and the Phase78 fast paths. It reuses Chat Completions, sampling, SSE, cancellation,
and the existing single-active-request scheduler. The `openwebui` compatibility
profile connects it to OpenWebUI through `/v1/models` and `/v1/chat/completions`.
Tokenization and apply-template utilities use the artifact's verified frontend.
Embeddings and tool protocol are unavailable for this text-only profile.

The 2026-09-06 deployment snapshot did not include Phase83 MXFP8 E4 KV fast-path integration/MTP,
batching, vision, MXFP6 KV, other artifacts, or other GPUs. The current companion
selection is described in the Phase84 section below. See the
[deployment and validation record](../history/2026/09/1-10/qwen38-nvfp4-r9700-server.md).

Phase82 enables the verified Qwen3.8 projection packs, deferred completion and
stateless decode Graph spans without optimization opt-ins; the append-attention
chain additionally requires FP16 KV. The fresh gfx1201 API probe used FP16 KV
with the fixed sampling profile and checked SSE, cancellation and cleanup.
The text profile's unsupported-tools response remains HTTP 400; it is not a
positive tool-calling claim. This source change does not replace the locally
deployed binary or change its explicitly configured Phase78 preset.
See the [Phase82 adoption scope](../history/2026/09/1-10/phase82-default-adoption-scope.md)
and [validation record](../history/2026/09/1-10/phase82-optimization-cleanup-default-adoption.md).

## Phase84 MTP companion selection

The Qwen3.8 NVFP4 profile accepts a verified quantized companion directory via
`--mtp-weights` on exact `gfx1030` and `gfx1201`, logical device 0. Selection is
server configuration, not a new Chat Completions request field. Requests reuse
the configured recipe; target weights and KV encoding remain unchanged. Omitting
the option preserves the BF16 companion. Combining it with `--draft disabled`
is rejected before model loading.

The audit records `mtp_weight_encoding` and `mtp_companion_digest`, plus observed
proposal/acceptance counts and separate prefix/proposal wall times. An encoding
being selectable does not imply a throughput or full-model quality improvement.
See [conversion, CLI/API usage and evaluation limits](../development/mtp-companion-quantization.md)
and the [Phase84 measurements](../history/2026/09/11-20/phase84-mtp-weight-quantization.md).

## Phase85 Qwen3.5-4B MX weight and KV compatibility scope (2026-09-13)

The existing Chat Completions, non-stream, SSE, cancellation, usage, and
cleanup contracts also cover the reviewed Qwen3.5-4B weight/KV combinations on
exact `gfx1030` and `gfx1201`. The [reviewed BF16 source lock](../models/locks/qwen3.5-4b-bf16.json)
is paired with each converted GGUF's verified derived lock.
These body artifacts use GGUF plus their derived lock directly; adapters and
quantization sidecars are not part of this body compatibility scope.

The accepted body combinations are BF16 or reviewed MXFP8/MXFP6 W/A weights
with explicit FP16 KV, and BF16 or reviewed MXFP8/MXFP6 W/A weights with
standard OCP MXFP8 E4 KV. The KV resolver runs before the weight/KV guard, so
an omitted KV option resolves to the existing `kv-mxfp8-e4` value for this
reviewed 4B recipe and reaches the same allowed combination; explicit `fp16`
remains available. The selector value and its resolver policy did not change;
the compatibility guard now accepts the reviewed MX body plus MXFP8 E4 KV
combination. MXFP8 E5 retains its existing scope. The MXFP8/MXFP6 body plus MXFP8 E4 KV combination on
`gfx942` is unverified/rejected and is not promoted by this entry.

Final r6 API evidence passed all 10 body configurations across the two exact
targets, with five requests per configuration covering non-stream, SSE,
cancellation, recovery, and a separate request. The compact source report is
`.local-artifacts/phase85/final-api-summary.json`; the detailed records remain
under `.local-artifacts/phase85/final-api-gfx*-body-*/`. The eight same-format
quality comparisons (20 logit rows each) were exact before/after: top-1 was
`20/20`, maximum absolute difference and KLD were zero. This is an
optimization-before/after statement and does not claim equality versus BF16.

The 12 full inference comparison rows per target kept generated token sequences
and VRAM equal. On R9700 `gfx1201`, MXFP8/MXFP6 body decode improved by about
`3.8–4.6%`/`4.7–5.6%`, including about `4.4%`/`5.5%` with MXFP8 E4 KV; the
V620 `gfx1030` rows were approximately flat (about `-0.4%` to `+1.5%`).
These are scoped comparison results, not a quality or completion claim for all
Qwen3.5 artifacts. See the [Phase85 history](../history/2026/09/11-20/phase85-mxfp8-mxfp6-common-kernels.md#最終採用と結論)
and the ignored quality reports `.local-artifacts/phase85/compare-final-quality-report.json`
and `.local-artifacts/phase85/compare-final-quality-report.md`.

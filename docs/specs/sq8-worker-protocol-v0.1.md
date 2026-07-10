# SQ8 Worker JSONL Protocol v0.1

Status: implementation contract for P8-A

## 1. Scope

This specification defines the private stdin/stdout protocol between one local
gateway process and one SQ8 inference worker process. The worker owns one resident
model and implements the synchronous session contract in
`sq8-serving-session-v0.1.md`.

The v0.1 admission contract is fixed:

- active requests: exactly zero or one (`active = 1` maximum);
- waiting requests: always zero (`waiting = 0`);
- a second `generate` is rejected immediately and is never queued;
- M=8 represents eight consecutive prompt tokens from the active request, never
  eight requests; and
- only a `released` event proves that the worker can accept another request.

The shorthand for this admission boundary is `active1 / waiting0`.

This protocol is not a public HTTP API and is not a replacement for the unchanged
P7 result schemas `ullm.sq8.generation.v1` and
`ullm.sq8.generation_benchmark.v1`.

## 2. Process and thread model

The worker contains these logical roles:

1. The inference thread constructs and owns the HIP context, stream, resident
   model, scheduler, KV cache, and serving session.
2. The stdin reader reads bounded JSONL commands and owns the active request's
   cross-thread cancellation token.
3. The stdout writer serializes all events through one ordered writer and flushes
   after every line.
4. A watchdog observes terminal-cleanup completion without touching HIP state.

The roles may share a thread except that command input must remain able to set the
atomic cancellation flag while inference is running. GPU-owned values MUST NOT
cross from the inference thread, and unsafe `Send` implementations are forbidden.

All command and event order in this specification refers to the order of complete
newline-terminated records at the relevant pipe boundary.

## 3. Framing and parser limits

Each record is exactly one UTF-8 JSON object followed by LF (`0x0a`). Writers MUST
emit LF, not an unterminated final record. Readers MAY accept CRLF by stripping one
trailing CR before JSON parsing.

The maximum record payload is 4,194,304 bytes excluding LF and optional CR. The
reader MUST enforce the limit while streaming and MUST NOT first allocate an
unbounded line. For an oversized but LF-terminated command, the reader drains
through that LF with bounded memory, rejects the record as `invalid_command`, and
keeps the current active request unchanged. EOF before a terminating LF after the
limit has been exceeded is a fatal framing error.

Additional parser requirements:

- top-level arrays, scalars, and `null` are rejected;
- duplicate object keys are rejected at every nesting level;
- unknown fields are rejected;
- missing required fields are rejected;
- JSON numbers must fit the destination integer type exactly;
- JSON non-finite numbers are invalid;
- maximum nesting depth is 16; and
- each object contains a string `schema_version` and string `type`
  discriminator.

Commands and events both use the single schema version `ullm.worker.v1`. The
fields `schema`, `protocol_version`, and separate command/event schema names are
forbidden. Any other `schema_version` is rejected.

Only `generate`, `cancel`, and `shutdown` are valid on stdin. Only `ready`,
`started`, `progress`, `token`, `released`, and `error` are valid on stdout. A
known type used in the wrong direction is still invalid.

Stdout is protocol-only. Human logs, diagnostics, HIP output, panics, and progress
not defined here go to stderr. A library that writes to stdout must be redirected
or disabled before the worker declares readiness.

## 4. Common field rules

`request_id` is 1 to 128 ASCII bytes and must match:

```text
^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$
```

It is opaque and case-sensitive. It MUST be unique while active. Once matching
`released` has been flushed and the active slot is cleared, the worker retains no
request-ID history. The gateway normally generates a new ID for each public
request.

Token IDs and counters are non-negative JSON integers. Token IDs must be less than
the loaded model vocabulary size. Timestamp fields are intentionally absent from
v0.1; the gateway may timestamp records when it receives them.

Field sets shown below are exact. Optional fields are explicitly marked; sending
an omitted optional field as `null` is allowed only where the schema says
`string|null`.

## 5. Commands

### 5.1 `generate`

```json
{"schema_version":"ullm.worker.v1","type":"generate","request_id":"req-1","prompt_token_ids":[1,2,3],"max_new_tokens":256,"sampling":{"temperature":0.6,"top_p":0.95,"top_k":20,"seed":0},"eos_token_ids":[151645,151643]}
```

Exact fields:

| Field | Type | Constraint |
| --- | --- | --- |
| `schema_version` | string | `ullm.worker.v1` |
| `type` | string | `generate` |
| `request_id` | string | common request-ID rule |
| `prompt_token_ids` | integer array | length 1..4096, each ID in vocabulary |
| `max_new_tokens` | integer | 1..512 |
| `eos_token_ids` | integer array | exactly `[151645,151643]` in v0.1 |
| `sampling` | object | exact sampling fields below |

The exact `sampling` fields are:

| Field | Type | Constraint |
| --- | --- | --- |
| `temperature` | JSON number | finite and `0 <= temperature <= 2` |
| `top_p` | JSON number | finite and `0 < top_p <= 1` |
| `top_k` | integer | exactly the configured product value 20 |
| `seed` | integer | signed 64-bit, `-9223372036854775808..=9223372036854775807` |

All sampling fields are required effective values. The gateway resolves omitted
public API values before writing the command: temperature 0.6, top-p 0.95,
top-k 20, an OS-random signed 64-bit seed, and default maximum completion 256.
The worker applies no implicit request defaults.

The worker also requires checked
`prompt_token_ids.length + max_new_tokens <= 4096`.

Before installing the active slot, the stdin reader validates every request-owned
field using the immutable loaded vocabulary and limits, including all token IDs,
sampling fields, and checked context arithmetic. The inference thread repeats the
same request validation and then validates runtime/model/cache invariants before
session mutation. Reader acceptance is not session acceptance.

### 5.2 `cancel`

```json
{"schema_version":"ullm.worker.v1","type":"cancel","request_id":"req-1","reason":"client_disconnect"}
```

Exact fields:

| Field | Type | Constraint |
| --- | --- | --- |
| `schema_version` | string | `ullm.worker.v1` |
| `type` | string | `cancel` |
| `request_id` | string | must equal the active request ID |
| `reason` | string | `client_disconnect`, `slow_client`, `shutdown`, or `operator` |

For the matching active request, the reader atomically sets the request's
cancellation token with a Release store. Repeated matching cancels are idempotent;
the first reason is retained. A cancel does not synchronously imply completion,
and the gateway must keep draining events until `released`, fatal `error`, or EOF.

### 5.3 `shutdown`

```json
{"schema_version":"ullm.worker.v1","type":"shutdown"}
```

When idle, the worker shuts down cleanly after all prior event lines are flushed.
When active, `shutdown` acts as a cancel with reason `shutdown`; the worker drains
through `released` and then exits zero. A later command is rejected because input
has entered closing state.

EOF on stdin has the same semantics as `shutdown`.

## 6. Events

### 6.1 `ready`

Emitted exactly once after the model, session, scheduler, caches, and stream have
loaded and the session has proven its baseline:

```json
{"schema_version":"ullm.worker.v1","type":"ready","model":"ullm-qwen3-14b-sq8","model_revision":"9a283b4a5efbc09ce247e0ae5b02b744739e525a","artifact_content_sha256":"2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147","package_manifest_sha256":"c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb","device":"gfx1201","execution_profile":"rdna4_w8a8_block_ck","context_length":4096,"max_new_tokens":512}
```

The exact required fields are `schema_version`, `type`, `model`,
`model_revision`, `artifact_content_sha256`, `package_manifest_sha256`, `device`,
`execution_profile`, `context_length`, and `max_new_tokens`. The artifact field is
the frozen canonical artifact content SHA-256, not a source-model or arbitrary
file hash. The package field is the SHA-256 of the frozen thin package's
`manifest.json`. The values shown above are the product constants. The gateway
MUST compare every identity and limit with its configured deployment identity
before forwarding traffic.

No command may be accepted before `ready`. A load failure produces a fatal error
when the writer is available, then a nonzero exit; it never produces `ready`.

### 6.2 `started`

Emitted once after the inference thread validates the request, atomically claims
its scheduler/cache ownership, and enters `Prefilling`:

```json
{"schema_version":"ullm.worker.v1","type":"started","request_id":"req-1","prompt_tokens":42}
```

The exact fields are `schema_version`, `type`, `request_id`, and `prompt_tokens`.
A cancellation already set at this boundary does not suppress `started`; the
request still drains through its cancel and reset path.

### 6.3 `progress`

A progress event reports completed synchronized prompt work. It is never a reuse
signal. The exact fields are `schema_version`, `type`, `request_id`, `phase`, and
`processed_prompt_tokens`:

```json
{"schema_version":"ullm.worker.v1","type":"progress","request_id":"req-1","phase":"prefill","processed_prompt_tokens":8}
```

`phase` is exactly `prefill` in v0.1. During prompt processing the worker emits a
progress event whenever the cumulative processed count reaches another eight
tokens: 8, 16, 24, and so on. It also emits progress at the prefill/decode
transition with `processed_prompt_tokens` equal to the full prompt length. When a
multiple-of-eight milestone is also the transition, one event satisfies both
requirements. Thus a prompt shorter than eight still emits one transition event.
The cadence is independent of whether those tokens ran as eight M=1 steps or one
single-request M=8 chunk.

There is no `resetting` progress event and no separate decode progress event.
Every flushed `token` counts as protocol progress.

### 6.4 `token`

```json
{"schema_version":"ullm.worker.v1","type":"token","request_id":"req-1","index":0,"token_id":123}
```

Exact fields are `schema_version`, `type`, `request_id`, `index`, and `token_id`.
`index` starts at zero and is contiguous. The worker validates its internal cache
length against the session equation, but cache length is not a protocol field.

The line MUST be flushed before the token is considered published. After the
inference thread observes cancellation, no further token line may be written.

### 6.5 `released`

`released` is the sole successful terminal and reuse-readiness event:

```json
{"schema_version":"ullm.worker.v1","type":"released","request_id":"req-1","outcome":"stop","prompt_tokens":42,"completion_tokens":17,"reset_complete":true}
```

Exact fields:

| Field | Type | Constraint |
| --- | --- | --- |
| `schema_version` | string | `ullm.worker.v1` |
| `type` | string | `released` |
| `request_id` | string | active request ID |
| `outcome` | string | `stop`, `length`, or `cancelled` |
| `prompt_tokens` | integer | original prompt length |
| `completion_tokens` | integer | number of published token events |
| `reset_complete` | boolean | always `true` |

For `outcome = cancelled`, the exact field set additionally contains required
`cancel_reason` with the retained first cancel reason:

```json
{"schema_version":"ullm.worker.v1","type":"released","request_id":"req-2","outcome":"cancelled","cancel_reason":"client_disconnect","prompt_tokens":42,"completion_tokens":3,"reset_complete":true}
```

For `stop` and `length`, `cancel_reason` MUST be omitted rather than sent as
`null`.

The inference thread creates the release summary only after synchronized reset and
all baseline checks. The writer flushes `released` before the reader clears the
active slot. A new `generate` arriving before that flush is busy and is rejected;
it is not queued.

`reset_complete` can only be `true`. Internally, it means scheduler empty,
allocator at baseline, and every KV cache length zero. Cleanup failure is fatal
and produces no `released`.

### 6.6 `error`

```json
{"schema_version":"ullm.worker.v1","type":"error","request_id":"req-2","code":"busy","recoverable":true,"message":"one request is already active"}
{"schema_version":"ullm.worker.v1","type":"error","request_id":"req-1","code":"runtime_failed","recoverable":false,"message":"runtime entered a poisoned state"}
```

Exact fields are `schema_version`, `type`, `request_id`, `code`, `recoverable`, and
`message`. `request_id` is a string or `null`. `message` is bounded to 1024
UTF-8 bytes and MUST NOT contain model paths, prompt text, tokens, stack traces, or
unescaped control characters.

Recoverable pre-mutation codes are:

- `invalid_command`;
- `invalid_request`;
- `busy`;
- `unknown_request`.

An invalid `generate` gets one recoverable `error` and no `started` or `released`,
because it never owns worker resources. A second `generate` gets `busy`; the
active request continues unchanged. A nonmatching cancel gets `unknown_request`;
the active request remains active.

Fatal codes include `load_failed`, `runtime_failed`, `invariant_failed`,
`protocol_framing_failed`, and `cleanup_deadline_exceeded`. A fatal error sets
`recoverable = false`. If it refers to the active request, `request_id` is that ID;
otherwise it is `null`.

After a fatal error, no `released` is allowed and the process exits nonzero. When
the inference thread is hung, the watchdog may be unable to emit an error through
the single writer; nonzero exit or EOF is itself the fatal signal.

## 7. Admission and active-slot ordering

The reader maintains one active control slot containing only:

- accepted `request_id`;
- the request's `Arc<AtomicBool>` cancellation flag;
- the first cancellation reason, if any; and
- a monotonically increasing internal generation number.

It contains no HIP or session state. Access is synchronized so the ID and token
belong to the same generation.

Admission ordering is:

1. If closing, reject the command.
2. If an active slot exists, reject `generate` with `busy` immediately.
3. Perform the complete reader-side request validation defined in section 5.1.
4. Install the active control slot.
5. Send the immutable request and token to the inference thread over a capacity-1
   channel.
6. If the send fails, remove the matching generation and terminate fatal.
7. The inference thread performs full validation and either reports rejection or
   emits `started`.
8. Clear the active slot only after a rejected request's `error` is flushed or a
   successful request's `released` is flushed.

There is no waiting slot or pending request queue. Channel capacity is not request
queue capacity; it only transfers the single admitted request to an idle inference
thread.

## 8. Required event sequences

Normal completion:

```text
ready
started(request A)
progress(A, prefill)+
token(A)+
released(A, stop|length)
```

Cancellation:

```text
ready
started(request A)
progress(A, prefill)*
token(A)*
released(A, cancelled)
```

Pre-mutation rejection:

```text
ready
error(request B, recoverable=true)
```

Fatal execution or reset failure:

```text
ready
started(request A)
...
error(request A, recoverable=false)?
EOF with nonzero exit
```

`?` means the error is best effort only when the process is not able to use its
ordered writer. In all sequences, event order is strict for a request. Events from
different requests cannot interleave because `active = 1` and `waiting = 0`.

## 9. Cancellation race contract

The command reader performs a Release store; the inference thread performs
Acquire loads at the session boundaries defined by the serving-session spec.

- A token line fully flushed before the cancel command's flag store is valid.
- If the flag is observed before token publication, that token is not emitted.
- The worker never fabricates a token to complete a partially written line.
- A normal terminal token already flushed before a later cancel wins; the worker
  releases with `stop` or `length`.
- If cancellation is observed before the normal terminal token is published,
  cancellation wins and release outcome is `cancelled`.

The gateway MUST continue reading after it sends cancel. It may close the client
connection, but it must keep the worker event pump alive through `released`, fatal
error, or EOF.

## 10. Output backpressure

The gateway's per-request client queue has a maximum capacity of 32 token events.
The worker stdout and stderr pipes are continuously drained by dedicated pumps; an
HTTP client must never read directly from worker stdout.

The first failed nonblocking enqueue because the 32-event queue is full MUST
immediately send `cancel` with reason `slow_client`. There is no wait, retry, or
five-second grace period before cancellation. The pump then enters discard mode
for that request: every later `token` is parsed and discarded while both worker
pipes continue to drain through `released`, fatal `error`, or EOF. Progress and
terminal events are still parsed. After successful terminal release, the gateway
closes the HTTP stream without a final choice, usage chunk, error record, or
`[DONE]`.

The five-second deadline defined below starts after the cancel command is sent; it
is not permission to wait before sending cancel. The gateway must not declare the
worker idle merely because the client connection closed.

Progress events may be coalesced or dropped by the gateway after parsing, but the
worker emits them as specified and the pump must drain them. `token`, `error`, and
`released` events may not be dropped.

## 11. Hard deadlines and process poison

The gateway owns all four hard watchdogs:

| Deadline | Value | Start and successful terminal |
| --- | ---: | --- |
| worker Ready | 600 s | worker process spawn to validated `ready` |
| complete request | 180 s | gateway active-slot acquisition to matching `released` |
| protocol progress | 30 s | `generate` send; reset by matching `started`, `progress`, `token`, or `released` |
| cancel to release | 5000 ms | matching `cancel` send to matching `released` |

The worker also arms a 5000-ms terminal-cleanup watchdog when the inference thread
enters `Resetting`, for normal and cancelled requests. No separate longer cleanup
deadline applies. For cancellation, the gateway deadline may have started earlier
and therefore wins first. Successful verified reset cancels the worker watchdog
before `released` is enqueued, but only the flushed matching event satisfies the
gateway deadlines.

Measured cancel release latency starts when the reader validates the matching
cancel and sets its atomic flag, and ends when the protocol reader observes the
flushed `released(reset_complete=true)`. Its p95 target is at most 2000 ms. The
5000-ms deadline is the fatal correctness boundary, not the performance target.

Any hard-deadline breach, fatal worker error, unexpected EOF, or protocol
corruption is fatal to the gateway even if no `error` event exists. The gateway
MUST perform this bounded ordering:

1. mark readiness false;
2. attempt a nonblocking client error flush for at most 250 ms when applicable;
3. close the active client transport;
4. terminate the worker with at most a 2-second grace before kill; and
5. exit nonzero within 5 seconds of detecting the fatal condition.

Client backpressure cannot extend this sequence. Systemd, not the gateway, starts
a fresh control group. No `released` may be emitted after the worker cleanup
watchdog expires.

A synchronously detected fatal error is flushed when possible, and the process
must begin exit immediately. It must not wait for another command or attempt a
second reset.

## 12. Gateway readiness rules

The gateway considers the worker:

- `loading` before a validated `ready` event;
- `ready` after `ready` or a valid `released` event;
- `active` after it admits a `generate`, including before `started` arrives;
- still `active` throughout cancellation and reset;
- `failed` after fatal error, unexpected EOF, malformed output, event-order
  violation, identity mismatch, or nonzero exit; and
- `closing` after shutdown begins.

Only `ready` and `released` can transition the gateway to reusable ready state.
`progress`, the last token, a cancel command, or a recoverable error for some other
request cannot do so.

Lifecycle readiness is distinct from the idle active slot. `/readyz` remains 200
during healthy `Prefilling` and `Decoding`, and returns 503 during `Loading`,
`Restarting`, `Failed`, or `Shutdown`. Admission still rejects a request while the
single active slot is occupied.

If the worker emits a record that violates its schema, counters, active ID, or
event order, the gateway treats the worker as failed and terminates it. It must not
try to resynchronize at the next line.

## 13. Validation and conformance tests

The protocol implementation MUST include tests for:

- partial reads, multiple records per read, CRLF acceptance, and required flushes;
- exactly-at-limit and oversized records without unbounded allocation;
- invalid UTF-8, duplicate keys, unknown/missing fields, wrong `schema_version`, excessive
  nesting, integer overflow, and invalid float values;
- request-ID syntax, mismatched cancels, and repeated cancels;
- immediate second-request `busy` behavior with no waiting request;
- every legal normal, cancel, rejection, shutdown, and fatal event sequence;
- token indices, completion counts, and internal cache-length equations;
- cancel before start, during M=1, between M=8 chunks, during decode, and racing a
  terminal token;
- clearing the active slot only after `released` or rejection `error` flush;
- no `released` after reset failure or cleanup watchdog expiry;
- exact 600-second startup, 180-second request, 30-second progress, and 5000-ms
  cancel-release watchdog boundaries;
- immediate slow-client cancellation on the first failed nonblocking enqueue and
  discard/drain behavior for later token events;
- stdout containing protocol records only; and
- unchanged P7 generation and benchmark schema validators and evidence.

The end-to-end worker test must send a valid request, drain it through release,
then send a second valid request in the same process and prove that the second
request begins from zero cache and allocator baseline.

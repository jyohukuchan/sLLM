# SQ8 Worker JSONL Protocol v0.1

Status: implementation contract for P8-C, amended after the P8-B2 M=128 selection

## 1. Scope

This specification defines the private stdin/stdout protocol between one local
gateway process and one SQ8 inference worker process. The worker owns one resident
model and implements the synchronous session contract in
`sq8-serving-session-v0.1.md`.

The v0.1 admission contract is fixed:

- active requests: exactly zero or one (`active = 1` maximum);
- waiting requests: always zero (`waiting = 0`);
- a second `generate` is rejected immediately and is never queued;
- M=128 represents 128 consecutive prompt tokens from the active request, never
  128 requests; and
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
the loaded model vocabulary size. Absolute timestamp fields are intentionally
absent from v0.1; the gateway may timestamp records when it receives them. The
optional derived `timings` object on a successful `released` event contains only
request-relative durations and rates.

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
{"schema_version":"ullm.worker.v1","type":"progress","request_id":"req-1","phase":"prefill","processed_prompt_tokens":128}
```

`phase` is exactly `prefill` in v0.1. The selected product path emits one progress
event after every completed synchronized M=128 prompt chunk: 128, 256, 384, and so
on. It also emits progress at the prefill/decode transition with
`processed_prompt_tokens` equal to the full prompt length. When an M=128 chunk is
also the transition, one event satisfies both requirements. A prompt shorter than
128 therefore emits one transition event. M=1 tail steps do not synthesize old
eight-token milestones; cancellation is still checked between every M=1 step.

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

The session prepares a sampled token without advancing request-local RNG,
scheduler generated counters, or completion counters. The inference side then
enters a short publication critical section shared with the reader's matching
cancel store. Inside that section it rechecks cancellation, writes the complete
JSON line, flushes it, and only then commits the prepared RNG/scheduler/counter
state before unlocking. A matching cancel therefore linearizes either before
publication, in which case the token is discarded, or after flush and commit, in
which case the token is published. A write or flush failure advances no generated
state, is fatal, and produces no `released` event. An internal commit failure after
a successful flush is also fatal and produces no `released` event.

### 6.5 `released`

`released` is the sole successful terminal and reuse-readiness event:

```json
{"schema_version":"ullm.worker.v1","type":"released","request_id":"req-1","outcome":"stop","prompt_tokens":42,"completion_tokens":17,"timings":{"cache_n":0,"prompt_n":42,"prompt_ms":420.0,"prompt_per_token_ms":10.0,"prompt_per_second":100.0,"predicted_n":17,"predicted_ms":800.0,"predicted_per_token_ms":47.05882352941177,"predicted_per_second":21.25},"reset_complete":true}
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
| `timings` | object | optional compatibility extension for `stop`/`length`; forbidden for `cancelled` |
| `reset_complete` | boolean | always `true` |

The current resident worker MUST include `timings` for every `stop` and `length`
release. The gateway accepts its omission only for compatibility with earlier
`ullm.worker.v1` fixtures and workers. The object has the exact llama-server field
set shown above:

| Field | Meaning |
| --- | --- |
| `cache_n` | reused prompt tokens; always `0` because this worker has no prompt cache reuse |
| `prompt_n` | processed prompt tokens; equals `prompt_tokens` |
| `prompt_ms` | worker prompt start through first sampled token, in milliseconds |
| `prompt_per_token_ms` | `prompt_ms / prompt_n` |
| `prompt_per_second` | `1000 * prompt_n / prompt_ms` |
| `predicted_n` | all published completion tokens, including the first token and EOS |
| `predicted_ms` | first sampled token through final sampled token, in milliseconds |
| `predicted_per_token_ms` | `predicted_ms / predicted_n` |
| `predicted_per_second` | `1000 * predicted_n / predicted_ms` |

The prompt timer starts after the `started` event is flushed and immediately
before prompt execution. Each sample time is captured when the worker has
synchronously selected its token ID. `predicted_ms` excludes time to first token,
terminal cleanup, reset, the `released` write, gateway detokenization, and HTTP
delivery. It is clamped to at least `0.001` ms, matching llama-server's one
microsecond lower bound. Consequently, a successful one-token completion reports
`predicted_ms=0.001` and `predicted_per_second=1000000`.

For `outcome = cancelled`, the exact field set additionally contains required
`cancel_reason` with the retained first cancel reason:

```json
{"schema_version":"ullm.worker.v1","type":"released","request_id":"req-2","outcome":"cancelled","cancel_reason":"client_disconnect","prompt_tokens":42,"completion_tokens":3,"reset_complete":true}
```

For `stop` and `length`, `cancel_reason` MUST be omitted rather than sent as
`null`. For `cancelled`, `timings` MUST also be omitted; cancellation does not
fabricate successful generation performance data.

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
- the request's cancellation token containing the atomic flag and
  token-publication mutex;
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
ordered writer. Lifecycle event order is strict for the active request. A
recoverable `busy`, `unknown_request`, or `invalid_command` error for a rejected
command may appear between active-request events because the reader remains
responsive during inference. Such an error never claims GPU ownership and never
changes the active request's ordering or counters.

## 9. Cancellation race contract

The command reader performs a Release store while holding the active-slot
publication mutex; the inference thread performs Acquire loads at the session
boundaries defined by the serving-session spec and holds the same mutex across
token write/flush and internal commit.

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
- cancel before start, during M=1, between M=128 chunks, during decode, and racing a
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

## 14. Release measurement contract

### 14.1 Scope and primary series

This section freezes the resource-release measurement used by the P8 release
validator. A measurement point is taken only after a flushed matching
`released(reset_complete=true)` event. The 5-second idle settle described below
starts after that event, so it neither weakens nor extends the 180-second request,
30-second progress, or 5000-ms cancel-to-release deadlines in section 11.

The two primary byte series are:

- host bytes: systemd service cgroup v2 `memory.current`; and
- VRAM bytes: the target worker's `mem_usage.value` from the frozen AMD SMI
  process command, with `mem_usage.unit` exactly `B`.

Gateway and worker RSS, thread count, FD count, and child PIDs are required
diagnostics. KFD VRAM is a required cross-check. Neither the sum of process RSS nor
KFD VRAM may replace a missing primary value.

### 14.2 Frozen percentile and median

For percentile input `x` and probability `p`:

1. Reject an empty input, a non-finite input value, a non-finite `p`, or `p`
   outside `[0, 1]`.
2. Sort `x` in ascending order as `x[0]..x[n-1]`.
3. Compute `r = (n - 1) * p`, `lo = floor(r)`, and `hi = ceil(r)`.
4. If `lo == hi`, return `x[lo]`.
5. Otherwise return
   `x[lo] + (r - lo) * (x[hi] - x[lo])`.
6. Reject a non-finite result.

This calculation is used for every frozen p50 and p95 value. It is not a nearest
rank percentile.

Median input must also be nonempty and finite. Its sorted middle value is the
median for odd cardinality. For even cardinality it is the arithmetic mean of the
two middle values. The five resource samples at a measurement point therefore
select the third sorted value. Integer inputs may yield a fractional median in
other uses and MUST NOT be rounded before later calculations.

### 14.3 Frozen Theil-Sen slope

Let `value[0]..value[n-1]` be ordered post-release point medians in bytes. Reject
`n < 2` or any non-finite value. Construct every pairwise slope for all `i < j`:

```text
(value[j] - value[i]) / (j - i)
```

Sort all `n * (n - 1) / 2` finite slopes and take their median using section 14.2,
including the average of the two central slopes when the slope count is even. No
pair may be sampled or omitted. The request ordinal, not elapsed time, is the
independent variable, so the result unit is bytes/request (`B/request`).

The normal-operation acceptance slope uses exactly the 100 ordered post-release
medians, excluding warmups and baseline samples. It is computed independently for
primary host bytes and primary VRAM bytes. Each slope MUST be at most 262144
bytes/request. A negative slope is valid.

The post-restart segment uses only its own ordered post-release medians and the
same formula and threshold. It MUST NOT replace, extend, or be concatenated with
the required 100-value normal-operation series.

### 14.4 Segment schedule and baselines

The normal-operation segment is collected in this order:

1. Complete 10 warmup chats and observe release for each.
2. Wait at least 5 seconds after the tenth warmup's release and perform no request
   during that settle.
3. Capture the first sample, then capture four more resource samples with at least
   1 second between successive sample starts.
4. Take each field's five-sample median as the normal baseline.
5. Run exactly 100 sequential measured chats.
6. After each matching release, perform no request for at least 5 seconds, then
   capture the first sample and four more samples at intervals of at least 1
   second and compute that request's post-release median.

After the intentional fatal worker/service restart, wait for a newly validated
`ready`, complete 10 new warmup chats, and repeat the same baseline procedure. The
post-restart segment then contains exactly 20 sequential measured chats and the
same per-release settle and sample procedure.

Normal and post-restart raw samples, baselines, medians, deltas, and slopes are
separate populations. No value from one segment may be used to fill or compute a
value in the other. The planned restart must change the recorded process identity;
an unplanned identity change invalidates its segment.

For each segment and each primary byte series:

```text
final_delta = final_post_release_median - segment_baseline_median
```

`final_delta` MUST be at most 67108864 bytes. It is a signed delta; negative values
are not converted to absolute values. The normal final is request 100 and the
post-restart final is request 20.

For gateway and worker diagnostics, the median thread, FD, and child counts after
each release MUST equal their own segment baseline medians. Individual RSS values
and their derived deltas/slopes are recorded but remain diagnostic because cgroup
`memory.current` is the primary host-memory series.

### 14.5 systemd and cgroup host bytes

The frozen service unit is `ullm-openai.service`. The collector runs on systemd
255 with unified cgroup v2. It obtains `ControlGroup` and `MainPID` together from
`systemctl show`, requires a nonempty absolute `ControlGroup` without `..`, and
requires `MainPID > 0`.

The primary host byte path is exactly:

```text
/sys/fs/cgroup${ControlGroup}/memory.current
```

`stat -fc %T /sys/fs/cgroup` MUST return `cgroup2fs`. `memory.current` MUST contain
one non-negative base-10 integer, interpreted directly as bytes. The resolved path
must remain beneath `/sys/fs/cgroup`. A missing controller value, `max`, a unit
conversion, or a sum of per-process RSS is invalid.

`MainPID` is the gateway PID. Its `/proc/PID/stat` starttime and the returned
`ControlGroup` MUST remain stable throughout one segment. They are expected to
change only across the intentional fatal restart, after which a new baseline is
mandatory.

### 14.6 Process diagnostics and identity

For both the gateway and worker PID, the collector records:

- PID and `/proc/PID/stat` field 22 (`starttime`, in clock ticks);
- resolved `/proc/PID/exe` path;
- `/proc/PID/status` `VmRSS` and `Threads`;
- the number of directory entries in `/proc/PID/fd`; and
- the whitespace-separated direct child PIDs from
  `/proc/PID/task/PID/children`.

`VmRSS` MUST have the literal `kB` suffix. Convert it to bytes using checked
`VmRSS_kB * 1024`. `Threads` and every count are non-negative base-10 integers.
FD collection counts directory entries only. It MUST NOT `stat`, `readlink`, open,
or otherwise follow any FD symlink.

The worker is the unique direct gateway child whose resolved executable is the
frozen `ullm-sq8-worker` binary. Other child PIDs remain diagnostics and their
count is baseline-gated. The normal segment uses one unchanged gateway
PID/starttime pair and one unchanged worker PID/starttime pair, proving that no
normal request reloaded the worker.

For each resource sample, read `/proc/PID/stat` before the other `/proc` files and
again afterward. The before and after PID/starttime pairs MUST match. Re-read
systemd `MainPID` after the sample and require it to match the gateway identity.
A disappeared PID, PID reuse, changed starttime, changed parent relationship, or
identity change during sampling is a race failure, not a zero sample.

The `/proc/PID/stat` parser identifies the rightmost valid `) <state> ` delimiter
and validates the remaining field count before reading field 22. It MUST NOT split
the full line on whitespace or assume that `comm` contains no spaces or `)`
characters.

### 14.7 Physical R9700 and primary VRAM

The release target is the physical Radeon AI PRO R9700 with all identities fixed:

| Identity | Required value |
| --- | --- |
| AMD SMI GPU index | `2` |
| PCI BDF | `0000:47:00.0` |
| UUID | `a8ff7551-0000-1000-80e9-ddefa2d60f55` |
| KFD GPU ID | `51545` |

Before measurement, `amd-smi list --json` MUST contain exactly one entry matching
all four values. An index-only match is insufficient.

For every resource sample, run:

```text
amd-smi process --gpu 2 --general --json
```

The GPU 2 `process_list` MUST contain exactly one real process record. Its PID MUST
equal the identity-checked worker PID; no sentinel such as `No running processes
detected` is accepted. No unrelated process may appear. The target record MUST
contain non-negative integer `mem_usage.value` with `mem_usage.unit` exactly `B`.
That value is the primary VRAM byte sample.

Read the worker's KFD counter from:

```text
/sys/class/kfd/kfd/proc/${PID}/vram_51545
```

It MUST be a non-negative base-10 byte value equal to the AMD SMI primary value.
Enumerate the other KFD process directories and require that no other PID has a
positive `vram_51545` value. A missing KFD path, unit mismatch, value mismatch, or
unrelated R9700 process fails the measurement.

Run `amd-smi metric --gpu 2 --json` immediately before and after each segment. Its
GPU 2 temperature, socket power, and gfx/memory/fabric clocks are workload
diagnostics only. Metric-level `used_vram` is not process-isolated and MUST NOT be
used as the primary or KFD cross-check value.

### 14.8 Frozen commands, versions, and probe facts

Commands are executed without a shell, with placeholders replaced by validated
decimal PIDs or the validated `ControlGroup`. Their logical command strings are
frozen as follows:

| Key | Command string |
| --- | --- |
| `systemd_version` | `systemctl --version` |
| `service_identity` | `systemctl show ullm-openai.service --property=ControlGroup --property=MainPID --no-pager` |
| `cgroup_type` | `stat -fc %T /sys/fs/cgroup` |
| `host_memory` | `cat /sys/fs/cgroup${ControlGroup}/memory.current` |
| `proc_stat` | `cat /proc/${PID}/stat` |
| `proc_status` | `cat /proc/${PID}/status` |
| `proc_exe` | `readlink /proc/${PID}/exe` |
| `proc_fds` | `find -P /proc/${PID}/fd -mindepth 1 -maxdepth 1 -printf '%f\n'` |
| `proc_children` | `cat /proc/${PID}/task/${PID}/children` |
| `amd_smi_version` | `amd-smi version` |
| `amd_smi_list` | `amd-smi list --json` |
| `amd_smi_process` | `amd-smi process --gpu 2 --general --json` |
| `amd_smi_metric` | `amd-smi metric --gpu 2 --json` |
| `kfd_proc_probe` | `test -d /sys/class/kfd/kfd/proc` |
| `kfd_processes` | `find -P /sys/class/kfd/kfd/proc -mindepth 1 -maxdepth 1 -printf '%f\n'` |
| `kfd_vram` | `cat /sys/class/kfd/kfd/proc/${PID}/vram_51545` |

The probed release environment is:

- systemd major version 255 (`255.4-1ubuntu8.16` at contract freeze);
- cgroup filesystem type `cgroup2fs`;
- `/sys/class/kfd/kfd/proc` present;
- AMD SMI tool `26.2.2+e1a6bc5663` and library `26.2.2`; and
- ROCm `7.2.1`.

Every evidence run records the complete first line from `systemctl --version` and
the complete `amd-smi version` output in addition to the parsed values. A required
major/tool/ROCm mismatch is contract drift and fails before warmup.

### 14.9 Raw JSONL evidence schema

The raw evidence file is UTF-8 JSONL. Unknown fields, duplicate keys, non-finite
numbers, invalid UTF-8, and schema versions other than
`ullm.sq8.release_measurement.raw.v1` are rejected. It contains one `header`, 610
`resource_sample` records, and four `gpu_metric` records. Record order is header,
normal metric-before, normal baseline and request samples, normal metric-after,
restart metric-before, restart baseline and request samples, then restart
metric-after.

This is an offline evidence artifact, not worker stdin or stdout. It therefore
does not alter the `ullm.worker.v1` protocol discriminator from section 3.

The exact `header` record fields are `schema_version`, `record_type`,
`service_unit`, `commands`, `tools`, `probes`, and `schedule`. The nested field
sets are exactly those shown here:

```json
{"schema_version":"ullm.sq8.release_measurement.raw.v1","record_type":"header","service_unit":"ullm-openai.service","commands":{"systemd_version":"systemctl --version","service_identity":"systemctl show ullm-openai.service --property=ControlGroup --property=MainPID --no-pager","cgroup_type":"stat -fc %T /sys/fs/cgroup","host_memory":"cat /sys/fs/cgroup${ControlGroup}/memory.current","proc_stat":"cat /proc/${PID}/stat","proc_status":"cat /proc/${PID}/status","proc_exe":"readlink /proc/${PID}/exe","proc_fds":"find -P /proc/${PID}/fd -mindepth 1 -maxdepth 1 -printf '%f\\n'","proc_children":"cat /proc/${PID}/task/${PID}/children","amd_smi_version":"amd-smi version","amd_smi_list":"amd-smi list --json","amd_smi_process":"amd-smi process --gpu 2 --general --json","amd_smi_metric":"amd-smi metric --gpu 2 --json","kfd_proc_probe":"test -d /sys/class/kfd/kfd/proc","kfd_processes":"find -P /sys/class/kfd/kfd/proc -mindepth 1 -maxdepth 1 -printf '%f\\n'","kfd_vram":"cat /sys/class/kfd/kfd/proc/${PID}/vram_51545"},"tools":{"systemd_major":255,"systemd_version_line":"systemd 255 (255.4-1ubuntu8.16)","amd_smi_tool":"26.2.2+e1a6bc5663","amd_smi_library":"26.2.2","rocm":"7.2.1","amd_smi_version_output":"AMDSMI Tool: 26.2.2+e1a6bc5663 | AMDSMI Library version: 26.2.2 | ROCm version: 7.2.1 | amdgpu version: 6.16.13 | hsmp version: N/A"},"probes":{"cgroup_fs_type":"cgroup2fs","kfd_proc_present":true,"gpu_index":2,"gpu_bdf":"0000:47:00.0","gpu_uuid":"a8ff7551-0000-1000-80e9-ddefa2d60f55","kfd_gpu_id":51545},"schedule":{"normal_warmups":10,"normal_requests":100,"restart_warmups":10,"restart_requests":20,"idle_settle_ms":5000,"samples_per_point":5,"sample_interval_ms":1000}}
```

Each `resource_sample` has the exact top-level fields shown below. `segment` is
`normal` or `restart`; `phase` is `baseline` or `post_release`; and `sample_index`
is `0..4`. Baseline records set `request_index`, `request_id`, `release_outcome`,
`release_observed_monotonic_ns`, and `reset_complete` to `null`.
Post-release records require request index `1..100` for normal or `1..20` for
restart, a matching request ID, `release_outcome` equal to `stop`, `length`, or
`cancelled`, a release timestamp, and `reset_complete=true`.

```json
{"schema_version":"ullm.sq8.release_measurement.raw.v1","record_type":"resource_sample","segment":"normal","phase":"post_release","request_index":1,"request_id":"req-1","release_outcome":"stop","release_observed_monotonic_ns":1000000000,"reset_complete":true,"idle_settle_started_monotonic_ns":1000000000,"sample_index":0,"sample_monotonic_ns":6000000000,"systemd":{"control_group_before":"/system.slice/ullm-openai.service","control_group_after":"/system.slice/ullm-openai.service","main_pid_before":1200,"main_pid_after":1200},"host":{"memory_current_bytes":1000000000},"gateway":{"pid":1200,"ppid":1,"exe":"/usr/bin/python3.12","starttime_ticks_before":10000,"starttime_ticks_after":10000,"vmrss_kb":100000,"vmrss_bytes":102400000,"threads":8,"fd_count":32,"children":[1201]},"worker":{"pid":1201,"ppid":1200,"exe":"/opt/ullm/bin/ullm-sq8-worker","starttime_ticks_before":10001,"starttime_ticks_after":10001,"vmrss_kb":200000,"vmrss_bytes":204800000,"threads":12,"fd_count":24,"children":[]},"gpu":{"index":2,"bdf":"0000:47:00.0","uuid":"a8ff7551-0000-1000-80e9-ddefa2d60f55","kfd_gpu_id":51545,"process_record_count":1,"worker_pid":1201,"mem_usage":{"value":20000000000,"unit":"B"},"kfd_vram_bytes":20000000000,"unrelated_process_pids":[]}}
```

All timestamps are non-negative monotonic-clock nanoseconds from one boot. All byte
and count fields are non-negative integers. `children` and
`unrelated_process_pids` are ascending unique PID arrays. The validator checks
`vmrss_bytes == vmrss_kb * 1024`, both starttime pairs, gateway/worker parentage,
systemd identity, GPU identity, exactly one process record, empty unrelated PID
list, and AMD SMI/KFD byte equality for every sample.

Within each baseline or post-release point, records appear in `sample_index` order
`0,1,2,3,4`. Sample 0 requires
`sample_monotonic_ns - idle_settle_started_monotonic_ns >= 5000000000`. Each later
sample requires its start timestamp to be at least 1000000000 ns after the prior
sample's start timestamp. Actual scheduler delay is recorded; no exact equality or
upper timing bound is required.

Each `gpu_metric` record has exactly `schema_version`, `record_type`, `segment`,
`boundary`, `captured_monotonic_ns`, `gpu_index`, `raw_output_file`, and
`raw_output_sha256`. `boundary` is `before` or `after`. The referenced file is the
unmodified stdout JSON from the frozen metric command and is part of the
checksummed evidence bundle. Its basename is exactly
`amd-smi-metric-${segment}-${boundary}.json`, and its SHA is 64 lowercase
hexadecimal characters.

```json
{"schema_version":"ullm.sq8.release_measurement.raw.v1","record_type":"gpu_metric","segment":"normal","boundary":"before","captured_monotonic_ns":500000000,"gpu_index":2,"raw_output_file":"amd-smi-metric-normal-before.json","raw_output_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
```

The independent validator derives all five-sample medians, final deltas,
percentiles, and Theil-Sen slopes from these raw records. A producer `passed`
field is not part of the schema and is rejected. It cross-checks the 10 warmup
releases, 100 normal releases, intentional restart, 10 restart warmup releases,
and 20 restart releases against the machine-readable request/release matrix.

### 14.10 Failure behavior

The measurement fails closed on any required command failure, timeout, parse
error, missing/extra record, sample-count or ordering error, invalid unit,
non-finite value, integer overflow, identity race, unplanned restart, segment
mixing, hardware/probe/version mismatch, unrelated R9700 process, KFD mismatch, or
missing successful release. It MUST NOT drop a sample, impute a value, retry only
the failed point, substitute RSS/KFD/metric VRAM for a primary value, or reuse the
other segment's baseline. A new result requires a fresh complete segment run.

A worker deadline breach, fatal `error`, EOF, or `released` with an invalid schema
remains a product failure under sections 8 and 11. Resource collection does not
turn it into a measurement-only failure and does not extend the fatal ordering.

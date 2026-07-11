# SQ8 OpenWebUI Product Release Evidence v0.1

Status: frozen P8-F release contract

Date: 2026-07-11

This document is the normative P8-F release-evidence contract for the single
resident SQ8 worker, the OpenAI gateway, and OpenWebUI. The words **MUST**,
**MUST NOT**, **SHOULD**, and **MAY** are normative. A successful producer run
does not release the product by itself; `tools/validate-sq8-openwebui-release.py`
MUST independently reconstruct every gate in this contract from raw evidence.

## 1. Scope and Concurrency

The release target has exactly one active GPU request and no waiting request.
All GPU-mutating workloads in this contract run sequentially. A fixed-size
prefill chunk contains tokens from that one request only and is not request
batching. Request batching, continuous batching, a waiting queue, and
request-to-request KV reuse are absent from the v0.1 product and from every
release workload.

This contract covers:

- the OpenWebUI product smoke and 20-chat browser soak;
- five specifically phased cancellation cases and immediate recovery;
- one post-header worker failure, the resulting full systemd restart, and UI
  failure presentation;
- 100 normal-operation and 20 post-restart resource-measured HTTP chats;
- exact HTTP TTFT and decode schedules through the OpenWebUI Docker network;
- gateway lifecycle journal correlation; and
- independent validation and checksummed publication.

The HTTP surface remains the frozen
`docs/specs/openai-chat-subset-v0.1.md` contract. Worker protocol, reset meaning,
resource commands, sampling interval, percentile, median, and Theil-Sen rules
remain the frozen `docs/specs/sq8-worker-protocol-v0.1.md` contract. This
document narrows the P8-F schedule and binds those contracts together; it does
not redefine them.

## 2. Run Identity and Fail-Closed Rules

The session raw schema is `ullm.sq8.openwebui_release.raw.v1`. The resource raw
schema is the existing `ullm.sq8.release_measurement.raw.v1`. The gateway
lifecycle schema is `ullm.gateway.lifecycle.v1`.

All monotonic timestamps MUST be non-negative integer nanoseconds from the same
boot unless a field explicitly uses journal microseconds. The producer records
the boot ID and rejects a change. Session and resource JSON parsing rejects
duplicate keys, unknown fields, non-finite numbers, invalid UTF-8, invalid base64,
and a schema mismatch. Raw journal objects may retain additional journal metadata,
but every required field in section 7 is mandatory and is checked. Every raw
JSONL record is one object terminated by LF. A producer `passed` field is
forbidden in session and resource raw records.

The producer MUST record the start and final Git commit and porcelain-v1 status,
the gateway and worker source hashes, worker binary hash, model revision,
artifact and package identities, tokenizer file and chat-template hashes, frozen
fixture hashes, Python/Rust/package versions, vLLM oracle identity, physical GPU
identity, the complete derived OpenWebUI image identity below, Docker network
identity, systemd unit identity, and every input evidence SHA-256. The independent
validator takes the expected commit and worker binary SHA-256 as separate
command-line inputs and MUST NOT trust values copied only from producer output.

The producer writes each raw file as `NAME.incomplete`, flushes and `fsync`s it,
then atomically renames it only after that file's complete schedule succeeds.
The bundle is not publishable while an `.incomplete` file exists. A missing or
extra scheduled record, timeout, parse error, identity change outside the planned
restart, absent release acknowledgement, hash mismatch, unrelated positive R9700
owner, or collection failure invalidates the complete run. Samples are not
dropped, imputed, or replaced by rerunning only one point.

## 3. Complete Workload Order

GPU-mutating phases MUST run in this order:

1. identity, isolation, service, readiness, model-list, auth, request-schema, and
   fixed OpenWebUI smoke checks;
2. exactly 20 sequential successful OpenWebUI chats;
3. the five required cancellation phase cases from section 4, each followed by
   one successful recovery chat;
4. the normal resource segment from section 5;
5. one OpenWebUI chat that has visible content, followed by the sole intentional
   post-header worker kill, a visible failed state, and full service recovery;
6. the post-restart resource segment from section 5;
7. the HTTP TTFT and decode matrix from section 6; and
8. final readiness, identity, source-state, journal, and checksum capture.

The post-header worker kill in step 5 is the intentional fatal restart required
by the resource contract. It MUST NOT be implemented as a second independent
restart test. Any other fatal or process identity change invalidates the run.
TTFT requests that close after first content are latency samples and do not
satisfy or replace any of the five required cancellation phase cases.

## 4. Frozen Cancellation Phases

The required cancellation set contains one measured case with each of these
exact `cancel_phase` values, in this order:

1. `after_started_before_progress`;
2. `prefill_after_128`;
3. `prefill_after_2048`;
4. `decode_after_first_content`;
5. `openwebui_stop_after_visible_content`.

The first three cases use a direct streaming HTTP client and a prompt long enough
to have a later prefill boundary. The fourth uses the direct streaming HTTP
client. The fifth uses the actual OpenWebUI Stop button through browser
automation; calling a gateway endpoint directly or merely closing a test client
does not satisfy it.

The independent validator derives the phase from raw HTTP/browser evidence and
the correlated gateway lifecycle events:

| `cancel_phase` | required event ordering |
| --- | --- |
| `after_started_before_progress` | `request_started`, then `request_cancel_requested`, with no intervening `request_progress`, `request_first_token`, or response content |
| `prefill_after_128` | `request_progress(processed_prompt_tokens=128)`, then `request_cancel_requested`, before any later progress or first token |
| `prefill_after_2048` | `request_progress(processed_prompt_tokens=2048)`, then `request_cancel_requested`, before any later progress or first token |
| `decode_after_first_content` | first non-empty SSE content becomes observable to the client, then the client closes and `request_cancel_requested` follows |
| `openwebui_stop_after_visible_content` | browser evidence proves non-empty assistant content visible, then records an enabled Stop-button click, then `request_cancel_requested` follows |

Every case MUST end in `request_released` with `outcome=cancelled`, retained
`cancel_reason`, and `reset_complete=true` within the frozen five-second cancel
deadline. No content may appear after cancellation is observed. The next recovery
chat MUST be admitted only after release and MUST complete successfully without a
worker or gateway identity change.

Other cancellation observations in the full release run, including TTFT client
closure, are classified by their own phase and are not counted in this exact
five-case matrix.

## 5. Frozen Resource Soak

The resource file is exactly `soak-resources.raw.jsonl` and uses section 14 of
`docs/specs/sq8-worker-protocol-v0.1.md` without substitution. It contains one
`header`, exactly 610 `resource_sample` records, and exactly four `gpu_metric`
records:

- normal baseline: 5 samples;
- normal post-release points: `100 * 5 = 500` samples;
- restart baseline: 5 samples; and
- restart post-release points: `20 * 5 = 100` samples.

The normal segment performs exactly 10 released warmup chats, establishes its
five-sample baseline after the five-second idle settle, then performs exactly 100
sequential measured chats. The non-greedy sampled request indices are fixed to
`5, 10, 15, ..., 100`; no other normal measured request is classified as sampled.
Those 20 requests use `temperature=0.6`, `top_p=0.95`, and an explicit seed equal
to the one-based request index. The other 80 use `temperature=0`, `top_p=1`, and
`seed=0`. All use the same recorded synthetic chat fixture, `stream=true`,
`stream_options.include_usage=true`, and `max_tokens=2`.

Three negative requests occur between valid measured requests and are not part of
the 100-request index:

- after request 25 and its resource point: one context-overflow rejection;
- after request 50 and its resource point: one malformed JSON rejection; and
- after request 75 and its resource point: a second context-overflow rejection.

Each negative request MUST be rejected before worker admission. The next indexed
valid request MUST succeed. The raw body, status, error body, gateway event set,
and absence of an admitted request are validated independently.

After the planned post-header fatal restart, the validator requires new gateway
and worker PID/starttime identities, a newly validated Ready event, exactly 10
new released warmup chats, and a separate five-sample baseline. It then requires
exactly 20 sequential measured chats with the same synthetic fixture,
`temperature=0`, `top_p=1`, `seed=0`, and the same post-release sampling cadence.
Normal and restart populations MUST NOT be concatenated or share a baseline.

Primary host memory is only the systemd service cgroup v2 `memory.current`
(`MemoryCurrent` in systemd terminology). Gateway RSS and worker RSS are required
diagnostics and MUST NOT be summed or substituted for primary host memory.
Primary VRAM is isolated R9700 worker-process `mem_usage.value` in bytes from the
frozen AMD SMI command; KFD VRAM is its required exact cross-check.

For both the 100-point normal series and the 20-point restart series, independently
for cgroup memory and primary VRAM:

- the final signed median delta from that segment baseline MUST be at most
  67108864 bytes; and
- the complete Theil-Sen slope MUST be at most 262144 bytes/request.

Gateway and worker RSS deltas and slopes are reported as diagnostics only.
Gateway/worker median thread, FD, and direct-child counts after every release
MUST equal their own segment baseline medians. The matching
`request_released(reset_complete=true)` event binds every point to the worker's
exact allocator-empty, scheduler-empty, and zero-KV reset acknowledgement.

## 6. HTTP Latency Schedule

The latency client runs in a fresh recorded container attached to the exact
OpenWebUI Docker network. It talks to the same bridge-restricted gateway endpoint
that OpenWebUI uses. It MUST NOT use host loopback, a host-network container, or a
parallel request. The container image digest, network ID, subnet, gateway, bridge
interface, target address, and client implementation hash are evidence inputs.

### 6.1 TTFT matrix

Fixture order is exactly:

1. `exact-p0032` (32 prompt tokens);
2. `exact-p0128` (128 prompt tokens);
3. `exact-p0512` (512 prompt tokens);
4. `exact-p2048` (2048 prompt tokens); and
5. `exact-p3584` (3584 prompt tokens).

For each fixture, run two warmups followed immediately by ten measured requests
before advancing to the next fixture. Every request uses the fixture's exact
`messages`, `stream=true`, `stream_options.include_usage=true`,
`max_tokens=512`, `temperature=0`, `top_p=1`, and `seed=0`.

The client establishes the connection before timing. TTFT starts immediately
after the synchronous write that contains the last request-body byte returns and
ends when the client has received enough bytes to parse the first SSE data object
whose `choices[0].delta.content` is a non-empty string. If that SSE object spans
multiple socket reads, its observation time is the timestamp of the final raw
body chunk needed to parse the complete object. Socket read boundaries are not
SSE boundaries and MUST NOT be treated as such.
The matching `request_first_token.observed_monotonic_ns` MUST NOT be later than
that first non-empty SSE content observation on the same monotonic clock.

After observing first content, the client closes the response, waits for the
matching `request_cancel_requested` and `request_released(reset_complete=true)`,
then verifies readiness before the next request. Warmups are retained as raw
evidence but excluded from percentiles. The ten measured TTFT values use the
frozen linear-interpolation percentile and MUST pass:

| prompt tokens | p50 maximum | p95 maximum |
| ---: | ---: | ---: |
| 32 | 2.5 s | 3 s |
| 128 | 4 s | 5 s |
| 512 | 10 s | 12 s |
| 2048 | 30 s | 35 s |
| 3584 | 50 s | 60 s |

### 6.2 Decode case

After the TTFT matrix, `exact-p0032` runs two warmups and ten measured complete
streaming requests with `max_tokens=64` and the same greedy settings. Each request
MUST publish 64 completion tokens and release with `outcome=length` and
`reset_complete=true`. The validator reconstructs SSE objects from the
concatenated raw body bytes and associates each completed object with the final
raw chunk needed to parse it.

Each valid decode sample MUST contain exactly 64 non-empty content SSE objects in
addition to a usage count of 64 and the matching release count of 64. For one
request, decode throughput is 63 divided by the elapsed seconds from the first to
the 64th content object. Its measured-request p50 MUST be at least 15 token/s.
Pool all 630 consecutive-content intervals from the ten measured requests; their
frozen linear-interpolation p95 MUST be at most 0.100 seconds.
An SSE read that contains multiple complete token objects gives those objects the
same observation timestamp; the raw chunks remain authoritative and no synthetic
spacing may be introduced.

## 7. Gateway Lifecycle Journal

Every gateway lifecycle JSON object has common fields
`schema_version="ullm.gateway.lifecycle.v1"`, `event`, and
`observed_monotonic_ns`. Event names and their additional exact fields are:

| `event` | additional exact fields |
| --- | --- |
| `request_admitted` | `request_id`, `completion_id`, `stream`, `prompt_tokens`, `max_completion_tokens` |
| `request_started` | `request_id`, `completion_id`, `stream`, `prompt_tokens`, `admit_to_start_ns` |
| `request_progress` | `request_id`, `completion_id`, `phase`, `processed_prompt_tokens`, `prompt_tokens` |
| `request_first_token` | `request_id`, `completion_id`, `stream`, `completion_tokens` (exactly `1`) |
| `request_cancel_requested` | `request_id`, `completion_id`, `stream`, `reason`, `admit_to_cancel_ns` |
| `request_released` | `request_id`, `completion_id`, `stream`, `outcome`, `cancel_reason`, `prompt_tokens`, `completion_tokens`, `reset_complete`, `admit_to_start_ns`, `start_to_release_ns`, `admit_to_release_ns` |
| `worker_fatal` | `request_id`, `completion_id`, `reason`, `admit_to_fatal_ns`; the IDs and duration are nullable together only when no request is active |

`request_released.cancel_reason` is a string for cancellation and JSON `null` for
`stop` or `length`. `reset_complete` is exactly `true`. All durations are
non-negative integers. `admit_to_release_ns` MUST equal
`admit_to_start_ns + start_to_release_ns`; lifecycle event observation times
independently enforce event order but are not substituted for the supervisor's
stored admission boundary. Lifecycle messages contain no API key, request
headers, user field, prompt/response text, or token IDs.

`service-journal.raw.jsonl` is the raw `journalctl --output=json` export bounded by
the run's start and final cursors. The required journal fields are `__CURSOR`,
`__MONOTONIC_TIMESTAMP`, `_BOOT_ID`, `_PID`, `_SYSTEMD_UNIT`, `PRIORITY`, and
`MESSAGE`. A `gateway_event` session record stores the exact raw `MESSAGE`, its
SHA-256, the journal cursor/monotonic/PID fields, and the decoded lifecycle object.
The validator locates the cursor in the raw journal, checks every copied byte and
hash, and decodes `MESSAGE` by exactly one of two rules:

1. use the complete `MESSAGE` when its first byte is `{`; or
2. remove exactly one ASCII prefix `INFO:     ` and require the next byte to be
   `{`.

No other prefix, repeated prefix, leading/trailing whitespace normalization, or
substring search is allowed. The remaining bytes MUST be exactly one lifecycle
JSON object. The validator correlates HTTP completion IDs, worker request IDs,
browser/fault timestamps, event order, terminal counts, reset acknowledgement,
and service restart identity from these raw records.

## 8. Session Raw JSONL

`raw-session-results.jsonl` uses only these record types:

| `record_type` | role |
| --- | --- |
| `header` | immutable run, clock, source, deployment, fixture, schedule, and threshold identities |
| `http_request` | exact request bytes and the last-body-byte timing boundary |
| `http_response_start` | raw status and response headers with their observation time |
| `http_body_chunk` | one socket-read byte string and its observation time |
| `http_response_end` | EOF, deliberate client close, timeout, or error and the complete-body hash |
| `gateway_event` | one raw journal `MESSAGE` plus hash and its exactly decoded lifecycle event |
| `api_journal_observation` | one non-lifecycle API-gate journal row projected as cursor, time, PID, and `MESSAGE` byte/hash identity |
| `lifecycle_quiet_check` | one fixed API-gate observer/journal quiet boundary and its cumulative journal cursor/count |
| `browser_action` | one browser command or wait assertion, selector, timing, result hashes, and screenshot identity |
| `lifecycle_probe` | bounded systemd identity, readiness, PID/starttime, restart count, and cgroup observation |
| `fault_injection` | the sole planned post-header worker kill and its target identity/timing |
| `run_end` | final source state, raw record counts, final journal cursor, and completion time |

There is no producer `sse_event` record. The validator concatenates
`http_body_chunk` bytes by request and chunk index, parses SSE framing itself, and
derives content, usage, error, and `[DONE]` events. A producer summary of parsed
SSE is not authoritative.

Every record has exact common fields `schema_version`, `record_type`, `sequence`,
`phase`, and `case_id`. `sequence` starts at zero and is contiguous. `phase` is
one of `preflight`, `api_contract`, `openwebui`, `cancellation`,
`resource_normal`, `post_header_failure`, `resource_restart`, `latency`, or
`final`. `case_id` is a non-empty stable string except in `header` and `run_end`,
where it is null.

Additional exact top-level fields are:

- `header`: `run_id`, `started_utc`, `clock`, `boot_id`, `identities`,
  `input_files`, `schedule`, and `thresholds`;
- `http_request`: `request_index`, `request_key`, `method`, `target`, `headers`,
  `body_base64`, `body_sha256`, `body_bytes`, `connect_completed_monotonic_ns`,
  `write_started_monotonic_ns`, and `last_body_byte_sent_monotonic_ns`;
- `http_response_start`: `request_key`, `status`, `headers`, and
  `observed_monotonic_ns`;
- `http_body_chunk`: `request_key`, `chunk_index`, `body_base64`, `body_sha256`,
  `body_bytes`, and `observed_monotonic_ns`;
- `http_response_end`: `request_key`, `outcome`, `error`, `body_bytes`,
  `body_sha256`, and `observed_monotonic_ns`;
- `gateway_event`: `journal_cursor`, `journal_monotonic_usec`, `journal_pid`,
  `message`, `message_sha256`, and `event`;
- `api_journal_observation`: `observation_index`, `journal_cursor`,
  `journal_monotonic_usec`, `journal_pid`, `message_utf8_bytes`, and
  `message_sha256`;
- `lifecycle_quiet_check`: `quiet_sequence`, `label`,
  `checked_monotonic_ns`, `observer_open`, `observer_event_count`,
  `new_journal_record_count`, `journal_record_count`, and `journal_cursor`;
- `browser_action`: `browser_case`, `action_index`, `action`, `selector`,
  `input_sha256`, `started_monotonic_ns`, `completed_monotonic_ns`, `result`,
  `screenshot_file`, and `screenshot_sha256`;
- `lifecycle_probe`: `probe`, `observed_monotonic_ns`, `service_active`,
  `ready_http_status`, `control_group`, `gateway_pid`, `gateway_starttime_ticks`,
  `worker_pid`, `worker_starttime_ticks`, and `n_restarts`;
- `fault_injection`: `injection`, `target_pid`, `target_starttime_ticks`,
  `signal`, `command`, `started_monotonic_ns`, and `completed_monotonic_ns`; and
- `run_end`: `completed_utc`, `completed_monotonic_ns`, `final_git_commit`,
  `final_git_status_raw`, `final_git_status_sha256`, `record_counts`, and
  `final_journal_cursor`.

The non-GPU API contract phase records exactly thirteen
`lifecycle_quiet_check` rows: one after each of the ten fixed HTTP cases, then
`http-client-shutdown`, `post-observer-close`, and
`final-readiness-and-identity`. Their observer lifecycle count is zero. Every
API service-journal row in that phase is copied once as an ascending,
zero-indexed `api_journal_observation`; each quiet check's cumulative count and
cursor MUST select the corresponding observation, and the final check MUST cover
the complete observation list. The independent validator locates the first
observation cursor in `service-journal.raw.jsonl` and requires the entire list to
be one byte-identical, uninterrupted global-journal span through the final
observation cursor.

The header `clock` is exactly `python.time.monotonic_ns`. `identities` contains
exactly `environment_file`, `environment_sha256`, `model_identity_file`,
`model_identity_sha256`, `openwebui`, `docker_network_id`,
`gateway_source_sha256`, `worker_source_sha256`, and `worker_binary_sha256`.
`openwebui` contains exactly `version`, `source_revision`, `base_image_digest`,
`base_image_id`, `derived_image_id`, `Dockerfile_sha256`, `patch_sha256`, and
`patched_middleware_sha256`. The fixed upstream base requires both its registry
digest and content-addressed image ID. The local derived image may have an empty
`RepoDigests` array, but its content-addressed Docker `.Id` is mandatory and MUST
equal `derived_image_id`. The Dockerfile, applied patch, and installed patched
middleware bytes are hashed independently; a matching tag is not an identity.
`input_files` is an ascending path array whose elements contain exactly `path`,
`bytes`, and `sha256`; it binds every fixture, prior oracle, source, executable,
and configuration input consumed by the run.

The header `schedule` contains exactly `openwebui_chats=20`,
`cancel_phases` equal to the ordered section 4 list, `normal_warmups=10`,
`normal_requests=100`, `sampled_normal_indices` equal to
`[5,10,...,100]`, `restart_warmups=10`, `restart_requests=20`,
`ttft_fixture_ids` equal to the ordered section 6.1 list,
`latency_warmups_per_case=2`, `latency_measured_per_case=10`,
`decode_warmups=2`, `decode_measured=10`, `idle_settle_ms=5000`,
`samples_per_point=5`, and `sample_interval_ms=1000`. The literal ellipsis is not
stored: `sampled_normal_indices` contains all 20 explicit integers.

The header `thresholds` contains exactly the five TTFT p50/p95 pairs,
`decode_p50_tokens_per_second_minimum=15`,
`decode_p95_inter_content_seconds_maximum=0.1`,
`cancel_release_max_ns=5000000000`,
`final_delta_max_bytes=67108864`, and
`theil_sen_max_bytes_per_request=262144`.

The request `headers` object contains exactly `content_type`, `content_length`,
and `authorization_mode`; `authorization_mode` is `valid_bearer`,
`invalid_bearer`, or `missing`, and no token or token hash is retained. Response
headers are an ordered array of two-string arrays so duplicate headers remain
observable. Request and chunk SHA-256 values are over decoded raw bytes, not the
base64 text.

`http_response_end.outcome` is `eof`, `client_closed`, `timeout`, or `error`.
`error` is null for `eof` and `client_closed` and otherwise is a bounded diagnostic
string. Its complete-body bytes/hash MUST equal the ordered concatenation of all
matching chunks, regardless of socket chunk boundaries.

A `browser_action` is the only browser observation record. Its `action` is one of
`navigate`, `select_model`, `submit_chat`, `wait_visible`, `click_stop`,
`wait_failed`, or `wait_ready`. `result` contains exactly `visible`, `enabled`,
`text_utf8_bytes`, and `text_sha256`, using null for a field not applicable to
that action. User-visible text is represented by byte length and SHA-256, not
stored in clear text. Screenshot fields are both null or name a regular bundle
file and its hash. The Stop case requires a screenshot immediately before the
click and the post-header failure case requires one after the failed state is
visible.

The sole `fault_injection` has `injection=post_header_worker_kill`,
`signal=SIGKILL`, and the identity-checked current worker PID. It occurs after
browser-visible non-empty content and before `worker_fatal`. The later lifecycle
probe MUST show a different gateway and worker identity, incremented systemd
restart count, and readiness 200.

### 8.1 Release matrix

`release-matrix.json` has schema
`ullm.sq8.openwebui_release.matrix.v1` and contains exactly
`schema_version`, `run_id`, `files`, `schedule`, and `thresholds`. `run_id`,
`schedule`, and `thresholds` MUST be JSON-type-sensitive matches for the same
values in the raw-session header. In particular, `thresholds` contains
`ttft_seconds_maximum`, whose exact fixture keys are the ordered section 6.1
IDs and whose values contain exactly numeric `p50` and `p95`, plus the five
remaining threshold fields defined above. A `passed` key is forbidden at every
nesting level.

`files` contains exactly one entry for every matrix input named below, sorted by
the UTF-8 bytes of `path`, with no duplicate path. Each entry contains exactly
`role`, `path`, `bytes`, and `sha256`; size and digest bind the regular file's
actual bytes. Roles are fixed as follows:

- `environment.json`: `environment`;
- `model-identity.json`: `model_identity`;
- `raw-session-results.jsonl`: `session_raw`;
- `soak-resources.raw.jsonl`: `resource_raw`;
- `service-journal.raw.jsonl`: `service_journal_raw`;
- each of the four `amd-smi-metric-*.json` files: `gpu_metric_raw`;
- each of `sampling-results.json`, `cancel-results.json`,
  `prefill-latency-results.json`, `api-contract-results.json`,
  `openwebui-smoke.json`, and `soak-results.json`: `derived_view`; and
- each required browser PNG: `browser_screenshot`.

The matrix never lists itself, `release-validation.json`, `summary.md`, or
`SHA256SUMS`. Those exclusions avoid self-reference and prevent a narrative or
producer verdict from becoming a trusted measurement input.

## 9. Evidence Bundle

The mandatory bundle layout is:

```text
benchmarks/results/YYYY-MM-DD/sq8-openwebui-v0.1/
  environment.json
  model-identity.json
  raw-session-results.jsonl
  soak-resources.raw.jsonl
  service-journal.raw.jsonl
  amd-smi-metric-normal-before.json
  amd-smi-metric-normal-after.json
  amd-smi-metric-restart-before.json
  amd-smi-metric-restart-after.json
  sampling-results.json
  cancel-results.json
  prefill-latency-results.json
  api-contract-results.json
  openwebui-smoke.json
  soak-results.json
  release-matrix.json
  release-validation.json
  browser/
    openwebui-stop-before.png
    post-header-failure.png
  summary.md
  SHA256SUMS
```

All listed paths are mandatory regular files or the one mandatory directory.
Symbolic links, devices, sockets, and additional unlisted evidence files are
rejected. `SHA256SUMS` lists every regular file above except itself and
`release-validation.json`, in ascending bytewise path order, using lowercase
SHA-256. Aggregate result JSON files and `release-matrix.json` are derived views.
The independent validator reconstructs
request counts, statuses, SSE, completion IDs, lifecycle ordering, cancellation
phases, TTFT/decode statistics, resource medians/deltas/slopes, identities, and
all file hashes from raw evidence rather than trusting aggregate `passed` fields.

`release-validation.json` is the validator's machine-readable result and MUST
contain its own gate details and the verified `SHA256SUMS` file hash, but it is
not an input to its pass/fail derivation and is excluded from `SHA256SUMS` to
avoid a self-reference. The validator writes it only after checking every other
bundle file, requires it to be absent before validation, and creates it with
exclusive-create semantics so an existing result cannot be overwritten. It is
mandatory only in the final successful layout. Publication and commit occur only
when the validator exits zero and `SHA256SUMS` verifies.

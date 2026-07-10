# SQ8 Standalone Worker Acceptance Evidence v0.1

Status: frozen P8-C measurement contract

Date: 2026-07-10

## 1. Scope

This contract closes P8-C for one resident `ullm-sq8-worker` process on the
physical R9700. It covers:

- cancellation latency with two warmups and ten measured requests;
- 100 sequential resource-measured requests with prompt and decode cancellation;
- reset and immediate normal-request recovery after every measured cancellation;
- bounded worker RSS and process-isolated R9700 VRAM growth.

It is not the P8-F service release measurement. P8-F uses
`ullm-openai.service` cgroup `memory.current`, includes the gateway, an intentional
restart, a second 20-request segment, 610 resource samples, and schema
`ullm.sq8.release_measurement.raw.v1`. This contract has no gateway, systemd
service, or restart segment and MUST use its own schema.

## 2. Fixed Identity

The raw schema version is `ullm.sq8.worker_acceptance.raw.v1`. The independent
validation result schema is `ullm.sq8.worker_acceptance.validation.v1`.

The run MUST use a tracked-clean source tree and one release worker built from
the recorded commit. Untracked profiler output does not affect `tracked_clean`.
The validator receives the expected commit and worker binary SHA-256 separately
from the raw producer and requires exact matches.

The fixed model inputs are:

- artifact manifest file SHA-256:
  `23977f4e9bed4bac4cc64c177c35d7f83355861426bf32027a69cf7a241552e2`;
- artifact content SHA-256:
  `2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147`;
- package manifest SHA-256:
  `c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb`.

The physical GPU identity is AMD SMI index `2`, PCI BDF `0000:47:00.0`, UUID
`a8ff7551-0000-1000-80e9-ddefa2d60f55`, and KFD GPU ID `51545`.
`HIP_VISIBLE_DEVICES` MUST be exactly `1`. Every required HIP guard recorded in
the header MUST equal `1`:

- `ULLM_REQUIRE_HIP_ADD_KERNEL`;
- `ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL`;
- `ULLM_REQUIRE_HIP_BF16_ROW_KERNEL`;
- `ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_F32_FLASH2_KERNEL`;
- `ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL`;
- `ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL`;
- `ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL`;
- `ULLM_REQUIRE_HIP_RMSNORM_KERNEL`;
- `ULLM_REQUIRE_HIP_ROPE_KERNEL`;
- `ULLM_REQUIRE_HIP_SILU_MUL_KERNEL`.

No other process may have positive VRAM on KFD GPU `51545` during the run.

## 3. Clock and Cancellation Bound

Every timestamp is a non-negative integer from Python `time.monotonic_ns()` on
one boot. The producer records a cancel command timestamp immediately before
calling stdin `write` and `flush`, and the stdout pump records the matching
`released` timestamp immediately after reading and strictly decoding the complete
line.

The measured interval is therefore:

```text
released_observed_monotonic_ns - cancel_write_started_monotonic_ns
```

This is a conservative upper bound, not an estimate of the exact internal
interval. Command write start necessarily precedes complete reader receipt,
validation, and the atomic cancellation store. Matching line observation follows
the ordered writer flush. Thus every true flag-store-to-release-observed sample is
at most its recorded bound. Element-wise upper bounds preserve the inequality
under the frozen linear-interpolation percentile. A measured upper-bound p95 at
or below two seconds therefore proves the original P8-C limit without changing
the worker protocol. Command write completion MUST NOT be used as the start.

All twelve latency requests are cancelled. Odd request ordinals target prompt
execution by sending cancel immediately after observing `started`; even ordinals
target decode by sending cancel immediately after observing token index `0`.
Prompt-target releases MUST contain zero completion tokens. Decode-target releases
MUST contain at least one completion token. Warmups are excluded from p95.

Each latency cancellation is followed immediately by one normal recovery request
in the same phase and with the same phase-local request index. Its request ID is
the cancelled request ID plus suffix `-recovery`; it uses the normal prompt and
generation settings from section 4 and MUST complete normally before the next
latency cancellation begins. Recovery requests are not p95 samples.

Every cancellation upper bound MUST be at most 5,000,000,000 ns. The ten measured
upper bounds use the percentile algorithm in section 7 and p95 MUST be at most
2,000,000,000 ns.

## 4. Request Schedule

All requests use `sampling = {temperature: 0.0, top_p: 1.0, top_k: 20, seed: 0}`
and `eos_token_ids = [151645, 151643]`. Prompt IDs are ascending integers starting
at one. `prompt_token_ids_sha256` hashes the concatenation of those token IDs as
unsigned 32-bit little-endian integers.

Request IDs are fixed to `p8c-latency-warmup-NN`,
`p8c-latency-measured-NN`, `p8c-resource-warmup-NN`, and
`p8c-resource-measured-NNN`, where the suffix is the one-based phase-local request
index with the shown zero padding. Latency recovery IDs append `-recovery`.

Normal requests use prompt length `8` and `max_new_tokens = 2`. They MUST release
with outcome `length`, completion count `2`, and `reset_complete = true`.

Prompt-cancel requests use prompt length `128` and `max_new_tokens = 512`.
Decode-cancel requests use prompt length `8` and `max_new_tokens = 512`.
They MUST release with outcome `cancelled`, cancel reason `operator`, a completion
count below `512`, and `reset_complete = true`.

After the latency segment, the resource segment runs in this order:

1. capture the `before` GPU metric;
2. complete exactly ten warmup requests;
3. wait at least five seconds after warmup 10 release with no request;
4. capture five baseline samples, with at least one second between sample starts;
5. complete exactly 100 measured requests sequentially;
6. after each matching release, perform no request for at least five seconds,
   then capture five samples with at least one second between sample starts;
7. capture the `after` GPU metric.

Each five-request block uses three normal requests, one cancellation, then one
normal recovery request. Cancellation indices are `4, 9, ..., 99`; the following
indices `5, 10, ..., 100` prove recovery. Cancellation ordinal 1, 3, ..., 19
targets prompt execution and ordinal 2, 4, ..., 20 targets decode. The same rule
applies to resource warmup indices 4 and 9.

There is one worker process, one active request, and no waiting request throughout
the complete run. A process identity change invalidates the run.

## 5. Resource Series

The primary host series is worker `/proc/PID/status` `VmRSS`, converted with
checked `VmRSS_kB * 1024`. The primary VRAM series is the matching worker process
`mem_usage.value` from:

```text
amd-smi process --gpu 2 --general --json
```

Its unit MUST be exactly `B`. The worker KFD counter at
`/sys/class/kfd/kfd/proc/PID/vram_51545` is a required byte-for-byte cross-check;
it is not a substitute primary value.

Each point contains exactly five samples and uses their median. The resource raw
file therefore contains exactly `5 + 100 * 5 = 505` `resource_sample` records.
Warmups have no resource samples. The 100 ordered post-release medians exclude
the baseline.

For worker RSS and primary VRAM independently:

- the complete 100-point Theil-Sen slope MUST be at most `262144` bytes/request;
- `point[100] median - baseline median` MUST be at most `67108864` bytes.

Negative slopes and deltas are valid. Worker thread, FD, and direct-child-count
medians at every post-release point MUST equal their baseline medians. Worker PID,
start time, executable, and parent PID MUST remain constant. AMD SMI VRAM and KFD
VRAM MUST be equal in every sample, and the unrelated-positive-KFD PID list MUST
be empty.

## 6. Raw JSONL Records

The evidence is UTF-8 JSONL. Each record MUST be one object terminated by LF.
Duplicate keys, unknown fields, non-finite numbers, invalid UTF-8, a line larger
than 8 MiB, or a different schema version are rejected. The raw file contains no
producer `passed` field.

Record order is:

1. one `header`;
2. one validated `ready` `worker_event`;
3. latency commands and events;
4. resource `before` `gpu_metric`;
5. resource warmup commands and events;
6. five baseline `resource_sample` records;
7. 100 measured request commands, events, and their 500 samples;
8. resource `after` `gpu_metric`;
9. one shutdown `command`;
10. one `process_exit`, which is the final record.

### 6.1 Header

The exact top-level fields are `schema_version`, `record_type`, `clock`, `build`,
`worker`, `device`, `environment`, `schedule`, and `thresholds`.

- `clock` is exactly `python.time.monotonic_ns`.
- `build` contains exactly `git_commit`, `tracked_clean`, `binary_sha256`,
  `artifact_manifest_sha256`, `artifact_content_sha256`, and
  `package_manifest_sha256`.
- `worker` contains exactly `pid`, `ppid`, `starttime_ticks`, and `exe`.
- `device` contains exactly `gpu_index`, `bdf`, `uuid`, `kfd_gpu_id`,
  `amd_smi_list_raw_json`, and `amd_smi_list_raw_sha256`. The validator reparses
  the raw list and requires exactly one entry matching all four identities.
- `environment` contains exactly `hip_visible_devices`,
  `required_hip_guards`, `amd_smi_version_raw`, and
  `amd_smi_version_raw_sha256`; the guard object contains exactly the ten names
  from section 2 with string value `1`. The AMD SMI raw version MUST contain tool
  `26.2.2+e1a6bc5663`, library `26.2.2`, and ROCm `7.2.1`.
- `schedule` contains exactly `latency_warmups=2`, `latency_measured=10`,
  `resource_warmups=10`, `resource_requests=100`, `cancel_block_size=5`,
  `cancel_block_offset=4`, `idle_settle_ms=5000`, `samples_per_point=5`, and
  `sample_interval_ms=1000`.
- `thresholds` contains exactly `cancel_sample_max_ns=5000000000`,
  `cancel_p95_max_ns=2000000000`, `theil_sen_max_bytes_per_request=262144`, and
  `final_delta_max_bytes=67108864`.

### 6.2 Command

Every `command` contains `schema_version`, `record_type`, `phase`,
`request_index`, `request_id`, `command_type`, `write_started_monotonic_ns`, and
`write_completed_monotonic_ns`. Timestamps are strictly increasing.

`phase` is `latency_warmup`, `latency_measured`, `resource_warmup`,
`resource_measured`, or `shutdown`. Generate records additionally contain exactly
`prompt_tokens`, `prompt_token_ids_sha256`, `max_new_tokens`, `sampling`, and
`eos_token_ids`. Cancel records additionally contain exactly `cancel_reason` and
`cancel_target`. Shutdown uses null request index and ID and has no additional
fields.

`cancel_target` is `prompt` or `decode`. There is exactly one cancel command for
each scheduled cancellation and no repeated cancel.

### 6.3 Worker Event

A `worker_event` contains exactly `schema_version`, `record_type`,
`observed_monotonic_ns`, and `event`. `event` is the unmodified decoded
`ullm.worker.v1` object. The validator independently checks its exact field set,
request identity, event ordering, progress, contiguous token indices, terminal
counts, outcome, cancel reason, and `reset_complete=true`.

### 6.4 Resource Sample

A `resource_sample` contains exactly `schema_version`, `record_type`, `phase`,
`request_index`, `request_id`, `release_outcome`,
`release_observed_monotonic_ns`, `settle_started_monotonic_ns`, `sample_index`,
`sample_started_monotonic_ns`, `worker`, and `gpu`.

For baseline records, `phase=baseline` and all request/release fields are null.
For post-release records, `phase=post_release`, request index is `1..100`, and the
request fields match the measured release. Sample indices are `0..4` in order.
Sample 0 starts at least 5,000,000,000 ns after settle start; later sample starts
are at least 1,000,000,000 ns apart.

The nested `worker` object contains exactly `pid`, `ppid`, `exe`,
`starttime_ticks_before`, `starttime_ticks_after`, `vmrss_kb`, `vmrss_bytes`,
`threads`, `fd_count`, and `children`. `children` is an ascending unique PID list.

The nested `gpu` object contains exactly `index`, `bdf`, `uuid`, `kfd_gpu_id`,
`process_raw_json`, `process_raw_sha256`, `worker_pid`, `mem_usage_value`,
`mem_usage_unit`, `kfd_vram_bytes`, `kfd_positive_processes`, and
`unrelated_positive_kfd_pids`. `kfd_positive_processes` is an ascending unique
array of objects containing exactly `pid` and `vram_bytes`. The validator reparses
`process_raw_json`, checks its SHA-256 and exact worker record, derives the
unrelated list from the KFD array, and requires that list to be empty.

### 6.5 GPU Metric

A `gpu_metric` contains exactly `schema_version`, `record_type`, `boundary`,
`captured_monotonic_ns`, `raw_json`, and `raw_sha256`. Boundary is `before` or
`after`. The raw value is the unmodified output from
`amd-smi metric --gpu 2 --json` and its SHA-256 MUST match.

### 6.6 Process Exit

The final `process_exit` contains exactly `schema_version`, `record_type`,
`stdout_eof_monotonic_ns`, `exit_observed_monotonic_ns`, `exit_code`,
`stderr_file`, and `stderr_sha256`. Exit code MUST be zero. `stderr_file` is a
regular sibling named `worker-stderr.jsonl` containing the complete worker stderr
byte stream and its SHA-256 MUST match. The successful raw evidence basename is
`raw.jsonl`; the producer writes `raw.jsonl.incomplete` and renames it only after
the complete successful run.

## 7. Frozen Statistics

For percentile input `x` and probability `p`, reject empty or non-finite input,
sort ascending, compute `r=(n-1)*p`, and linearly interpolate between
`floor(r)` and `ceil(r)`. This is used for p50 and p95.

The median is the middle sorted value for odd cardinality and the arithmetic mean
of the two middle values for even cardinality.

For ordered point medians `value[0]..value[99]`, construct all `4950` slopes
`(value[j]-value[i])/(j-i)` for every `i<j`, then take their median. No pair may be
sampled or omitted. The independent variable is request ordinal.

## 8. Failure Behavior

The producer and validator fail closed on any process, command, timeout, parse,
schema, ordering, identity, hash, resource, schedule, or threshold error. They do
not drop or replace a sample, retry only a failed point, continue after worker
fatal/EOF, or convert an incomplete run into a result. A new result requires a
fresh complete worker process and complete run.

An upper-bound p95 above two seconds does not prove the internal exact latency is
above two seconds, but it does fail this evidence contract. A narrower result then
requires a separately specified internal before/store/after monotonic bracket; it
must not overwrite or reinterpret this raw run.

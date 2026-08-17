# Local Qwen3.8 subagent

## Main agent contract

The local Qwen agent is the preferred command-backed delegation path for
bounded coding, repository inspection, test, and summarization work that fits
its capabilities. The normal agent loop is Pi coding agent 0.84.2 using Chat
Completions. It does not change the parent Codex model and must not be created
as a native Codex subagent. The main agent remains responsible for scope,
reviewing the report or edits, and running proportionate checks before
accepting the result.

Use the `qwen38-subagent` skill, then follow this sequence from the task's target
workspace:

```bash
/home/homelab1/.local/bin/qwen38-subagent-server status
/home/homelab1/.local/bin/qwen38-pi --profile standard --read-only \
  "Workspace: /absolute/workspace. Read-only task: <bounded objective>. Do not edit, commit, or push. Return: <expected output>."
```

For an editing task, use `qwen38-pi --workspace-write`, replace `Read-only`
with an exact allowed-file list, and state the required checks. The wrapper
defaults to the Standard profile and read-only mode. `--timeout-seconds N`
changes the profile deadline and `--no-timeout` removes it. Do not grant commit
or push authority unless the user explicitly requested it. Up to two
independent Qwen Pi or Harness tasks may be active at once. Both wrappers use
the same two process leases for their complete agent loops and exit without
queueing when both are occupied.

`qwen38-pi` checks the authenticated local health contract, starts the user
systemd service when absent, configures the Pi Chat Completions model, and runs
Pi offline. Read-only mode exposes read/search/shell tools; workspace-write
adds write/edit. A Landlock launcher leaves the host readable but permits
filesystem writes only in an ephemeral per-run scratch directory and, for
workspace-write mode, the current workspace. The task prompt still defines the
allowed file list; Landlock independently enforces the workspace boundary. If
local Qwen is unavailable or unsuitable, both leases are occupied, or more than
two parallel delegates are useful, use native Codex subagents immediately
rather than waiting. The same rule applies while the main task needs either
V620 for sLLM GPU work: stop the idle service to reclaim the pair and use Codex.

## Pi profiles and capability boundary

| Profile | Thinking/output | Hard deadline | Intended work |
| --- | --- | ---: | --- |
| Fast | off / 2,048 | 300 s | exact lookup, short summary, tiny mechanical edit |
| Standard | low, 1,024 reasoning / 8,192 | 900 s | targeted inspection, existing-pattern edit, focused test |
| Deep | low, 4,096 reasoning / 8,192 | 3,600 s | bounded multi-file implementation/review and test-fix loop |

Deep may use `--timeout-seconds N` or `--no-timeout` for a progressing task when
the longer runtime is explicitly justified. The 3,600-second default is a hard
runaway bound, not a target duration. Each reasoning budget applies to each
model call; tool-heavy tasks can consume it repeatedly. Large file creation
must therefore be split into write/edit calls below about 12 KB rather than one
tool argument near the model output cap.

Pi exposes `read`, `grep`, `ls`, and Bash; write/edit are conditional. The
installed Pi `find` tool depends on an unavailable `fd` binary and is omitted;
focused `rg` through Bash is available. Recursive subagents, workflows, goals,
web searches, and external connectors are outside this profile. MCP is not
connected by default; filesystem/Git MCP would duplicate native tools and add
schema/context cost. Project `AGENTS.md`/`CLAUDE.md` context discovery remains
enabled, while Pi sessions are ephemeral and startup network operations are
disabled.

The Pi model catalog advertises the actual 491,520-token per-slot server
capacity so compaction does not assume either the combined 983,040-token
reservation or the old 262,144-token native limit. The latter remains the
model's metadata and quality boundary. Per-request output is capped at 8,192
tokens for Standard/Deep and 2,048 for Fast.

Use native Codex instead of forcing Qwen for ambiguous architecture decisions,
security-critical acceptance, external connectors, open-ended multi-repository
work, or work needing more than two simultaneous delegates. Deep Qwen can
implement a medium parser and converge a compile/test loop, but main-agent
review must reconcile exact format/numeric constants with authoritative
sources and inspect complexity at attacker-controlled limits.

## 2026-08-17 Pi comparison

- Fast read-only Phase-20 placement audit completed in 144.92 seconds with 10
  successful tool calls and a final report. Native Codex was faster and more
  precise; Fast is therefore kept narrow.
- Standard metadata-parser implementation reached a 600-second experimental
  deadline unfinished. A 6,144-token cap truncated a 20 KB one-shot write. The
  agent recovered the body but did not integrate or test it. Standard now has
  8,192 output tokens and chunking guidance; large new modules use Deep.
- Deep tensor-table/range implementation completed in 1,605.56 seconds with 57
  tool calls, recovered from tool and compile/test errors, passed 25 focused
  tests and the full crate suite, and produced a final report. Native Codex
  completed the equivalent task in roughly five minutes and passed 27 focused
  tests. Qwen identified that the benchmark task's `NVFP4=32/17` instruction
  conflicted with the authoritative `64/36` project contract; neither artifact
  was merged.
- Independent review found Qwen's range-overlap scan O(n^2) at the 65,536
  tensor bound and its allocation ordering weaker than required. Codex used an
  O(n log n) design but did not flag the same task/project constant conflict.
  This establishes Deep Qwen as a useful bounded implementer and second
  opinion, not an unsupervised acceptance authority.

## DeepSeek Harness compatibility path

DeepSeek Harness 0.1.0-rc.6 and `qwen38-dsh` remain installed for explicit
compatibility/debug work. The former normal Responses route reproduced
duplicate tool IDs and mismatched arguments in both Harness and Pi, while Pi
Chat Completions kept unique IDs and correct arguments. Do not use DSH as the
normal delegate until the common Responses translation issue is fixed. Its
historical single-V620 and Code Mode results below remain evidence about the
old path, not current defaults.

### 2026-08-17 single-V620 Harness profile validation (historical)

- A native read-only task used the bounded repository instructions and returned
  the requested one-sentence answer in 30.85 seconds.
- The equivalent Code Mode task completed in 37.37 seconds and emitted Node's
  experimental TypeScript-stripping warning. Native tools therefore remain the
  default; Code Mode is an optional task-specific control.
- A workspace-write Python task created two allowed files, ran four `unittest`
  cases, and returned a scoped report in 54.87 seconds. The main agent reread
  both files and independently reran all four tests successfully.
- A read-only task's single attempted file creation was denied by the Harness
  sandbox in 12.86 seconds, and the target file did not exist afterward.

Earlier broad Phase-20 repository surveys took more than three minutes and were
interrupted before a final report. The model server was healthy; the dominant
cost was repeated tool exploration and large retained reads. The scoped
persona, smaller read/result bounds, reduced tool catalog, and explicit task
contract address that failure mode. They do not make open-ended surveys an
appropriate local-Qwen task.

## Verified runtime configuration

The operational source of truth is
`/home/homelab1/.local/bin/qwen38-subagent-server`. The 2026-08-17 verified
configuration is:

| Field | Value |
| --- | --- |
| Model | Qwen3.8-27B UD Q5_K_XL GGUF |
| llama.cpp | build 901, commit `4df29be4f4c3673f428170fda944a5b19f743bb8` |
| Backend | HIP with quantized-KV Flash Attention enabled at build time |
| GPU | two V620 `gfx1030` devices, tensor split `1,1` |
| GPU offload | all model layers; no CPU layer offload |
| Runtime context | 491,520 tokens per slot, 983,040 total, two non-unified slots |
| Model native context metadata | 262,144 tokens |
| KV | Q5_1 for target and MTP draft contexts |
| Speculation | MTP, maximum draft width 3 |
| Batch | logical 512, physical 128 |
| Fit policy | automatic fit disabled; context mismatch is a startup failure |

The runtime context deliberately exceeds the model's native context metadata.
The wrapper overrides `qwen35.context_length` so llama.cpp does not cap the
server slot at 262,144. This establishes available capacity, not output-quality
evidence beyond the native window; a task near or above 262,144 tokens must be
described that way.

The prior single-V620 runtime could not fit 524,288 tokens with all layers,
target/draft Q5_1 KV, and MTP width 3. The two-V620 tensor profile did fit
524,288 tokens per slot but left only about 1 GB headroom per GPU. The normal
profile therefore reserves 491,520 tokens per slot, 983,040 total, to increase
headroom while retaining two half-million-class contexts. Do not raise the
context, batch sizes, or MTP width without repeating startup, per-slot actual
context, VRAM/GTT, and two-Harness checks. Reducing the service to one V620 is
not an operational fallback.

### 2026-08-17 TP2 operational validation

- The managed service reported two non-unified slots at 491,520 tokens each,
  matching the 983,040-token command-line total and Harness model catalog.
- Idle headroom after startup was about 2.48 GB per V620. After the two-task
  check it was about 2.40 GB per GPU; GTT stayed near 17/23 MB.
- Two simultaneous read-only Harness processes completed independently in
  26.85 and 24.58 seconds and llama.cpp assigned them to different slots.
- A third wrapper process exited immediately with status 75 instead of
  queueing, and both leases were available again after the first two exited.
- Readiness verifies the managed process identity, pinned V620 pair, required
  TP2/KV/MTP arguments, `/props.total_slots == 2`, and per-slot context. Normal
  status/startup output omits the local endpoint address.
- This PCIe topology still uses meta-backend butterfly after internal
  AllReduce initialization fails. Tensor-split backend sampling is unsupported,
  so token sampling runs on CPU; that warning is not model-layer CPU offload.

## Verification and operation

`ready` exits successfully only when health and actual context match; `status`
shows the same contract for a human. A usable service reports:

```text
context:  491520 per slot, 983040 total configured
slots:    2
split:    tensor 1,1
kv:       q5_1/q5_1
mtp:      3
health:   ready (actual context 491520)
```

The `actual context` value is the per-slot value read from llama.cpp `/props`;
it is not merely the configured command-line total. The service pins both V620
devices, uses `--split-mode tensor --tensor-split 1,1 --parallel 2`, and
disables unified KV so each agent task owns an independent slot. `--fit off`
prevents llama.cpp from silently shrinking unset resources.

Available service commands are:

```bash
/home/homelab1/.local/bin/qwen38-subagent-server start
/home/homelab1/.local/bin/qwen38-subagent-server stop
/home/homelab1/.local/bin/qwen38-subagent-server restart
/home/homelab1/.local/bin/qwen38-subagent-server status
/home/homelab1/.local/bin/qwen38-subagent-server ready
/home/homelab1/.local/bin/qwen38-subagent-server logs
/home/homelab1/.local/bin/qwen38-subagent-server command
```

Do not print the API-key file or expose the localhost port. If delegation fails,
inspect `status` and the last service log lines:

```bash
journalctl --user -u qwen38-subagent.service -n 100 --no-pager
```

Report the exact local failure. Do not silently reduce context, change KV type,
disable MTP, select another GPU configuration, or substitute another hosted
model. When delegation remains useful, use a native Codex subagent instead.

## Main agent acceptance

Treat Pi or DeepSeek Harness output as a subagent report, not as verified completion.
For read-only work, compare its claims with the named files. For edits, inspect
the diff, preserve unrelated user changes, and run checks proportionate to the
affected code. The local Qwen agent may provide an independent view, but using
it is not an integration or release gate.

## Prior multi-GPU measurement and operational promotion

The two-V620, 1,048,576-total-context tensor-split run performed on 2026-08-17
was originally recorded as benchmark-only. A later explicit user decision
promoted the same TP2 shape to the normal local-Qwen path after reducing total
context to 983,040. The old 524,288-per-slot performance and memory values
remain historical evidence rather than measurements of the new
491,520-per-slot profile. Results and limitations are recorded in the [TP2 1M
bounded summary](../../ci/matrix/phase-x-qwen38-v620-tp2-1m-summary-v1.json).

### Multi-GPU profile selection

The 2026-08-17 follow-up compared two independent V620 servers, V620 layer and
tensor splits, and a mixed R9700 plus two-V620 process with the same 11,058-token
coding prompt and two simultaneous 128-token outputs. These were measurement
profiles used to select the later operational policy:

- two independent V620 servers at 368,640 context each had the best aggregate
  throughput and completed the measured requests in 45.58/47.01 seconds;
- V620 x2 tensor split provided exactly 524,288 context per slot, but completed
  in 59.14/60.78 seconds, left less than about 1.01 GB VRAM per GPU, used the
  host-staged butterfly AllReduce path, and sampled on CPU;
- V620 x2 layer split could not start at 1M total context with MTP. Reducing the
  total to 917,504 made it fit, but the two measured requests took 67.31/69.56
  seconds;
- R9700 plus V620 x2 layer split at `5,2,2` provided 524,288 context per slot
  and completed in 45.82/47.90 seconds. It is the best single-process 1M-total
  profile measured here, but remains non-operational because it occupies the
  R9700 used for sLLM development;
- heterogeneous tensor split was slower than the two-V620 tensor profile and
  is rejected. Upstream deprecates row split, so it was not benchmarked.

Pi and DeepSeek Harness continue to target one endpoint. The endpoint owns both
V620 devices and exposes two slots, so no multi-endpoint dispatcher is needed.
Two independent `qwen38-pi` or `qwen38-dsh` processes are required to occupy
both slots; one agent does not split a task across two slots. The wrappers share
two leases so a third Qwen process cannot silently queue. Full comparison results
and limits are in the [multi-GPU selection
summary](../../ci/matrix/phase-x-qwen38-multi-gpu-selection-v1.json).

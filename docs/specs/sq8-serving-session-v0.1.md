# SQ8 Serving Session v0.1

Status: implementation contract for P8-C, amended after the P8-B2 M=128 selection

## 1. Scope

This specification defines the synchronous, single-request SQ8 serving session used
by the OpenWebUI worker. It turns the resident P7 generation core into an
incremental session without weakening the P7 fail-closed behavior.

The v0.1 product envelope is fixed as follows:

- public model ID `ullm-qwen3-14b-sq8`, backed by
  `Qwen/Qwen3-14B-FP8` revision
  `9a283b4a5efbc09ce247e0ae5b02b744739e525a`;
- canonical artifact content SHA-256
  `2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147`;
- thin-package manifest SHA-256
  `c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb`;
- one Radeon AI PRO R9700 (`gfx1201`) using profile
  `rdna4_w8a8_block_ck`;
- one active request (`active = 1`);
- no waiting request (`waiting = 0`);
- context length 4096 tokens;
- default `max_new_tokens = 256`, hard maximum 512;
- one HIP context, one model instance, one inference stream, and one inference
  thread;
- synchronous GPU calls only; and
- M=1 prompt processing as the correctness oracle and selected M=128 cached-prefix
  prompt processing with an M=1 tail in the product worker.

The shorthand for this admission boundary is `active1 / waiting0`.

This document does not define HTTP, OpenAI-compatible request conversion,
tokenization, chat templates, stop-string matching, automatic truncation, a wait
queue, request batching, speculative decoding, or multi-GPU execution.

## 2. P7 compatibility boundary

The existing P7 schemas and their validators are immutable under P8:

- `ullm.sq8.generation.v1`; and
- `ullm.sq8.generation_benchmark.v1`.

P8 may call shared internal kernels or loaders, but it MUST NOT change a P7 schema
version, field, field meaning, validation rule, evidence filename, or fixed P7
request behavior. A P7 command MUST continue to use its existing fixed prompt,
request validation, completion validation, and reset validation. Serving-only
state or fields MUST NOT be serialized into a P7 result.

The serving session is a distinct API. Passing a serving request through the P7
fixed-request API, or relaxing P7 validation to accept a serving request, is
forbidden.

## 3. Terms and counters

- **Prompt token**: a token supplied in `prompt_token_ids`.
- **Generated token**: a sampled token returned to the worker.
- **Execution unit**: one synchronized M=1 step or one synchronized M=128 prompt
  chunk.
- **Cache length**: the number of token positions already written to every
  layer's paged KV cache.
- **Baseline**: scheduler has no request, allocator usage equals the post-load
  baseline, every layer cache has length zero, and no request-owned buffers or
  metadata remain.
- **Mutation**: any scheduler allocation, request registration, session state
  transition away from `Ready`, cache write/reset, model enqueue, sampling RNG
  advance, or request counter update.

For prompt length `P` and `G` already emitted generated tokens:

- after prompt ingestion, cache length is `P`;
- before the first generated token is emitted, the final prompt hidden state is
  sent to the model head;
- after `G >= 1` generated tokens are emitted, cache length is `P + G - 1`; and
- the last emitted token is not written to KV unless another decode step consumes
  it.

All counters use checked integer arithmetic. Overflow is a validation or fatal
invariant error, never wrapping behavior.

## 4. Public data contract

The concrete Rust names may remain crate-local during P8, but the API MUST expose
the following semantics:

```rust
pub struct Sq8ServingRequest {
    pub request_id: String,
    pub prompt_token_ids: Vec<usize>,
    pub max_new_tokens: usize,
    pub eos_token_ids: Vec<usize>,
    pub sampling: Sq8SamplingParams,
}

pub struct Sq8SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: i64,
}

#[derive(Clone)]
pub struct Sq8CancellationToken {
    inner: std::sync::Arc<Sq8CancellationState>,
}

struct Sq8CancellationState {
    flag: std::sync::atomic::AtomicBool,
    publication: std::sync::Mutex<()>,
}

pub enum Sq8ServingAdvance {
    PromptProgress {
        prompt_tokens_processed: usize,
        cache_len: usize,
        execution_width: usize,
    },
    Token {
        token_id: usize,
        generated_index: usize,
        cache_len: usize,
        terminal_reason: Option<Sq8FinishReason>,
    },
    CancellationObserved,
}

pub struct Sq8PreparedToken {
    pub token_id: usize,
    pub generated_index: usize,
    pub cache_len: usize,
    pub terminal_reason: Option<Sq8FinishReason>,
}

pub enum Sq8PreparedAdvance {
    PromptProgress {
        prompt_tokens_processed: usize,
        cache_len: usize,
        execution_width: usize,
    },
    Token(Sq8PreparedToken),
    CancellationObserved,
}

pub enum Sq8FinishReason {
    Stop,
    Length,
}

pub enum Sq8ReleaseOutcome {
    Stop,
    Length,
    Cancelled,
}

pub struct Sq8ReleaseSummary {
    pub request_id: String,
    pub outcome: Sq8ReleaseOutcome,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub reset_complete: bool,
}
```

The synchronous session surface is:

```rust
impl Qwen3Sq8ServingSession {
    pub fn start(
        &mut self,
        request: Sq8ServingRequest,
        cancel: Sq8CancellationToken,
        stream: &mut RuntimeStream,
    ) -> Result<(), Sq8ServingError>;

    pub fn advance_synchronized(
        &mut self,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8ServingAdvance, Sq8ServingError>;

    pub fn prepare_advance_synchronized(
        &mut self,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8PreparedAdvance, Sq8ServingError>;

    pub fn publish_prepared_token(
        &mut self,
        token: Sq8PreparedToken,
        stream: &mut RuntimeStream,
        publish: impl FnOnce(&Sq8PreparedToken) -> Result<(), String>,
    ) -> Result<Sq8ServingAdvance, Sq8ServingError>;

    pub fn finish_and_reset_synchronized(
        &mut self,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8ReleaseSummary, Sq8ServingError>;

    pub fn abort_and_reset_synchronized(
        &mut self,
        stream: &mut RuntimeStream,
    ) -> Result<Sq8ReleaseSummary, Sq8ServingError>;
}
```

`advance_synchronized` executes at most one execution unit. It MUST synchronize
that unit before returning. It MUST NOT hide an entire prompt loop inside one
call, because cancellation and progress must be observable between units.

The worker uses the prepared-token surface. Preparing a token may run the model
head and create a transactional sampling proposal, but it does not advance the
request RNG, scheduler generated count, generated-token count, feedback token, or
terminal state. `publish_prepared_token` acquires the cancellation token's
publication mutex, performs a final cancellation check, invokes and requires the
publisher callback to flush successfully, then commits the pending state before
unlocking. Cancellation observed before the callback discards the proposal,
consumes no RNG draw, advances no generated counter, and enters `Cancelling`.
The legacy one-call `advance_synchronized` may remain as an internal
prepare-and-immediate-commit convenience for non-protocol P8-B evidence, but the
resident worker MUST NOT use it.

The implementation may provide a convenience `next_token_synchronized` wrapper,
but the worker MUST use the one-unit advance surface.

## 5. Thread and ownership contract

The inference thread MUST construct, own, use, and destroy all of the following:

- HIP context and stream;
- resident model and model-head runtime;
- layer and stack workspaces;
- paged KV caches and block tables;
- scheduler and device allocator handles; and
- the `Qwen3Sq8ServingSession`.

None of those values may cross a thread boundary. Their non-`Send` status MUST
not be bypassed with an unsafe wrapper.

Only `Sq8CancellationToken` may cross from the command-reader thread to the
inference thread. Request commands and immutable owned CPU data may be transferred
through a bounded channel before they become session-owned.

There is exactly one stream operation in flight. A session method may return only
after its requested GPU work has synchronized successfully or after it has
entered `Failed`.

## 6. State machine

The required states are:

```text
Loading -> Ready

Ready -> Prefilling -> Decoding -> Finishing -> Resetting -> Ready
Ready -> Prefilling -> Finishing -> Resetting -> Ready
Ready -> Prefilling/Decoding -> TokenPrepared -> Decoding/Finishing -> Resetting -> Ready
Ready -> Prefilling/Decoding -> TokenPrepared -> Cancelling -> Resetting -> Ready
Ready -> Prefilling/Decoding -> Cancelling -> Resetting -> Ready

Loading/Prefilling/Decoding/TokenPrepared/Finishing/Cancelling/Resetting -> Failed
```

`Loading` belongs to construction. A successfully constructed serving session is
`Ready`. `Failed` is terminal; no method may make a failed session ready again.
The worker must terminate the process and rely on a fresh process to reload the
model.

| State | Permitted operation | Resulting state |
| --- | --- | --- |
| `Ready` | `start(valid_request, token)` | `Prefilling` |
| `Ready` | rejected `start` | `Ready`, baseline unchanged |
| `Prefilling` | prompt execution unit, prompt remains | `Prefilling` |
| `Prefilling` | final prompt unit prepares first token | `TokenPrepared` |
| `Decoding` | decode unit prepares next token | `TokenPrepared` |
| `TokenPrepared` | publish and flush nonterminal token, then commit | `Decoding` |
| `TokenPrepared` | publish and flush terminal token, then commit | `Finishing` |
| `TokenPrepared` | cancellation wins before publication | `Cancelling` |
| `Prefilling`/`Decoding` | cancellation observed | `Cancelling` |
| `Finishing` | `finish_and_reset_synchronized` begins | `Resetting` |
| `Cancelling` | `abort_and_reset_synchronized` begins | `Resetting` |
| `Resetting` | all reset invariants pass | `Ready` |
| any mutable state | unexpected error or invariant failure | `Failed` |

Calling a method from any other state is an error. An invalid-state call MUST NOT
attempt a repair reset. It transitions to `Failed` if request-owned mutation may
already exist; an invalid call against a demonstrably untouched `Ready` session
may return a validation error and stay `Ready`.

## 7. Validation before mutation

`start` MUST complete all validation below before the first mutation:

1. Session state is exactly `Ready`.
2. Scheduler, allocator, every layer cache, request slot, counters, and sampling
   state satisfy the baseline invariant.
3. Loaded model identity, thin-package identity, architecture, device identity,
   kernel profile, vocabulary size, layer count, KV geometry, context limit, and
   block-table geometry match the immutable runtime configuration.
4. `request_id` is valid under the worker protocol and is not the active ID.
5. `prompt_token_ids.len()` is in `1..=4096`.
6. Every prompt token is less than vocabulary size.
7. `max_new_tokens` is in `1..=512`.
8. `prompt_token_ids.len() + max_new_tokens` is at most 4096 using checked
   addition.
9. `eos_token_ids` is nonempty, contains no duplicate, every value is less than
   vocabulary size, and it matches the immutable model generation configuration.
   The product value is `[151645, 151643]` in that order.
10. `temperature` and `top_p` are finite; `0 <= temperature <= 2`;
    `0 < top_p <= 1`; and `top_k` equals the immutable product value 20.
11. The selected prompt execution mode is enabled and has enough preallocated
    workspace. No request-time allocation may silently select another kernel.

A failure in this list is `InvalidRequest` or `InvalidConfiguration`. It MUST
leave the session byte-for-byte equivalent to its prior `Ready` baseline,
including scheduler and RNG state. No reset is required, and no `released` event
is produced for the rejected request.

After validation succeeds, `start` may atomically assume ownership of the request,
claim its scheduler entry and fixed block table, initialize its counters and RNG,
and enter `Prefilling`. Partial acquisition is forbidden: if acquisition cannot be
completed, the session must prove that it restored the baseline or enter `Failed`.

## 8. Fixed cache and scheduler boundary

The v0.1 cache uses 16-token blocks and a maximum context of 4096, so each layer
has exactly 256 logical blocks available to the single active request. Its block
table is the identity mapping `0..255` for the request lifetime.

Because `active = 1` and `waiting = 0`:

- a scheduler entry represents only the active request;
- no cache block belongs to a second request;
- no block is reordered, swapped, evicted, or shared in v0.1;
- a fixed M=128 prompt chunk is 128 consecutive positions from that same request,
  not a batch of requests; and
- release must remove the scheduler entry before the session reports baseline.

Every layer MUST have the same cache length at each synchronized boundary. A
length mismatch is fatal.

## 9. M=1 oracle prompt path

The M=1 path is the mandatory correctness path and remains available after M=128
is selected.

For prompt position `p` in `0..P`:

1. Load prompt token `prompt_token_ids[p]` into the resident M=1 input buffer.
2. Execute the full stack with RoPE position `p`.
3. Attend to cached positions `0..p` inclusive, causally.
4. Write the new K/V at cache position `p` for every layer.
5. Synchronize and validate every layer cache length equals `p + 1`.
6. Return `PromptProgress` if `p + 1 < P`.
7. If this is the final prompt token, apply final norm and the model head to that
   hidden row, sample generated index zero, and return `Token`.

The first prompt token initializes an empty cache. Later prompt tokens use the same
paged decode attention contract as generation, except their input token comes from
the prompt.

The head MUST run only for the final prompt position. Intermediate prompt logits
must not be read back or sampled.

## 10. M=128 cached-prefix prompt path

M=128 is an optimization of a single request's prompt. It is selected only after
the M=1 and M=8 oracle gates and the unchanged formal performance gates in section
17 pass.

In `M128ChunksWithM1Tail` mode:

- each complete 128-token prompt range `[p, p + 128)` uses one M=128 execution;
- `p` is always the current cache length;
- query row `i` uses absolute RoPE position `p + i`;
- query row `i` attends to all cached prefix keys `0..p` and current-chunk keys
  `p..=p+i`;
- the 128 new K/V rows are written to positions `[p, p + 128)`; and
- after synchronization, every layer cache length is exactly `p + 128`.

The first M=128 chunk has an empty prefix. Every later M=128 chunk has a nonempty
cached prefix. The current P7 `PrefillPaged` operation, which requires an empty
cache and attends only within the current causal chunk, MUST NOT be reused for a
nonempty-prefix chunk. A distinct cached-prefix attention mode is required and
must fail closed if its cache position is not the current written length.

If fewer than 128 prompt tokens remain, process the tail as consecutive M=1
steps. Generated decode is always M=1 in v0.1. When the prompt length is a multiple
of 128, the head consumes row 127 of the final M=128 output; otherwise it
consumes the final M=1 tail output.

The implementation MUST NOT pad a prompt to 128 tokens, combine two requests, or
let an M=128 chunk cross the request's prompt boundary.

## 11. Generated-token semantics

Generated index zero is sampled from the final prompt hidden state. For generated
index `g > 0`, the session:

1. checks cancellation;
2. embeds generated token `g - 1`;
3. runs M=1 stack decode at absolute position `P + g - 1`;
4. writes that input token to KV;
5. runs final norm and model head;
6. samples token `g`; and
7. synchronizes and validates the counters before returning it.

A token's `generated_index` starts at zero and increases by exactly one. The token
is terminal with `Stop` when it is in `eos_token_ids`; otherwise it is terminal
with `Length` when its emission makes the generated count equal
`max_new_tokens`. EOS wins when both conditions hold.

Once a terminal token is returned, the session enters `Finishing`. No further
`advance_synchronized` call is valid.

## 12. Sampling contract

Sampling happens on the inference thread after a synchronized logits readback.
Non-finite logits are fatal.

The product defaults resolved by the gateway are `temperature = 0.6`,
`top_p = 0.95`, `top_k = 20`, an OS-random signed 64-bit seed, and EOS IDs
`[151645, 151643]`. The worker command always carries these effective values;
defaults are not applied implicitly inside the session.

- `temperature == 0` means greedy argmax over the full vocabulary. Ties choose
  the lowest token ID. `top_p`, `top_k`, and `seed` do not alter greedy output.
- For `temperature > 0`, divide logits by temperature, select the `top_k` highest
  values with token-ID ascending as the tie break, compute stable softmax, then
  keep the shortest probability-descending prefix whose cumulative probability
  is at least `top_p`. Keep at least one token and renormalize.
- The stochastic draw uses one request-local `ChaCha8Rng`. The signed 64-bit
  `seed` is mapped to its two's-complement `u64` bit pattern before seeding. The
  crate version is pinned by `Cargo.lock`. Exactly one uniform draw is consumed
  per emitted stochastic token and none for validation, prompt progress,
  cancellation, greedy sampling, or reset.

The probability calculation uses `f64` CPU intermediates. The selected token is
validated against vocabulary size before it is recorded or emitted.

## 13. Atomic cancellation

`Sq8CancellationToken::cancel()` stores `true` with `Ordering::Release`.
The inference thread loads it with `Ordering::Acquire`:

- before the first GPU mutation after `start`;
- before every M=1 or M=128 execution unit;
- immediately after each synchronized unit and before publishing its progress or
  token; and
- before sampling and immediately before publishing a sampled token.

Cancellation is monotonic and idempotent. It cannot be cleared or reused for
another request.

If cancellation is observed after a GPU unit completes but before its token is
published, the unpublished token is discarded, generated counters and RNG state
must reflect only published tokens, and the session enters `Cancelling`. Cache
mutation performed by the completed unit is allowed because the subsequent abort
resets the entire request.

After the inference thread observes cancellation, it MUST NOT publish another
token. It returns `CancellationObserved`, and the caller must invoke
`abort_and_reset_synchronized` exactly once.

If a token event is completely written before the cancel store occurs, that token
is valid. This event boundary is the only completion/cancellation race rule.

Cancellation is checked between execution units; it cannot preempt a running HIP
kernel. Therefore observation latency is bounded by one synchronized execution
unit, not by a fixed wall-clock duration.

## 14. Finish, abort, and reset

`finish_and_reset_synchronized` is valid only in `Finishing`.
`abort_and_reset_synchronized` is valid only in `Cancelling`. Both operations:

1. enter `Resetting`;
2. preserve the terminal outcome and final published counters in CPU metadata;
3. release the scheduler request;
4. synchronously zero/reset all 40 layer K/V caches;
5. reset cache lengths, block-table request ownership, prompt/decode counters,
   resident request buffers, sampling state, and cancellation ownership;
6. verify scheduler emptiness and allocator baseline;
7. verify every cache length is zero and all request-owned metadata is absent;
8. construct the release summary from the preserved CPU metadata; and
9. enter `Ready` only after every check passes.

The release summary MUST set `reset_complete = true`. A result with
`reset_complete = false` is forbidden; failed cleanup is fatal instead.

No request may be accepted between the start of terminal cleanup and successful
return. The worker may publish `released` only after the method returns and the
session reports `Ready`.

## 15. Error classes and fatal behavior

Errors are divided into the following classes:

- `InvalidRequest`: detected before mutation; session remains `Ready`.
- `Busy` or `InvalidState`: rejected by the worker or before request mutation;
  session remains unchanged only when baseline is proven.
- `Cancelled`: expected terminal path through `Cancelling` and reset.
- `FatalRuntime`: HIP, kernel, synchronization, loader, scheduler, cache, sampling,
  or invariant failure after mutation might have occurred.

A `FatalRuntime` error immediately puts the session in `Failed`. The implementation
MUST NOT attempt to continue, accept a new request, emit a successful release, or
claim that reset completed. The worker emits a fatal error when possible and exits
nonzero.

## 16. Release deadlines

Cancellation observation remains bounded by one synchronized execution unit. From
the reader's matching cancel validation and atomic flag store to the protocol
reader observing a flushed `released(reset_complete=true)`, the measured release
latency p95 MUST be at or below 2000 ms. This is the performance target.

The worker arms a 5000-ms terminal cleanup watchdog when the inference thread
enters `Resetting`, for normal and cancelled requests. No separate longer cleanup
allowance applies. Independently, the gateway enforces a hard 5000-ms deadline
from sending a matching cancel command to observing matching `released`. The
earlier deadline wins. If a verified release is not available in time:

- the process is considered poisoned;
- no `released` event may be emitted;
- the watchdog terminates the process nonzero, even if the inference thread is
  blocked in HIP; and
- recovery requires a new worker process and a fresh model load.

The watchdog does not move or inspect HIP-owned values. It observes only an
internal reset generation counter or completion notification. Startup, complete
request, and no-progress deadlines are gateway/worker-protocol concerns and are
defined in `sq8-worker-protocol-v0.1.md`.

## 17. Required gates

Before M=1 serving is accepted:

1. The new M=1 session reproduces the existing P7 prompt `[1,2,3,4,5,6,7,8]` for
   all eight fixed generation steps.
2. Its final prompt hidden state and logits match the P7 M=8 prompt path within
   the frozen numerical tolerances, and top-1 is exact.
3. Prompt lengths 1, 8, 32, and 128 match the frozen vLLM token oracle under
   greedy decoding.
4. Normal finish, EOS finish, validation rejection, and cancellation all prove
   the baseline invariants.

Before M=8 cached-prefix serving is enabled:

1. Prompt lengths 8, 9, 16, 17, 32, and 128 match all-M=1 serving within the
   frozen hidden/logit tolerances and exact token sequence.
2. The same prompts match the frozen vLLM greedy token oracle.
3. Tests prove that later M=8 chunks attend to both cached prefix and their own
   causal chunk.
4. Tests prove cache length is identical across all layers after every execution
   unit.
5. Cancellation at every unit boundary returns to baseline without emitting a
   post-observation token.

The M=8 result remains the frozen chunk oracle. The product-selected M=128 path
additionally requires prompt 32/128/512/4095 comparison against all-M1 and the
source oracle, the exact 3584+512 deep boundary, and the unchanged formal
TTFT/decode matrix. The evidence under
`benchmarks/results/2026-07-10/sq8-serving-chunks-v0.1/` satisfies those gates.

## 18. Observability invariant

Internal progress returned by the session is evidence of a completed,
synchronized unit only. It is not permission to reuse the session. Only a
successful reset summary proves reuse readiness. The worker mapping from session
results to JSONL events is defined by `sq8-worker-protocol-v0.1.md`: protocol
`progress` is emitted after each complete M=128 prompt chunk and once at the
prefill/decode transition. M=1 tail steps do not each emit protocol progress.

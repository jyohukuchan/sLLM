# SQ8 Serving Session v0.2

Status: ratified production contract

Date: 2026-07-24

## 1. Version and scope

This specification incorporates every model, GPU, active1/waiting0,
thread-ownership, cache, scheduler, prompt-path, cancellation, watchdog, and
fatal-error rule from `sq8-serving-session-v0.1.md`. It adds the reasoning
state and accounting required by `ullm.served_model.v2` and
`ullm.worker.v2`.

The frozen v0.1 API and `ullm.worker.v1` behavior are not reinterpreted. A v1
session has no reasoning dialect, no reasoning execution, and no reasoning
release counters.

The v0.2 production target is independent `SQ8_0` for
`Qwen/Qwen3-14B-FP8`, not an AQ4 partial-tensor overlay.

## 2. Public request and summary

The v0.1 serving request gains one required execution value when the loaded
worker profile is v2:

```rust
pub struct ReasoningExecution {
    pub enabled: bool,
    pub budget_tokens: Option<usize>,
    pub dialect_id: String,
    pub end_sequence: Vec<usize>,
    pub forced_end_sequence: Vec<usize>,
    pub reserved_answer_tokens: usize,
}
```

`None` is valid only for a v1 profile. A v2 profile and request must both have
reasoning data; a v1 profile and request must both omit it. The execution
dialect, end sequences, and reservation exactly equal the loaded manifest.
The manifest start sequence and the request/manifest natural and forced end
sequences each have length exactly one. The `Vec` representation is retained
only as an internal API shape; it does not authorize multi-token v2
delimiters. The budget is `None` or an integer in
`0..=max_budget_tokens`.

The release summary additionally carries:

```rust
pub struct ReasoningUsage {
    pub reasoning_tokens: usize,
    pub forced_end_tokens: usize,
}
```

It is present for every v2 request, including disabled and cancelled requests,
and absent for every v1 request.

## 3. State and counters

The request-local phase is one of:

```text
Disabled
Reasoning
ForcingEndSequence
Answer
Finished
Cancelled
```

The session retains v0.1 `Ready`, `Prefilling`, `Decoding`,
`TokenPrepared`, terminal, reset, and failed states. Reasoning phase is nested
request state; it does not weaken the outer prepare/publish/commit boundary.

Request-local counters are:

- committed reasoning-body tokens;
- committed forced-end tokens;
- committed answer tokens;
- committed sampled tokens; and
- total committed completion tokens.

Total completion tokens equal the raw published token-event count. Reasoning
and forced counters are disjoint subsets and their sum cannot exceed total
completion. A natural delimiter is neither reasoning-body nor forced usage.

## 4. Validation before mutation

In addition to v0.1 validation, `start` verifies before mutation:

1. profile and request worker schemas are exactly v2;
2. the loaded reasoning dialect is valid for the loaded vocabulary and each
   start, natural-end, and forced-end sequence contains exactly one token;
3. request dialect, end sequence, forced sequence, and reservation exactly
   match that dialect;
4. a bounded budget does not exceed the manifest maximum; and
5. `max_new_tokens` can contain every required forced token and the positive
   answer reservation.

The manifest-level maximum budget plus the forced token plus answer
reservation must also fit the manifest completion maximum. Failure leaves
scheduler, cache, counters, and sampler RNG at the `Ready` baseline and emits
no release.

## 5. Phase transitions

Disabled reasoning starts in `Disabled`; every sampled token is an answer
token and release usage is `0/0`.

Enabled reasoning normally starts in `Reasoning`. A zero budget starts in
`ForcingEndSequence`. During reasoning:

- a sampled non-delimiter token becomes committed reasoning body only after
  its publication succeeds;
- the natural end token changes to `Answer`;
- reaching the hard budget changes to `ForcingEndSequence`;
- the length guard forces close before the remaining completion capacity
  would fall below the forced token plus answer reservation; and
- with EOS policy `close`, a reasoning-phase EOS proposal is discarded and
  replaced in the same completion slot by the first forced token.

The complete natural delimiter is suppressed from user-visible reasoning or
answer content by the gateway. It remains sampled raw worker output. A forced
delimiter is also suppressed from user-visible content, but is raw worker
output and forced usage.

After the forced token commits, the phase is `Answer`. At least
`reserved_answer_tokens` capacity remains. Normal EOS/length terminal
precedence then follows v0.1.

## 6. Prepare, publish, and commit

Each sampled proposal is evaluated against a cloned tentative reasoning state.
The active state, completion counters, and sampler state remain unchanged
until the matching token event has been completely published.

Forced tokens are known before model-head sampling whenever the phase or
length guard already requires them. In that case the session skips model-head
sampling. A forced token consumes no sampler draw and does not commit sampler
state.

If a sampled reasoning EOS requests close, any model-head proposal and
tentative RNG advance are discarded before forced replacement. The sampled
EOS is not published or counted.

Publication failure, cancellation before publication, or terminal poison
drops the tentative transition. Cache mutation performed by an already
completed GPU unit is permitted only because the required abort path resets
the entire request.

## 7. Cancellation, release, and reset

Cancellation retains v0.1 atomic race semantics. After cancellation wins:

- no further token is published;
- any pending reasoning/forced transition is discarded;
- only previously committed usage is preserved; and
- reasoning phase becomes `Cancelled`.

Before GPU/session reset, terminal cleanup copies committed completion and
reasoning usage into CPU release metadata. Reset clears reasoning state,
pending delimiter state, all counters, sampler state, cache, scheduler, and
request ownership. Only after reset invariants pass may the preserved summary
be returned and a `released(reset_complete=true)` event be published.

A reused session must begin with fresh `0/0` reasoning accounting. Reset
failure remains fatal and cannot produce a successful release.

## 8. Required conformance

CPU conformance covers exact profile schema selection, multi-token v2 profile
and request rejection, explicit disabled execution, budgets
`0/32/128/256`, unbounded natural/forced/EOS close, reservation,
transactional cancellation/publication rollback, release/reset preservation,
reuse reset, and forced-token RNG nonconsumption.

The standalone physical-GPU gate is
`sq8-worker-acceptance-v0.3.md`. Fresh OpenWebUI evidence is governed by
`sq8-openwebui-release-v0.2.md`. Neither document authorizes activation by
itself.

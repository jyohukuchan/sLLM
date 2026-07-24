# uLLM worker JSONL protocol v0.2

Status: ratified production contract

Date: 2026-07-24

## 1. Version boundary

`ullm.worker.v2` incorporates the bounded JSONL framing, strict parsing,
active1/waiting0 admission, cancellation, ordered output, release flush,
deadlines, poison behavior, and reset rules of
`sq8-worker-protocol-v0.1.md`. It adds explicit model-independent reasoning
execution and release accounting.

A worker loaded from `ullm.served_model.v2` accepts only `ullm.worker.v2`
generate, cancel, and shutdown commands and emits only `ullm.worker.v2`
events. Exact profile-schema equality is checked before Busy admission or
control dispatch. A v1 worker rejects v2, and a v2 worker rejects v1. Any
future compatibility mode must be a separately specified launcher mode; the
SQ8 production profile has none.

The frozen v1 command/event shapes do not gain reasoning fields.

## 2. Generate command

The generate command has exactly:

```json
{
  "schema_version": "ullm.worker.v2",
  "type": "generate",
  "request_id": "req-1",
  "prompt_token_ids": [1, 2, 3],
  "max_new_tokens": 256,
  "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 20, "seed": 7},
  "eos_token_ids": [151645, 151643],
  "reasoning": {
    "enabled": true,
    "budget_tokens": 128,
    "dialect_id": "qwen3-thinking-v1",
    "end_token_ids": [151668],
    "forced_end_token_ids": [151668],
    "reserved_answer_tokens": 1
  }
}
```

The reasoning object is required even when `enabled` is false and has exactly
the six fields shown. `budget_tokens` is JSON `null` for unbounded reasoning
or a nonnegative integer no greater than the loaded manifest maximum. A zero
budget requests immediate forced close. `end_token_ids` and
`forced_end_token_ids` MUST each contain exactly one token. The dialect ID,
both single-token arrays, and answer reservation MUST exactly equal the loaded
manifest. A multi-token delimiter requires a future worker schema and is
rejected during command decoding. The complete request MUST leave room for
the forced token and
`reserved_answer_tokens`; otherwise it is rejected before generation.

Cancel and shutdown retain their v1 field sets with the v2 discriminator.
Unknown, missing, duplicate, wrongly typed, mixed-version, and mutated
post-inspection fields fail closed.

## 3. Execution and publication

When disabled, generated tokens are answer tokens and reasoning usage remains
zero. When enabled, the worker starts in the manifest's initial phase and
tracks reasoning, forced-close, and answer tokens in request-local state.

A naturally sampled end token changes to answer phase without
exposing the delimiter as user-visible content. The sampled delimiter remains
a sampled raw worker token and is counted in completion usage, but not in
reasoning-body or forced-end usage.

At a hard budget, answer-reservation guard, or reasoning-phase EOS, the worker
publishes the configured forced token through the same
prepare/publish/commit boundary as sampled output. A reasoning-phase EOS
proposal is replaced in that completion slot by the forced token: the
sampled EOS is neither published nor counted and does not advance committed
sampler RNG. The forced token is counted in completion and forced-end usage,
not reasoning-body usage, and does not consume sampler RNG.

Reasoning/accounting state commits only after successful token publication.
Cancellation or publication failure discards tentative transitions. Release
captures committed accounting before reset; the next request starts from zero.

## 4. Released event

Every v2 `released` event has the corresponding frozen v1 fields plus both:

```json
{
  "reasoning_tokens": 0,
  "forced_end_tokens": 0
}
```

The fields are nonnegative integers, are required together for all v2
outcomes including disabled and cancelled requests, and their sum MUST NOT
exceed `completion_tokens`. Disabled reasoning reports `0/0`.

`reasoning_tokens` counts committed reasoning-body tokens only.
`forced_end_tokens` counts committed forced delimiter tokens only. Natural
delimiter tokens, answer tokens, discarded proposals, and unpublished tokens
belong to neither counter.

For a non-cancelled enabled request closed forcibly, completion usage MUST
still contain at least `reserved_answer_tokens` after
`reasoning_tokens + forced_end_tokens`. `reset_complete` remains true only
after successful reset. Cancelled releases retain no timings; non-cancelled
release timing rules are unchanged.

## 5. Conformance

Conformance requires CPU tests for exact schema selection, multi-token profile
and command rejection, disabled `0/0`, budgets `0/32/128/256`, unbounded
natural and forced closure, EOS replacement, answer reservation,
cancellation/publication rollback, release/reset accounting, worker reuse, and
forced-token RNG nonconsumption. GPU acceptance uses
`ullm.sq8.worker_acceptance.raw.v3`; archived raw.v1/raw.v2 remain validated
only under their frozen worker-v1 contracts.

# SQ8 OpenWebUI Product Release Evidence v0.2

Status: ratified production contract

Date: 2026-07-24

## 1. Version boundary

This contract incorporates the sequential active1/waiting0 workload,
cancellation phases, resource soak, TTFT/decode schedules, lifecycle journal,
HTTP/SSE reconstruction, browser evidence, statistics, and fail-closed
publication rules from `sq8-openwebui-release-v0.1.md`.

It replaces only the candidate and protocol identity boundary for independent
`SQ8_0`:

- candidate manifest `ullm.served_model.v2`;
- worker protocol `ullm.worker.v2`;
- standalone acceptance raw/validation v3;
- model identity `ullm.sq8.full_campaign.model_identity.v2`; and
- independent full report `ullm.sq8.openwebui_release.validation.v2`.

The frozen v0.1 release schema remains a historical worker-v1 campaign. No v2
validator may infer missing v2 identity from v0.1 bytes.

## 2. Exact candidate binding

One immutable candidate manifest is supplied before the run. The campaign
records and independently rehashes its exact bytes, worker binary, tokenizer,
artifact, package, promotion receipt, source identity, format, and reasoning
dialect.

Before every GPU-mutating campaign stage, the runner reads the actual active
manifest bytes and requires byte-for-byte equality with that candidate.
Comparing only model ID, path, parsed JSON, or digest copied from producer
state is insufficient. Any mismatch aborts the campaign.

Process identity is derived only from the validated candidate's worker binary.
The process executable is rehashed through `/proc/<pid>/exe`; a caller-supplied
process basename is not authoritative.

The complete campaign manifest includes an exact copy of the candidate
manifest and hashes every raw and derived component. It is immutable and
published without replacement.

## 3. Reasoning and HTTP behavior

Requests and responses follow the versioned OpenAI chat reasoning contract.
The campaign covers disabled reasoning and enabled Qwen3 reasoning with
bounded effort and unbounded execution. It verifies:

- public reasoning/answer separation;
- natural and forced delimiter suppression;
- budgets and answer reservation;
- completion, reasoning, and forced-end usage reconciliation;
- stop, length, cancellation, and disconnect paths;
- immediate post-cancellation recovery; and
- no usage/accounting carry-over after reset or worker restart.

Raw worker-v2 releases always contain reasoning and forced counters. The
gateway reconstructs public usage from worker-authoritative committed
accounting and rejects mismatch.

Fresh OpenWebUI browser cases require a real browser-login session JWT. A
synthetic API key or historical JWT does not satisfy this gate.

## 4. Campaign order and rollback

The entire AQ4/SQ8 sequence runs only inside the separately authorized locked
transaction. The transaction fixes authorization expiry, candidate, rollback
manifest, run/output locations, and required stages before claiming the
authorization exactly once.

Candidate activation during the temporary window is not final activation.
Whether any stage succeeds, fails, or is interrupted, the transaction restores
the exact prior AQ4_0 manifest bytes, restarts/reconciles through the authorized
operational path, and records final health/model observations. Authorization
and claim remain consumed after failure.

This release contract does not change the existing AQ4 bootstrap state or AQ4
bundle-v1 semantics.

## 5. Required evidence

The v0.2 evidence retains every applicable v0.1 raw workload component and adds
or versions the identity-bearing components required to prove:

- candidate manifest bytes at each stage;
- worker-v2 and served-model-v2 identity;
- standalone worker acceptance v3;
- reasoning/accounting outcomes;
- real OpenWebUI session identity without publishing the JWT;
- manifest-derived process identity;
- exact rollback bytes and reverse reconciliation; and
- complete candidate-bound campaign hashing.

The independent validator recomputes HTTP/SSE, browser, lifecycle, resource,
identity, stage-order, active-byte, and rollback gates from raw evidence. Its
successful report schema is
`ullm.sq8.openwebui_release.validation.v2`; it is not an input to its own
pass/fail decision.

The model-campaign evidence consumed by the release bundle is
`ullm.sq8.full_campaign.model_identity.v2`. The bundle also carries the
campaign manifest and the independently recomputed validation-v2 report as
separate hash-bound artifacts.

## 6. Publication and activation boundary

All candidate, acceptance, campaign, validator, authorization, claim, outcome,
receipt, and bundle outputs use atomic no-replace publication followed by
stable rehash. An existing target is never overwritten.

Successful fresh AQ4 and SQ8 campaigns permit assembly of
`ullm.generic_reasoning_release_bundle.v2`; they do not themselves perform
final activation. Final SQ8 activation consumes that independently validated
bundle through the explicit activation procedure only after human review.

The campaign and final activation are intentionally not executed while GPU
approval or the real OpenWebUI JWT is unavailable.

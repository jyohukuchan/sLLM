# AQ4 reasoning and OpenWebUI release contract v0.1

Status: current production candidate active with complete core release evidence; OpenWebUI manifest metadata and exact custom-budget UI reconciliation remain

This document defines the release boundary for Qwen3.5 9B AQ4 reasoning on the
R9700 resident worker. It does not change the OpenWebUI image. The current v2
candidate was restored through complete bundle-bound activation after the
identity-matched comparison window. Phase 0, core browser reasoning, 100-chat
stability, HTTP/SSE accounting, and formal p95 gates are complete. The remaining
operational follow-up is to reconcile the OpenWebUI managed-model manifest hash
and prove the exact `thinking_budget_tokens` custom parameter in the live UI.

## 1. Bound identity

An activation candidate MUST bind all of the following through the generated
`ullm.served_model.v2` manifest:

- Qwen3.5 AQ4 public model identity and context/completion limits;
- tokenizer files, chat-template hash, and `enable_thinking` request option;
- `ullm.worker.v2`, worker binary hash, required HIP guards, and `gfx1201` /
  `rdna4_aq4_resident` identity;
- the declared `ReasoningDialect`, including token sequences, effort budgets,
  EOS policy, history policy, and answer reservation;
- package and promotion-evidence hashes.

The manifest schema and worker protocol versions MUST be aligned. The prior
active v2 manifest, systemd unit, and environment file are retained in the
activation bundle as the rollback target; the older v1 profile remains an
operational fallback where required.

## 2. OpenAI and OpenWebUI behavior

The Gateway accepts `reasoning_effort` or `thinking_budget_tokens`, but not
both. It emits reasoning through `delta.reasoning_content` and the ordinary
answer through `delta.content`. Delimiter token IDs and their decoded text do
not appear in either field.

OpenWebUI MUST be tested against the existing image without a uLLM-specific UI
patch. The following behavior is required:

- exact budgets are supplied through the existing model `custom_params`
  mechanism as `thinking_budget_tokens`; the deployment configurator preserves
  that model parameter and the OpenWebUI OpenAI payload adapter forwards it as
  an integer field;
- the reasoning panel starts and completes when the first answer content arrives;
- the hidden reasoning is not reinserted into the next turn when the dialect
  policy is `omit`;
- Stop, browser refresh, and multiple turns return the Gateway and worker to
  ready state;
- a direct Gateway 429 busy response and `Retry-After` remain unchanged.

The existing OpenWebUI Stop, worker-failure, and 20-chat soak gates retain
their SQ8 defaults but accept `ULLM_MODEL_ID` and `ULLM_MODEL_NAME` overrides.
The v2 candidate gates MUST set these to
`ullm-qwen3.5-9b-aq4` and `uLLM Qwen3.5 9B AQ4`; this keeps the gate logic
shared without silently measuring the SQ8 model.
For the stability gate, the normal run MUST set
`ULLM_OPENWEBUI_SOAK_COUNT=100`; the restart-recovery run MUST set it to
`20`. The default remains `20` for compatibility with the existing SQ8 gate.

The candidate profile uses the dedicated
`promotion-reasoning-v2-v0.1.json` receipt path. The existing v1 receipt MUST
not be copied or renamed for the v2 candidate.

## 3. Evidence layout

Evidence is written under a dated directory as streaming JSONL or bounded JSON.
User prompt/response text, authorization headers, API keys, and OpenWebUI DB
contents MUST NOT be stored. Public fixture IDs and hashes are used instead.

Each measured request records at least:

- source commit, manifest hash, worker binary hash, tokenizer hash, and image
  identity;
- fixture ID and prompt hash, prompt token count, requested budget, and stream
  mode;
- raw generated token count, reasoning-body count, forced-end count, answer
  count, usage cross-check, finish reason, and reset completion;
- HTTP status, non-stream field hashes, SSE event order, chunk count, and
  response field hashes;
- cancellation/failure outcome when the request is an abnormal-case run.

The producer's `passed` field is not authoritative. A validator MUST recompute
the gates from the raw records and reject incomplete or identity-mismatched
evidence. Temporary evidence uses an `.incomplete` name and is atomically
renamed only after validation.

The release evidence schema is defined in
`docs/specs/generic-reasoning-release-evidence-v0.1.md`, and its validator is
`tools/validate-generic-reasoning-release.py`.
Its `lifecycle` section binds sanitized `request_released` accounting and
reset timing to each measured case; a complete release artifact cannot omit
that correlation.
The browser-side Phase 5 smoke is
`deploy/openwebui/browser-reasoning-smoke.cjs`; it records only hashes, counts,
and boolean state, and must be run only after a v2 candidate is configured.
The checked-in runner `tools/run-openwebui-reasoning-browser-smoke.py` requires
the served-model manifest, rejects an active v1 manifest before starting a
browser, binds the candidate and comparison model IDs, validates the v2 record,
and atomically publishes only gate-eligible hash-only output.
Its current evidence schema is
`ullm.openwebui.reasoning_browser_smoke.v2`; v2 records the model hash for
each provider request. Existing evidence also contains a uLLM → llama.cpp →
uLLM switch cycle, but llama.cpp comparison and provider switching are not
required release gates after the 2026-07-14 user decision.
The validator retains read compatibility with the earlier v1 hash-only record.
The `expanded_view` field is a hash and byte count of the expanded assistant
view; it is required to be larger than the answer-only view without retaining
the visible text.
Its output is independently checked by
`tools/validate-openwebui-reasoning-browser-smoke.py` before it can enter the
release evidence bundle.
It validates the hash-only record shape, recomputes token accounting and usage
cross-checks, rejects budget overshoot and forbidden body or credential fields,
and reports structural validity separately from production-gate eligibility.

The Phase 0 HTTP collector and its validator use request/response hashes and
bounded protocol metadata only. A structurally valid record may still be
gate-ineligible when source identity or generated token IDs are missing.

For a v2 reasoning request, the worker `released` event carries
`reasoning_tokens` and `forced_end_tokens` in addition to
`completion_tokens`. The Gateway requires both fields, checks that their sum
does not exceed the committed completion count, and keeps the raw token split
as the response-side cross-check. The corresponding `request_released`
lifecycle record mirrors these two fields for reasoning requests so that
systemd-journal evidence can reproduce the raw accounting; v1 releases keep
the original event shape. The lifecycle record remains hash-only and never
contains token IDs or decoded content.
The resident promotion runner also sends a budget-zero reasoning request for
v2 candidates and verifies the complete forced-end sequence plus one reserved
answer token; the legacy v1 comparison remains limited to the raw no-reasoning
cases. Before starting either process, it requires a
`rocm-smi --showpids --json` preflight proving that the target R9700 has no positive-VRAM
KFD process. It records the empty positive-process result and fails closed when
the active worker or llama.cpp still owns the GPU.

## 4. Release gates

The candidate is not eligible for activation until all of these are true:

1. v1 no-reasoning token/API/SSE regression evidence is aligned to the same
   source commit and includes short, long, 1,024, 2,048, and 3,072-token
   prompts.
2. Synthetic multi-token dialect, v1/v2 strict parsing, forced-close budget,
   EOS, length, cancellation, publisher failure, and post-reset reuse tests
   pass.
3. Non-stream and stream reasoning fields concatenate to the same values,
   usage matches raw committed tokens, and budget overshoot is zero.
4. OpenWebUI browser tests pass for reasoning display, Stop, refresh, multiple
   turns, and hidden-history omission.
5. The reasoning-disabled performance regression is within the production
   thresholds in the generic reasoning plan; resource samples show no leak,
   zombie worker, or OOM.
6. Promotion evidence, validator output, receipt, worktree status, and rollback
   target are stored with the candidate identity.

## 5. Activation and rollback

Activation uses the existing atomic served-model activation tool. It MUST keep
the previous active manifest and service configuration. Before activation, the
candidate is validated using the real worker binary and real package paths;
generation alone is insufficient.

For a v2 candidate, the activation invocation MUST also provide the complete
generic release bundle, the current systemd unit, and the current environment
file. The activation tool recomputes the bundle gate, candidate manifest and
worker identity, and all three rollback hashes before switching the active
manifest. A v2 candidate without this binding is rejected before the atomic
replace.

Rollback restores the previous v1 manifest/profile, restarts the Gateway only
through the normal service procedure, and reruns a reasoning-disabled baseline
smoke. OpenWebUI has no reasoning-specific image rollback because no permanent
UI patch is allowed.

## 6. Current state

The repository has v2 schema, Gateway, worker, and AQ4 session contract tests,
including synthetic multi-token reasoning. The current service is active/running
with `NRestarts=0`, manifest SHA
`feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44`, source
commit `ae8b2bb7c2735f4dc761773957bf45f470dd5a8c`, and worker SHA
`177f3106414efc7cc4b08fa2d87bed6e147d4188e0a290f43b7a1ac591fae48d`.
Phase 0, resident promotion, release validator, and browser validator are
gate-eligible. The normal OpenWebUI soak passed 100/100 chats with 500 lifecycle
records and zero restarts. Previous-v2 and current-candidate HTTP/SSE populations
each contain 100 cases with 100/100 correctness, zero empty answers, zero budget
overshoot, and 100/100 resets; all identity-matched p95 deltas are within the
plan thresholds. The retained final bundle is
`release-bundle-ae8b2bb-20260714-final.json`; intermediate restoration records
are preserved in the external evidence archive.

The live OpenWebUI managed model still needs an operational reconciliation:
its stored manifest marker predates `feb3190d…`, and its current
`reasoning_effort=high` maps to 256 tokens without proving the explicit
`thinking_budget_tokens` custom parameter path. This does not invalidate the
Gateway/worker evidence, but it remains open before declaring exact-budget UI
operation complete. llama.cpp performance comparison is explicitly out of
scope.

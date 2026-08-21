# Phase 42 inference modes and public endpoints

## Status and scope

Phase 42 completed on 2026-08-22. It added OpenAI-subset Completions and
Embeddings, sLLM-native Rerank, tokenizer/template utilities, input-token
counting, and capability-gated FIM/infill to the shared frontend, scheduler,
HTTP server, and CLI. Existing Chat Completions profile-v1 semantics remain
unchanged.

The machine-readable contract is the
[Phase 42 profile fixture](../../../../../tests/fixtures/phase42_profiles_v1.json),
the [Draft 2020-12 schema](../../../../../ci/schema/phase42-profile-v1.schema.json),
and the [dependency-free validator](../../../../../ci/tools/validate_phase42_profiles.py).
Runtime integration and numerical evidence are covered by Rust tests and the
compact [exact-GPU summary](../../../../../ci/matrix/phase42-inference-gpu-summary-v1.json).

## Immutable references

- OpenAI OpenAPI `2.3.0` operation pin: commit
  [`117ce5680e4269f6656a4fd70d28f9755630d938`](https://github.com/openai/openai-openapi/tree/117ce5680e4269f6656a4fd70d28f9755630d938).
- llama.cpp technical reference: detached `b10453`, commit
  [`3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`](https://github.com/ggml-org/llama.cpp/tree/3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70).
- sLLM remains the specification owner. llama.cpp endpoint aliases, arbitrary
  templates, and implementation-specific options are not inherited. No
  llama.cpp source was directly reused.

## Contract decisions

### Completions

`POST /v1/completions` is an explicit OpenAI subset. `model` and `prompt` are
required. Prompt accepts string, string array, token array, or token-array
array. `max_tokens` is 1–4,096 (default 256), stop has at most four unique
nonempty strings, `n` is 1–8, logit bias is bounded to 4,096 unsigned 32-bit
token IDs with finite values in [-100,100], and `logprobs` is 0–5. Chat
`messages`, tools, and unlisted OpenAI fields are rejected rather than ignored.

Non-stream and SSE responses lower through the existing completion transport:
usage is prompt/completion/total, choices retain Phase 40 state, and a
post-header failure emits one JSON error event and closes without a completion
chunk or `[DONE]`.

### Embeddings and Rerank

`POST /v1/embeddings` accepts the same four input shapes and `float`/`base64`
encoding. Pooling is fixed to the arithmetic mean over final-RMSNorm hidden
rows with F64 accumulation, followed by L2 normalization and finite-F32
publication. Dimension is bound to model-lock hidden size, input ordering is
preserved, and usage reports prompt/total tokens. Empty or mixed input,
non-finite rows, unsupported encoding, multimodal input, and dimension mismatch
are rejected.

`POST /v1/rerank` is `sllm-rerank-v1`, not an OpenAI compatibility claim. It
scores the dot product of L2-normalized query/document embeddings, orders
higher scores first, and preserves original document order on ties. `top_n`
must be 1 through document count and is never clamped.

### Utilities and FIM

`/v1/tokenize`, `/v1/detokenize`, `/v1/apply-template`, and `/v1/input-tokens`
share one frontend tokenizer/renderer service with CLI. Model-default special
tokens, Unicode, lossless byte fallback, a 16 MiB input limit, 1,048,576-token
limit, and u32 token IDs are fixed. Template application accepts typed messages
and the reviewed renderer subset; arbitrary Jinja and custom kwargs are
rejected. These operations do not allocate GPU execution work.

`POST /v1/infill` requires a model lock with verified FIM prefix/suffix/middle
IDs, template digest, context limit, and provider support. Unsupported models
reject before execution, with no generic-completion fallback or visible FIM
markers. Current production Qwen and Gemma locks remain
`unsupported-until-verified-template`; a synthetic capability fixture covers
the supported state machine.

## Rejection and security contract

All Phase 42 profiles use the existing 96 MiB body and 256-byte model-alias
limits. Unknown fields, known-but-unsupported fields, wrong types, duplicate
JSON members, non-finite numbers, mixed arrays, invalid ranges, empty required
collections, and oversize content fail closed before model/GPU admission.
Prompt, token, vector, credential, and secret content is excluded from logs,
metrics, and props.

## Implementation and verification

- `sllm-frontend` owns typed raw/token/chat/FIM inputs and the shared utility
  service. `sllm-core` exposes final-normalized hidden rows and the fixed
  embedding arithmetic. Qwen dense, Qwen MoE, and Gemma production backends
  expose embeddings through the existing single-owner HIP runtime.
- The scheduler provides bounded typed generation, embedding, rerank, and CPU
  utility jobs with timeout/cancel handling. Strict HTTP and CLI integration
  tests cover positive endpoints, synthetic supported FIM, production
  unsupported FIM, SSE, errors, bounds, and legacy Chat regression.
- Scalar embedding/rerank tests cover non-aligned dimensions, F64 oracles,
  stable ties, base64, non-finite rejection, and L2 postconditions. The profile
  validator, full Draft 2020-12 schema, Markdown/link checks, workspace tests,
  clippy with warnings denied, formatting, and affected HIP/core tests passed.

Exact full-model embeddings passed on V620 `gfx1030` and R9700 `gfx1201` for
Qwen3.5-4B (dimension 2,560) and Gemma-4-12B (dimension 3,840). All four rows
used the same 8-token input, returned one finite vector with L2 norm within
`2e-9` of 1.0, reported usage `8/8`, and audited exact HIP target with no
fallback.

Gemma exposed a pre-existing static-FP8 state defect during validation: append
and causal attention treated absent dynamic scale planes as resident. Static
scales now remain checked descriptor scalars, per-row scale planes are written
only for dynamic FP8, and resident-byte accounting follows the two-plane static
contract. Full-model Gemma embeddings on both RDNA targets cover the repair.

The feature-pinned `gfx942:sramecc+:xnack-` wave64 release build passed. MI300X
execution remains deferred by user instruction and is not claimed by this
Phase; compile-only evidence does not promote runtime capability.

## Known limitations

- Current production Qwen/Gemma locks have no verified FIM capability, so
  `/v1/infill` fails closed.
- Embedding pooling is fixed to mean plus L2. Multimodal embedding, arbitrary
  projection, and classifier-head reranking are outside this profile.
- CLI utility JSON is a command report rather than a byte-for-byte HTTP
  envelope alias. Both surfaces share frontend semantics.
- MI300X runtime correctness and performance remain in the deferred Phase
  37/38 hardware lane.

## Links

[OpenAI compatibility profile](../../../../api/openai-compatibility.md) /
[runtime architecture](../../../../architecture/runtime.md) /
[model lock](../../../../models/model-lock.md) /
[Phase 42 profile fixture](../../../../../tests/fixtures/phase42_profiles_v1.json) /
[Phase 42 schema](../../../../../ci/schema/phase42-profile-v1.schema.json) /
[Phase 42 validator](../../../../../ci/tools/validate_phase42_profiles.py) /
[exact-GPU summary](../../../../../ci/matrix/phase42-inference-gpu-summary-v1.json) /
[archive plan](../../../../plans/archive/2026/08/21-31/phase42-inference-modes-public-endpoints.md) /
[main plan](../../../../plans/main-plan.md)

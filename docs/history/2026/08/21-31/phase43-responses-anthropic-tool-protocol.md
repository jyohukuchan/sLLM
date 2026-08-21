# Phase 43 history: Responses・Anthropic Messages・function/tool protocol

## Outcome

Phase 43 completed on 2026-08-22. OpenAI Responses API `2.3.0` was pinned to
openai-openapi commit `010421dcbd0475277ea8c3e6c1e1cbca4659c4bd`; Anthropic
Messages was separately pinned to `anthropic-version: 2023-06-01`. Both strict
profiles use the existing bounded scheduler but retain distinct request,
response, error and named-SSE contracts.

The frontend now owns a bounded ordered protocol for messages, function
definitions, choices, calls, results and parallel policy. Tool-enabled Qwen
generation renders one escaped fixed prompt, compiles a Phase 40 JSON-Schema
grammar before scheduler/GPU admission, and decodes a canonical message or
tool-call envelope. The current Qwen production backend advertises this
capability; backends that do not advertise it, including the current Gemma
path, reject tool requests without fallback. The Responses no-tool subset also
connects Phase 41 assistant prefill without republishing the prefix.

`/v1/responses` and `/v1/messages` implement strict parsing, non-stream output,
stable request/item/call IDs, usage and stop mapping, profile-specific named
SSE, post-header failure closure, cancellation and Phase 39 replay. Visible
deltas split at UTF-8 boundaries to at most 16 KiB. Resumable requests are
limited to 40 output tokens before scheduler admission; this combines the
128-byte token-piece cap and worst-case JSON escaping so full snapshot/done
events fit with margin for other bounded fields. Resumable output also
preflights the full serialized event batch against the configured event count and the
64 KiB/event and 256 KiB/session replay bounds, so an oversized batch emits one
error terminal rather than a partial success sequence. Public errors use fixed
redacted messages and never include backend strings or request/tool payloads.

## Security boundary

This is a protocol-only implementation. A generated function call is returned
to the client, and a later client request may supply its result as untrusted
history. No process, network, filesystem, environment, secret, credential,
MCP, hosted tool, worker or sandbox execution path was added. A malicious
command/path marker remained inert in raw HTTP tests. The separately numbered
Phase 47 remains approval-required and was not started.

## Verification

- Phase 43 fixture/schema validator and six Python mutation tests passed.
- `sllm-core` grammar bounds, 71 frontend unit tests and 15 tool-protocol
  integration tests passed.
- 80 `sllm-server` unit tests, six strict parser tests and three raw HTTP/SSE
  integration tests passed before the cumulative workspace gate.
- Raw HTTP covered Responses and Anthropic non-stream/SSE, required version
  header, text/tool output, result roundtrip, single/parallel policy,
  capability rejection, stop-sequence mapping, inert malicious payload,
  backend-error redaction, disconnect cancellation and resumable replay.
- GPU kernels/providers and selector routes were unchanged. No host or
  compile-only result is recorded as MI300X runtime evidence; exact gfx942
  execution remains deferred by user instruction.
- No llama.cpp source was directly reused; the pinned llama.cpp revision was a
  behavior/reference inventory only.

## References

- [Archived Phase plan](../../../../plans/archive/2026/08/21-31/phase43-responses-anthropic-tool-protocol.md)
- [Machine profile](../../../../../tests/fixtures/phase43_protocol_profiles_v1.json)
- [OpenAI compatibility](../../../../api/openai-compatibility.md)
- [Anthropic compatibility](../../../../api/anthropic-compatibility.md)
- [Runtime architecture](../../../../architecture/runtime.md)

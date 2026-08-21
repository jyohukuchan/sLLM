# Phase 44 history: template・reasoning control・interactive UX

## Outcome

Phase 44 completed on 2026-08-22 as a host/frontend/CLI integration phase. The
reviewed Qwen renderer remains the default and the Gemma raw-text capability is
unchanged. Generic templates are a separate explicit opt-in provider pinned to
MiniJinja `2.24.0`; no arbitrary client Jinja execution or implicit fallback was
added.

The generic provider verifies UTF-8 source bytes and a caller-supplied lowercase
SHA-256 digest, uses a JSON-only context, strict undefined behavior, a closed
filter/test/global surface, and rejects include/import/extends, private
attribute access, dynamic loaders, and host callbacks. Source, output, message,
kwargs, depth, recursion, and fuel limits are applied before tokenization or
GPU admission. Typed messages, special tokens, generation/thinking flags,
reasoning effort, and bounded kwargs share the `TokenizerUtilityServiceV1`
adapter. CLI `apply-template` and `input-tokens` accept custom templates only
when both file and digest flags are supplied; kwargs parsing rejects duplicate
keys, non-finite values, and non-object input. The CLI reads a regular
non-symlink file once with `O_CLOEXEC|O_NOFOLLOW`, bounded read and size-race
checks, and emits only data-only template/render identity fields in reports.

Reasoning mode and budget use the existing frontend generation owner rather than
a second decode loop. Disabled, enabled, and template-default modes map to the
existing thinking contract. Budgets from 1 through 4,096 generated reasoning
tokens include multi-token closing markers; early close, forced close,
`max_new_tokens` insufficiency, grammar mask intersection, stop/sampling, and
cancellation conflicts fail closed before generation. Forced tokens remain in
ordinary usage/generated history. Chat, Responses, and CLI adapters lower to
the same controller; Anthropic thinking and Gemma/raw-text reasoning remain
unsupported.

The `chat` CLI owns the closed prompt-source matrix, bounded regular prompt-file
reader, typed transcript, reverse-prompt turn boundary, JSONL event envelope,
and successful-turn-only publication. Persistent Qwen chat removes hidden
reasoning, selected stop tokens, and matched reverse markers from the reviewed
history semantics, re-renders that canonical history prefix, re-prefills a fresh
resident owner, and captures the result as opaque checkpoint state. This keeps
the next-turn and fresh-resume prompt prefix exact. Checkpoint load validates
exact model, renderer, tokenizer, target, plan, and KV identity transactionally.
Conversation bytes and KV pending/current state promote or rollback together;
failed generation/capture/save/commit and cancellation leave the prior current
state installed. CLI production preflights source, prompt-file contents, and
reasoning limits before model/backend open, while a dedicated SIGINT listener
cancels only the in-flight turn's `GenerationCancellationV1`. Existing one-shot
`generate` reports and semantics are unchanged. Mid-generation/wire session
resume, WebUI, and Phase 47 tool/MCP execution were not started.

## Verification

- Focused `sllm-frontend` Phase 44 template/adapter/reasoning/generic-generation tests passed.
- Focused `sllm-cli` binary tests passed, including custom-template file/kwargs and interactive state tests.
- `cargo fmt --all -- --check`, workspace/all-target clippy with warnings denied, and the full workspace test suite passed. The Rust dependency
  policy validator also passed its exact Rust 1.85 offline workspace/all-target build gate after mechanically updating the closed dependency graph.
- Frontend tests include source/digest/size boundaries, Unicode and special-token rendering, unknown builtin/directive rejection, output/depth/
  message/kwargs/fuel/recursion limits, typed adapter identity, raw/Gemma pre-tokenize rejection, and reasoning mask/close-marker oracles.
- CLI tests include duplicate/non-finite/non-object kwargs, symlink/path-redacted/file-race boundaries, custom flag pairing, raw input rejection,
  and unsupported-command rejection.
- No GPU kernel/provider/selector ABI was changed for Phase 44. The MI300X VM was deleted by user instruction; exact gfx942 correctness and
  performance remain deferred. Feature-pinned compile or host evidence is not recorded as MI300X runtime PASS.
- No llama.cpp source was directly reused. The pinned `b10453` revision is a behavior/reference comparison only.

## Known limitations and handoff

- Current production model locks do not advertise arbitrary custom-template HTTP fields; the generic provider is transport-independent and CLI opt-in.
- Mid-generation checkpoint resume, wire session ownership, WebUI, adapter/router lifecycle, and converter/quality tools remain later phases.
- Built-in tool/MCP execution remains approval-required Phase 47; generated tool calls and results remain inert protocol data.
- MI300X Phase 37/38 hardware work resumes only after a fresh exact runtime is available, from a new baseline rather than compile-only evidence.

## References

- [Archived Phase plan](../../../../plans/archive/2026/08/21-31/phase44-template-reasoning-interactive-ux.md)
- [Main plan](../../../../plans/main-plan.md)
- [Phase 37+ roadmap](../../../../plans/active/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md)
- [Runtime architecture](../../../../architecture/runtime.md)
- [Model lock](../../../../models/model-lock.md)
- [OpenAI compatibility](../../../../api/openai-compatibility.md)
- [Phase 41 history](phase41-prefix-session-speculation.md)
- [Phase 43 history](phase43-responses-anthropic-tool-protocol.md)

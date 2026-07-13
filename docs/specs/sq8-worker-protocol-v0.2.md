# uLLM worker JSONL protocol v0.2

Status: proposed implementation contract

`ullm.worker.v2` preserves the v1 bounded JSONL framing, active1/waiting0
admission, cancellation, release, and strict unknown-field policy. It adds
model-independent reasoning execution data to `generate`.

## 1. Generate command

```json
{
  "schema_version": "ullm.worker.v2",
  "type": "generate",
  "request_id": "req-1",
  "prompt_token_ids": [1, 2, 3],
  "max_new_tokens": 256,
  "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "seed": 7},
  "eos_token_ids": [248044, 248046],
  "reasoning": {
    "enabled": true,
    "budget_tokens": 128,
    "dialect_id": "qwen3.5-thinking-v1",
    "end_token_ids": [248069],
    "forced_end_token_ids": [248069],
    "reserved_answer_tokens": 1
  }
}
```

`budget_tokens` is `null` for unbounded reasoning or a nonnegative integer for a
hard budget. The worker MUST validate that the end sequences match its loaded
manifest and MUST reject a request that cannot reserve the forced close sequence
and minimum answer tokens.

## 2. Execution contract

The worker samples while the phase is `reasoning`. A completed natural end
sequence enters `answer` without publishing delimiter tokens as user-visible
content. A hard-budget or reasoning-EOS transition stops sampling and publishes
the declared forced sequence through the existing prepare/publish/commit
boundary. Forced tokens are counted in generated-token usage but not in
reasoning-token usage. At least `reserved_answer_tokens` answer tokens remain
available or the request is rejected before generation.

The worker MUST expose enough release accounting for the gateway to verify:

- raw generated token count;
- reasoning-body token count;
- forced-end token count;
- final outcome; and
- reset completion.

For a reasoning request, the `released` event MUST include the integer fields
`reasoning_tokens` and `forced_end_tokens` alongside `completion_tokens`. Both
fields are required together, their sum MUST NOT exceed `completion_tokens`, and
they count only tokens committed before the terminal release. A v1 release
keeps the frozen event shape and does not include these fields.

## 3. Compatibility

v1 commands remain v1-only and do not acquire hidden reasoning defaults. A v1
worker MUST reject v2 commands, while a v2 worker may support v1 only through an
explicit compatibility mode selected by the launcher.

# Served-model manifest v0.2

Status: ratified production contract

Date: 2026-07-24

## 1. Version boundary

`ullm.served_model.v2` incorporates every file, JSON, path, identity, and
failure rule of `served-model-manifest-v0.1.md` and adds exactly one required
top-level `reasoning` object. Its `worker.protocol` MUST be
`ullm.worker.v2`. Conversely, `ullm.served_model.v1` MUST use
`ullm.worker.v1`, MUST NOT contain `reasoning`, and retains its frozen
semantics.

There is no implicit upgrade, compatibility fallback, or schema inference.
Manifest schema, worker protocol, and the command/event discriminator are one
version-locked tuple.

## 2. Reasoning object

The independent Qwen3-14B-FP8 `SQ8_0` production identity is:

```json
{
  "reasoning": {
    "enabled_by_default": false,
    "dialect_id": "qwen3-thinking-v1",
    "start_token_ids": [151667],
    "end_token_ids": [151668],
    "forced_end_token_ids": [151668],
    "initial_phase": "reasoning",
    "eos_policy": "close",
    "effort_budgets": {"low": 32, "medium": 128, "high": 256},
    "max_budget_tokens": 256,
    "reserved_answer_tokens": 1,
    "history_reasoning_policy": "omit"
  }
}
```

The reasoning object has exactly the eleven fields shown.
`start_token_ids`, `end_token_ids`, and `forced_end_token_ids` are each an
array containing exactly one nonnegative integer token ID below
`generation.vocab_size`; JSON booleans are not integers. A multi-token
delimiter is outside `ullm.served_model.v2` and requires a future versioned
manifest and worker/session contract. Model-independent internal reasoning
utilities may support such sequences, but that capability is not a v2
production authorization.

`end_token_ids` and `forced_end_token_ids` MUST be identical in v0.2.
`initial_phase` is exactly `reasoning` or `answer`; `eos_policy` is exactly
`close`, `finish`, or `continue`; and `history_reasoning_policy` is exactly
`omit` or `preserve`. `enabled_by_default` is a JSON boolean.

`effort_budgets` has exactly `low`, `medium`, and `high`. Each value is a
positive integer no greater than the positive integer
`max_budget_tokens`. `reserved_answer_tokens` is a positive integer. The sum

```text
max_budget_tokens
+ len(forced_end_token_ids)
+ reserved_answer_tokens
```

MUST NOT exceed `generation.max_completion_tokens`.

The loader rejects a token sequence whose length differs from one, an
out-of-vocabulary ID, an unknown/missing field, an incomplete effort map, a
budget above the maximum, unequal natural/forced delimiters, or a start/end
collision.

## 3. Request and template boundary

The v2 tokenizer contract retains `add_generation_prompt` and
`enable_thinking`. The manifest value is the default template choice; it does
not authorize a request to replace token IDs, the dialect identity, the
history policy, or any other manifest field.

Every `ullm.worker.v2` generate command carries an explicit reasoning execution
object, including when reasoning is disabled. Request normalization selects
only `enabled` and a bounded/unbounded budget; all dialect identity and
reservation fields must equal the loaded manifest.

## 4. Identity, loading, and activation

The manifest SHA-256 binds the public model, tokenizer files and template,
worker binary/protocol, product manifests, promotion receipt, and reasoning
dialect. Python and Rust loaders MUST apply the same strict v1 file boundary,
exact-field checks, hash verification, and v2 version alignment before worker
launch.

Activation remains atomic and rollback restores the exact previous manifest
bytes. A v2 candidate is not production-admissible until the v2 worker,
serving-session, worker-acceptance, release-evidence, and bundle validators
have all accepted identities from that same candidate. This ratification does
not itself authorize activation.

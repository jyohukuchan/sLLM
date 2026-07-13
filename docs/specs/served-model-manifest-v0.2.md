# Served-model manifest v0.2

Status: proposed implementation contract

`ullm.served_model.v2` keeps every v1 field and adds one required top-level
`reasoning` object. The v1 loader remains strict and does not accept this field.

## 1. Reasoning object

```json
{
  "reasoning": {
    "enabled_by_default": false,
    "dialect_id": "qwen3.5.thinking.v1",
    "start_token_ids": [248068],
    "end_token_ids": [248069],
    "forced_end_token_ids": [248069],
    "initial_phase": "reasoning",
    "eos_policy": "close",
    "effort_budgets": {"low": 32, "medium": 128, "high": 256},
    "max_budget_tokens": 256,
    "reserved_answer_tokens": 1,
    "history_reasoning_policy": "omit"
  }
}
```

All token sequences are nonempty arrays of unique nonnegative IDs below the
declared vocabulary size. Sequences are token-level values and may contain more
than one token. `end_token_ids` and `forced_end_token_ids` must be identical in
v0.2. `initial_phase` is `reasoning` or `answer`; `eos_policy` is `close`,
`finish`, or `continue`. Effort keys are exactly `low`, `medium`, and `high`,
with positive budgets not exceeding `max_budget_tokens`.

`history_reasoning_policy` is `omit` or `preserve`. `reserved_answer_tokens`
must be at least one. The loader rejects empty arrays, duplicates, invalid IDs,
unknown keys, effort budgets above the maximum, and a delimiter prefix collision
that cannot be represented by the token state machine.

## 2. Template options

The v2 tokenizer contract keeps `add_generation_prompt` and allows
`enable_thinking` to be selected per normalized request. The manifest's value is
the default, not a permission to bypass the dialect or tokenizer identity.

## 3. Identity and activation

The manifest digest, tokenizer digest, worker digest, and promotion receipt bind
the dialect identity. Activation is atomic and rollback restores the previous
active manifest. A v2 manifest is not activated until the v2 validator and the
synthetic multi-token dialect fixture pass.

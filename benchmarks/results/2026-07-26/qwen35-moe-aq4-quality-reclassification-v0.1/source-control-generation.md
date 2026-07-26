# Qwen3.5-35B-A3B source vs AQ4_0 CPU streaming generation

This is a bounded CPU-only, one-decoder-layer-at-a-time greedy generation check. It uses source BF16 checkpoint values converted to F32 arithmetic on both tracks; only the right track's routed experts are decoded from the `AQ4_0` package. The final RMSNorm and `lm_head` are raw source/passthrough weights.

The suite is intentionally shortened from the 10-case lightweight-promotion suite for CPU feasibility. It is evidence for package reclassification, not a serving or promotion run. Greedy-token equality and route-set equality are recorded as observations, not pass/fail rules.

- right track: `source control`
- threads: `12`
- wall time: `885.604 s`

## Side-by-side outputs

### ja_concise_rollback (japanese_prose)

Prompt messages:

```text
[user] 次の文を自然な日本語で一文だけ続けてください。ロールバック手順を残す利点は
```

Source (nonquantized routed experts):

`````text
ロールバック手順を残す利点は、予期せぬ
`````

source control:

`````text
ロールバック手順を残す利点は、予期せぬ
`````

Observations (not thresholds):

- generated tokens: source `12`, source control `12`
- source-greedy token matches: `12`/`12`
- route observations during this path: selected-set `0`/`1720`, ordered `0`/`1720`
- source-greedy conditional NLL: source `0.111562`, source control `0.111562` (descriptive only)
- automatic symptom screen: source `{"empty": false, "replacement_character": false, "control_characters": [], "threefold_consecutive_fragment": null, "characters": 19}`, source control `{"empty": false, "replacement_character": false, "control_characters": [], "threefold_consecutive_fragment": null, "characters": 19}`

### en_concise_rollback (english_prose)

Prompt messages:

```text
[user] Complete this sentence in one concise English sentence: Keeping a rollback path is useful because
```

Source (nonquantized routed experts):

`````text
Keeping a rollback path is useful because it allows teams to quickly
`````

source control:

`````text
Keeping a rollback path is useful because it allows teams to quickly
`````

Observations (not thresholds):

- generated tokens: source `12`, source control `12`
- source-greedy token matches: `12`/`12`
- route observations during this path: selected-set `0`/`1600`, ordered `0`/`1600`
- source-greedy conditional NLL: source `0.093489`, source control `0.093489` (descriptive only)
- automatic symptom screen: source `{"empty": false, "replacement_character": false, "control_characters": [], "threefold_consecutive_fragment": null, "characters": 68}`, source control `{"empty": false, "replacement_character": false, "control_characters": [], "threefold_consecutive_fragment": null, "characters": 68}`

### python_is_even (code_generation)

Prompt messages:

```text
[user] Write only a one-line Python definition of is_even(n).
```

Source (nonquantized routed experts):

`````text
is_even = lambda n: n % 2 == 0
`````

source control:

`````text
is_even = lambda n: n % 2 == 0
`````

Observations (not thresholds):

- generated tokens: source `14`, source control `14`
- source-greedy token matches: `14`/`14`
- route observations during this path: selected-set `0`/`1480`, ordered `0`/`1480`
- source-greedy conditional NLL: source `0.006532`, source control `0.006532` (descriptive only)
- automatic symptom screen: source `{"empty": false, "replacement_character": false, "control_characters": [], "threefold_consecutive_fragment": null, "characters": 30}`, source control `{"empty": false, "replacement_character": false, "control_characters": [], "threefold_consecutive_fragment": null, "characters": 30}`

## Important retained observation

The original same-input 8-token × 40-layer prefill evidence remains the canonical route observation: selected expert sets changed 105/320 and ordered top-k changed 238/320 for source vs `AQ4_0`; the source-vs-source control changed 0/320. This generation record does not treat those rates as a quality gate.

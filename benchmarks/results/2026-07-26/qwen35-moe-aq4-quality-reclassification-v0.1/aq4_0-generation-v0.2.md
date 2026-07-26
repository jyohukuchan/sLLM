# Qwen3.5-35B-A3B source vs AQ4_0 CPU streaming generation

This is a bounded CPU-only, one-decoder-layer-at-a-time greedy generation check. It uses source BF16 checkpoint values converted to F32 arithmetic on both tracks; only the right track's routed experts are decoded from the `AQ4_0` package. The final RMSNorm and `lm_head` are raw source/passthrough weights.

The suite is intentionally shortened from the 10-case lightweight-promotion suite for CPU feasibility. It is evidence for package reclassification, not a serving or promotion run. Greedy-token equality and route-set equality are recorded as observations, not pass/fail rules.

- right track: `AQ4_0`
- threads: `12`
- wall time: `1503.778 s`

## Side-by-side outputs

### ja_rollback_recovery (japanese_prose)

Prompt messages:

```text
[user] 日本語で一文だけ答えてください。ロールバック手順を残すと、失敗した変更からどのように復旧できますか？
```

Source (nonquantized routed experts):

`````text
ロールバック手順を残すことで、失敗した変更を事前に定義された手順に従って迅速かつ確実に元の状態に戻す
`````

AQ4_0:

`````text
ロールバック手順を残すことで、失敗した変更を事前に定義された安全な手順に従って元の状態に迅速に
`````

Observations (not thresholds):

- generated tokens: source `24`, AQ4_0 `24`
- source-greedy token matches: `14`/`24`
- route observations during this path: selected-set `948`/`2400`, ordered `1728`/`2400`
- source-greedy conditional NLL: source `0.301512`, AQ4_0 `5.087652` (descriptive only)
- automatic symptom screen: source `{"empty": false, "replacement_character": false, "control_characters": [], "threefold_consecutive_fragment": null, "characters": 50}`, AQ4_0 `{"empty": false, "replacement_character": false, "control_characters": [], "threefold_consecutive_fragment": null, "characters": 47}`

### en_rollback_recovery (english_prose)

Prompt messages:

```text
[user] Answer in one concise English sentence: How does a rollback path help after a failed change?
```

Source (nonquantized routed experts):

`````text
A rollback path restores the system to its previous stable state by reversing the failed change, thereby minimizing downtime and mitigating potential
`````

AQ4_0:

`````text
A rollback path restores a system to its previous stable state by reversing a failed change, thereby minimizing downtime and preventing service disruption
`````

Observations (not thresholds):

- generated tokens: source `24`, AQ4_0 `24`
- source-greedy token matches: `19`/`24`
- route observations during this path: selected-set `899`/`2160`, ordered `1608`/`2160`
- source-greedy conditional NLL: source `0.232312`, AQ4_0 `2.072235` (descriptive only)
- automatic symptom screen: source `{"empty": false, "replacement_character": false, "control_characters": [], "threefold_consecutive_fragment": null, "characters": 149}`, AQ4_0 `{"empty": false, "replacement_character": false, "control_characters": [], "threefold_consecutive_fragment": null, "characters": 154}`

### python_is_even (code_generation)

Prompt messages:

```text
[user] Write only a one-line Python definition of is_even(n).
```

Source (nonquantized routed experts):

`````text
is_even = lambda n: n % 2 == 0
`````

AQ4_0:

`````text
is_even = lambda n: n % 2 == 0
`````

Observations (not thresholds):

- generated tokens: source `14`, AQ4_0 `14`
- source-greedy token matches: `14`/`14`
- route observations during this path: selected-set `429`/`1480`, ordered `949`/`1480`
- source-greedy conditional NLL: source `0.006532`, AQ4_0 `0.010436` (descriptive only)
- automatic symptom screen: source `{"empty": false, "replacement_character": false, "control_characters": [], "threefold_consecutive_fragment": null, "characters": 30}`, AQ4_0 `{"empty": false, "replacement_character": false, "control_characters": [], "threefold_consecutive_fragment": null, "characters": 30}`

## Important retained observation

The original same-input 8-token × 40-layer prefill evidence remains the canonical route observation: selected expert sets changed 105/320 and ordered top-k changed 238/320 for source vs `AQ4_0`; the source-vs-source control changed 0/320. This generation record does not treat those rates as a quality gate.

# `SQ8_0` direct vs GQA-grouped tile-20: human text review

This is a same-model review of the captures in `direct/capture-20260726T160603Z/`
and `gqa-grouped-tile20/capture-20260726T160603Z/`.  It is not a comparison
with `AQ4_0`.

## Runtime result

Both isolated workers reached ready and completed all ten fixed requests.  The
capture and comparison tools found no request failure, empty completion,
gateway/worker exit, repetition loop, garbling, or extreme-length finding.
The direct and grouped outputs have zero exact matches; that is a diagnostic
observation only and is not used as a threshold.

## Human reading result

The grouped candidate is readable and does not show a catastrophic text
failure, but it is **not quality-approved for a future activation** from this
capture alone:

- In `python_code`, the grouped response ends after a prose parameter/example
  introduction and supplies no Python function or executable example.  The
  direct control starts a Python code block.  This is an actual code-request
  abandonment in the candidate response, even though the automatic heuristic
  did not classify it as one.
- In `javascript_debug`, the grouped code uses `filter(isFinite)` correctly
  for this numeric array, but its explanation incorrectly says that
  `Boolean(NaN)` is true.
- In `ja_multiturn`, the grouped response ends at a bare `**` and does not
  provide the requested two implementation precautions.

Several direct outputs are also cut off by the fixed suite's 96/128-token
response limits (including the direct Python and structured-reasoning cases).
That prevents attributing every incompleteness solely to the tiled attention
route, but it does not turn the grouped candidate's observable defects into a
quality pass.  The policy intentionally treats exact match, content accuracy,
and prose differences as human-readable evidence rather than an automated
numeric gate; this review applies that rule directly.

## Decision

Keep the result as a validated service-candidate execution record, but hold
text-quality approval.  No promotion was attempted: independently of this
quality hold, promoting `SQ8_0` would replace the active `AQ4_0` product model
and is outside this task.

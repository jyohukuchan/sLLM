# `SQ8_1` W8A8 lightweight recheck

## Scope

This is an investigation-only CPU replay of the exact `SQ8_1` K=32 W8A8
fake-quant boundary used by the prior full-model quality gate.  It quantizes
the 248 selected transformer projections and their dynamic K=32 inputs using
signed int8 codes and upward-rounded FP16 scales; `lm_head` remains FP32.  It
does not claim a served candidate or an R9700 throughput result.  The frozen
10-case prompt suite is complete (`run-complete.json`), and
`comparison.md` preserves source-reference and W8A8 text side by side.

## Human review

The automatic text checks found no empty response, repetition loop, garbling,
or total code abandonment.  That is not sufficient to accept this candidate.
The `javascript_debug` candidate generated the correct filtering expression,
but then stated both that **`NaN` is truthy** and that `Boolean(NaN)` is
`true`.  Both are false: a local Node.js check returned
`{"booleanNaN":false,"booleanInfinity":true,"finiteNaN":false,"finiteInfinity":false}`.
The source-reference text correctly described `NaN` as falsy.

This is a concrete factual defect in an explanation requested by the fixed
suite, rather than an exact-token or logit comparison.  The case is retained
verbatim in `sq8_1-w8a8/javascript_debug.json`; the paired text is in
`comparison.md`.  The language-runtime confirmation is retained in
`javascript-semantics-check.json`.

Several other cases stop at the fixed 96-token completion budget (for example,
the Python example, Japanese explanation, and structured-reasoning case).
The source-reference outputs stop at the same budget too, so those incomplete
answers are a suite-cap limitation rather than evidence attributable to
`SQ8_1` W8A8.  They were not used as a rejection reason.

## Disposition

Do not promote `SQ8_1` W8A8.  Independently of the observed explanation
error, there is no full-model `SQ8_1` served selector, manifest, or R9700
dispatch to measure or promote.  The format plan explicitly says not to
dispatch W8A8 on gfx1201/R9700 in v0.1.  Historical V620 kernel timings are
not a substitute for the required R9700 full-model decode measurement and
were not rerun.

# Lightweight promotion attempt 1 — baseline did not complete

The generic `tools/promote-served-model.py` route was invoked with the fixed ten-case prompt
suite and a fresh evidence directory.  Its preflight accepted both manifests and its bounded
gateway readiness probe passed on the first attempt.  It stopped before any active-manifest
mutation:

```json
{"status":"baseline_failed_before_mutation"}
```

The active manifest before, during, and after this attempt is
`c57a2b6c5827b8ddd102560b3f5efd879711705cf4d8a36f4d7872821d05fca4`.
No `candidate-readiness.json`, candidate output, service event, rollback transaction, or
activation outcome exists because the candidate phase was never entered.

## What generated successfully

`ja_explanation.json` and `en_explanation.json` contain real HTTP 200 text responses from the
old active `AQ4_0` worker.  Their full request/response text is preserved unchanged in
`active-output/`.

## Interrupted baseline (not a candidate-quality result)

The Python-code case (`active-output/python_code.json`) reached the old worker and received one
first token.  At 21:45:59.160 JST, while that request was still active, a different session issued
`sudo systemctl stop ullm-openai.service`; the gateway then recorded `unexpected worker stdout
EOF` and returned HTTP 500.  Seven remaining cases have `container_transport` because the gateway
was no longer live after that external teardown.

The journal attributes the 21:45:59 stop to an external command but does not identify its session.
BH's separate explicit `systemctl stop`/R9700-lock holder began at 21:46:18 JST.  Therefore this
attempt is a service-window collision, not evidence that either the active or candidate model
produces bad text.

Consequently the generic route correctly made no swap.  This attempt alone does not provide a
candidate-versus-active output comparison; it will be retried only after the active service and
GPU singleton lock are exclusively available.

# Gemma4 E2B BF16 serving assessment

## Scope

- Candidate manifest: `e01fa275a8e682c44606df2f1549cb0676df04d7b55b29e7f238ec7ec43fc8c9`
- Candidate model: `ullm-gemma4-e2b-bf16`
- Isolation: a manual gateway on `127.0.0.1:18080`; the active AQ4 manifest was not
  replaced by this candidate.
- Evidence: `gateway-readiness.json`, `gateway-models-response.json`,
  `gateway-response-*.json`, and `gateway-policy-suite-responses.json`.

## Transport and worker result

`/readyz` returned ready after the required initial 3.25-second wait (3.276 seconds
observed). `/v1/models` returned the Gemma4 model. The two direct requests and all ten
policy-suite requests returned HTTP 200 with a nonempty OpenAI completion field.

The manifest-bound raw worker also reproduced both prior greedy traces exactly:

| Trace | Expected and observed token IDs |
| --- | --- |
| capital of France | `[9079, 236761, 108, 818]` |
| once upon a time | `[528, 496, 1902, 1298]` |

`raw-worker-sequential-result.json` records the full comparison and `reset_complete=true`
terminal events.

## Chat-template result

The assembled overlay loads as `GemmaTokenizer`, renders the official E2B-it turn template,
and keeps every rendered token ID inside the base E2B vocabulary. This verifies the gateway
tokenizer contract mechanically. The base `google/gemma-4-E2B` tokenizer has no upstream chat
template, however, so the E2B-it template is an explicit experimental overlay rather than a
verified base-model conversational contract.

## Text quality decision

Do not promote this candidate. The decision is based on the saved generated text, not a numeric
threshold:

- The Japanese explanation repeats the user prompt.
- The Japanese multi-turn answer enters a repeated `1.` sequence.
- The English multi-turn answer repeats the same sentence.
- The Japanese summary emits `<unused56>`; translation and structured-reasoning completions
  are empty.

The France prompt did include “The capital of France is Paris,” confirming that the endpoint
does generate text, but that does not outweigh the clear failures above. The next work item is
not kernel tuning: obtain an upstream-supported base-E2B chat interface or package an
instruction-tuned checkpoint under its own strict source/package/manifest contract.

# SQ8_0 redesign: lightweight-promotion applicability

Status: not promotable through the current generic served-model contract. This
is a configuration-representation blocker, not a numerical gate. No actual
candidate completion text was obtained, so the lightweight policy's text
quality criterion is **unconfirmed**, not passed or failed.

The measured redesign is selected by these exact test-only values:

```text
ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE=20
ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_ALLOW_MULTITILE=1
ULLM_EXPERIMENTAL_PAGED_DECODE_GQA_GROUPED_SPLIT=1
ULLM_EXPERIMENTAL_PAGED_DECODE_GQA_PIPELINED_SPLIT=1  # pipeline experiment only
```

The fastest valid full-model variant is grouped tile 20 without the optional
pipeline flag. Both it and the pipeline experiment require the non-boolean
tile value `20`, so the blocker applies to either selection.

The existing generic served-model schema has only
`worker.required_environment`, a list of environment *names*.  In manifest
mode the gateway sets every listed name to the literal value `"1"`
(`services/openai-gateway/src/ullm_openai_gateway/worker.py`, lines 145--182).
It cannot represent the required tile value `20`; the SQ8_0 parser explicitly
rejects `1` and accepts only `20`, `128`, `256`, or `512`
(`crates/ullm-engine/src/sq8_serving_runtime.rs`, lines 386--392).

The only pre-existing SQ8_0 candidate manifest references
`uLLM-sq8-manifest-candidate-release-ee62d04e/ullm-sq8-worker`, SHA-256
`57a877fff70373b5fc57811eb7ba72b638f3f596ae5c60503b7dbf5fae0af5ef`, which
predates this redesign and has no manifest value channel for tile 20.

Consequently, invoking `tools/promote-served-model.py` would either test a
different worker or fail to select this redesign. It and
`tools/rollback-promoted-served-model.py` were not invoked, and the fixed
10-prompt suite was not falsely labelled as a candidate test. Adding a generic
environment-value map to the served-model contract is a separate cross-service
change; adding a candidate-specific transport is explicitly out of scope for
this task.

# Typed split paged-decode resident dispatch

- `PackageSelfAttnResidentStepLayer` now resolves a typed paged-decode dispatch at load time.
- The split path is opt-in only when both experimental environment variables are valid; normal
  loads retain the existing Qwen3.5 single-operation registry path.
- Split workspace capacity is checked with the runtime helper and allocated once per request
  state. Decode steps select single or split using the resolved threshold without re-resolution.
- Unit coverage includes strict environment parsing, generic workspace sizing, and host threshold
  dispatch selection.
- Session audit keeps the canonical six load-time traces; split reader implementations are typed
  alternates accepted only for their matching plain/gated single reader family.

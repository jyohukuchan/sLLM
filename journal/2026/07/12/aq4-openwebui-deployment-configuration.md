# AQ4 OpenWebUI deployment configuration

## Scope

- Kept the Qwen3 14B SQ8 identity and bridge URL as defaults.
- Made the OpenWebUI provider URL, model ID, display name, context metadata,
  and description configurable through `ULLM_*` environment variables or
  matching `configure.py` command-line options.
- A custom AQ4 model ID is inserted independently and does not delete the
  existing SQ8 model row.
- Documented a Qwen3.5 9B AQ4 environment example and the single-resident-worker
  switch procedure.

## Validation

- `python3 -m pytest -q tests/test_openwebui_configure.py`: 3 passed.
- `python3 -m py_compile deploy/openwebui/configure.py`: passed.
- `git diff --check`: passed.

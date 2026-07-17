# Served-model preflight validator

## Scope

- Added `tools/validate-served-model.py` as a read-only preflight CLI for a
  candidate `ullm.served_model.v1` manifest.
- The CLI loads the gateway's `served_model.py` directly without importing the
  gateway application startup path.
- Successful output is one canonical JSON record containing only the manifest,
  public model, format, worker binary, and product identities.
- Failure output is a fixed message that does not expose manifest content,
  deployment paths, environment variables, or loader exception details.

## Intended systemd use

The CLI accepts `--manifest PATH`, so a later deployment change can call it
from `ExecStartPre` before the gateway process is launched. This work does not
modify the service unit itself.

## Validation

- `python3 -m pytest -q tests/test_validate_served_model.py`
- Ruff check and format check for the tool and test
- `python3 -m py_compile tools/validate-served-model.py`
- `git diff --check`

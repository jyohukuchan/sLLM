# AQ4 profile operator historical actual-v9 state

## Scope

- Added a version-stable historical actual-v9 authority boundary without changing the current ready, quiet, operator, or output version pins.
- The historical state accepts only two states: all nine pre-execution paths absent, or the committed actual-v9 failure evidence fully present and sealed.
- Partial and mixed states fail closed.

## Executed-state validation

- Pins seal commit `00358807d7f400d621c11e20b942ecd4fbbd656f` and tree `6f0f61be424057a9fd8ca3c455d565e6dc3a6c08`.
- Verifies six sealed roots, 35 files, SHA256SUMS coverage, 0444 members, single links, and exact Git blob authority.
- Validates v9 result/audit semantics and audit self-hash, invocation 1/1, return code 1, no retry, failure capture, all operator/runner/validator/rocprof streams, and maintenance capture/launcher/rocprof exact-one counts.
- Binds the historical result and audit to operator manifest commit `2df19a16723df952c0be58a5cff4a1d86bb80d99`.

## Verification

- `python3.12 -m pytest -q tests/test_prepare_aq4_p3_profile_operator.py` — 14 passed.
- `python3.12 -m py_compile tools/prepare-aq4-p3-profile-operator.py tests/test_prepare_aq4_p3_profile_operator.py` — passed.
- `git diff --check` for the owned source and test files — passed.

No GPU command, service operation, or actual execution was performed.

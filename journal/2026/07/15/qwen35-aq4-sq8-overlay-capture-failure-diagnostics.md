# Qwen3.5 AQ4 SQ8 overlay capture failure diagnostics

## Outcome

- The promotion runner now preserves the capture stage, return code, signal, and timeout in failed maintenance evidence.
- Capture stdout and stderr retain full byte counts and SHA-256 digests, but display text is capped at 32 KiB per stream and decoded with invalid UTF-8 replacement.
- Each stream separates whole-source identity and prefix truncation from post-redaction display truncation. The complete canonical serialized stream object, not only its raw input prefix, is capped at 32 KiB.
- Lines containing password, secret, API key, authorization, bearer, or token credential markers are replaced instead of persisted.
- The existing failure receipt binds the maintenance evidence SHA-256, and `SHA256SUMS` binds both files. Successful runs do not contain `capture_failure` evidence.
- Evidence finalization rejects unsafe names, pre-existing output/staging paths, symlinks, and hard-linked members, and verifies regular single-link mode `0444` files before publication.

## Offline verification

- Promotion runner, preparation, lock helper, and receipt tests: 53 passed.
- `uvx ruff check`: passed.
- `python3 -m py_compile`: passed.
- Fault coverage includes nonzero exit, signal termination, timeout, truncation, invalid UTF-8, huge single-line and multi-line output, many short secret lines, both output streams, secret redaction, create-new, symlink, hard-link, mode, link count, SHA binding, and success without failure evidence.
- No GPU access, service operation, sudo command, or authorized execution was performed.

## Remaining diagnostic boundary

- The already completed failed execution cannot be diagnosed retroactively because its capture process stdout/stderr were not persisted.
- This runner change preserves diagnostics from the capture tool on the next newly authorized run. Non-JSON worker stderr discarded inside the capture tool remains outside this file's ownership and needs separate hardening before reauthorization if raw worker diagnostics are required.

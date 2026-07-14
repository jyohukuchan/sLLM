# P2 resident runner failure capture hardening

## Outcome

- The runner now streams resident-driver stderr directly to a private `O_EXCL` file instead of an unread pipe.
- Driver stdout is binary, line-bounded to 1 MiB, and unexpected/non-JSON responses retain only byte counts, SHA-256, and a bounded secret-scanned tail.
- Immutable driver process evidence records spawn PID/PGID/time, protocol stage and counters, ready state, exit code or signal, stderr bytes/SHA-256, cleanup signals, descendant process-group state, and lock inode continuity.
- Stderr larger than 1 MiB retains only a 64 KiB tail. Secret markers suppress all stderr text retention.
- The process is started in a new session and cleanup escalates from protocol shutdown to process-group `SIGTERM` and `SIGKILL`.

## Offline verification

- `tests/test_run_aq4_p2_resident_batch.py`: 37 passed.
- Fault coverage includes early exit, 2 MiB stderr, signal termination, response timeout, invalid JSON, secret stderr, mid-run exit, protocol OOM, and a hanging descendant.
- Bundle validator: 36 passed.
- Launcher: 7 passed.
- Execute launcher: 58 passed.
- Maintenance: 69 passed with 7 fake execute-CLI cases deselected because the canonical runner output from actual attempt v5 already exists and must not be removed or reused.
- All regenerated binding, execute-binding, base/profile ready, and base/profile dry-run `SHA256SUMS` verified.
- No actual execution, service operation, GPU access, HTTP request, or sudo command was performed.

## Trust-chain commits

- Runner: `4005c80e`
- Validator: `741616fd`
- Binding v4: `1d964602`
- Launcher: `79ce2aa1`
- Execute binding: `4507b20b`
- Maintenance harness: `d3d9eaee`
- Ready/profile artifacts: `a1547b6f`

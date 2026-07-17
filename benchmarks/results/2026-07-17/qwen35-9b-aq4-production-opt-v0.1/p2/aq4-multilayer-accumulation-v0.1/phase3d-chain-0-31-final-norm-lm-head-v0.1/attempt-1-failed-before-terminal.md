# Phase 3d CPU-only chain attempt 1 (invalid, preserved)

## Scope and command

- Scope requested `layer 0:31 + final norm + LM head` through the CPU-only
  `ullm-aq4-layer0-family-isolation` chain.  No GPU, service, systemd,
  production manifest, or P3 harness was used.
- The comparator command was the one recorded in `time-v.txt`, with
  `--chain-layer-range 0:31 --chain-include-final-norm-lm-head`, a 20-minute
  timeout, and the Phase 2 hybrid input fixture.

## Outcome

- This attempt is **invalid for fidelity measurement**: it completed all
  decoder work but terminated before emitting either terminal frame, so there
  is no valid `comparison.json`, layer table, or final-norm result to use.
- `/usr/bin/time -v` recorded wall `405.62 s` (6:45.62), maximum RSS
  `331188 KiB`, and process swap operations `0` before the deterministic
  terminal-read failure.
- The failing message is preserved in `compare/aq4.stderr.log`:
  `lm_head.weight` has no passthrough payload matching the old reader.

## Cause and corrective diagnostic change

- The offline package manifest declares `lm_head.weight` as an AQ4 tensor
  (`family=lm_head`, group size 8), not a BF16 passthrough tensor.  The first
  terminal extension incorrectly used the passthrough row reader.
- The diagnostic was corrected before retry: the new CPU reader seeks and
  dequantizes only the 34 fixed AQ4 LM-head rows, including row-scale
  overrides, and never materializes the 248,320-row vocabulary matrix.
- The corrected chain contract is schema v3 and reports
  `lm_head_weight_representation=aq4_dequantized_fixed_rows`.  A targeted
  loader unit test covers packed-nibble order, scale lookup, selected-row
  ordering, and row-scale overrides.

## Next action

- Build the corrected CPU-only binary and execute the same contiguous 0:31
  chain once in a separate `attempt-2` directory.  Preserve this failed
  attempt rather than overwriting it.

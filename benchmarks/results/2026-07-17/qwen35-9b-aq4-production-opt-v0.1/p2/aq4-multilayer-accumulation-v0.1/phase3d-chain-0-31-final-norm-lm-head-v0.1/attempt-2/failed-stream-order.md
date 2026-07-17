# Phase 3d CPU-only chain attempt 2 (invalid comparator ordering, preserved)

## Outcome

- The corrected AQ4 fixed-row LM-head reader succeeded: the chain emitted a
  full-hidden `final_norm` frame and then a fixed-34-row `lm_head` frame for
  the first timestep.  The AQ4 binary itself had no stderr and did not fail.
- This attempt is nevertheless **invalid for fidelity measurement** because
  the Python comparator expected all final-norm frames before all LM-head
  frames.  The chain's intended bounded-stream order is instead per timestep:
  `final_norm`, then `lm_head`.
- The comparator consumed the first final-norm frame and then rejected the
  following valid LM-head header while it was incorrectly expecting the next
  final-norm frame.  The exact header and error are preserved in
  `comparison.stdout.log`.

## Resources

- Wall `412.37 s` (6:52.37), maximum RSS `334748 KiB`, process swap
  operations `0` (`time-v.txt`).  This remained within the preflight estimate
  and did not OOM.

## Corrective diagnostic change

- The comparator now consumes terminal frames in the producer's
  timestep-interleaved order and computes source fixed-row logits only for the
  current timestep.
- `tests/test_aq4_multilayer_accumulation.py` includes a framed-stream test
  that exercises the exact `final_norm -> lm_head -> end` sequence.

## Next action

- Preserve this attempt and perform one final CPU-only 0:31 terminal-chain
  measurement in a separate `attempt-3` directory.

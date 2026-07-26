# Candidate history and non-results

## Selected candidate: serial GQA staging

The retained candidate is the serial GQA implementation described in the
top-level README. It was selected from full-model throughput at all five prompt
lengths, not from an attention-only timing probe. Its oracle comparison is
F32-byte exact for the observed 128--4095 token cases.

## Rejected wave32 / exact-tile64 direction

An earlier grouped candidate changed the reduction/arithmetic schedule while
trying to process the five Q heads in parallel. Although its first generated
token and top-1 matched in the smoke case, the real full-model oracle showed
material hidden/logit differences. For example, the recorded max absolute
hidden/logit differences were 2.292209625 / 1.353124619 at prompt 128 and
1.047966 / 0.437481 at prompt 4095. The detailed observations are in
[`numerical/exact-tile64-comparison.json`](numerical/exact-tile64-comparison.json).

This was rejected because it demonstrated a genuine changed arithmetic path,
not because it crossed a newly invented scalar numerical threshold. The
throughput window was intentionally stopped after the numerical diagnosis and
its partial 128-token timing is not used as a selection result.

## Attention-only probe

The final serial window began a CPU-reference attention smoke probe for
generic and candidate forms, then both were terminated with exit status 143 to
release the service window for full-model measurement. Those logs have no
timing conclusion and are not used to claim a kernel speedup. This follows the
decode barrier-pipeline lesson: an isolated attention result is insufficient
to select a full-model candidate.

## Physical-traffic claim deliberately not made

The generic source has five semantic K/V scans per shared KV head, while serial
GQA stages each K/V segment once per group. However no HBM/TCC counter was
captured, so any resulting physical-HBM reduction, cache reuse amount, achieved
occupancy, or memory-bound classification remains unconfirmed.

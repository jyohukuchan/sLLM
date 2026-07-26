# `SQ8_1` W8A8 speed disposition

No R9700/gfx1201 full-model decode throughput was measured for `SQ8_1` W8A8.
This is an explicit **not-measurable**, not a zero-speed or slow-candidate result:

- the `SQ8_1` architecture selection rule says **do not dispatch** W8A8 on
  gfx1201/R9700 in v0.1;
- no `SQ8_1` served selector, candidate manifest, or worker exists to pass to
  the generic promotion flow; and
- the task forbids V620/gfx1030 use, while its historical isolated M=1 kernel
  timing is not a full-model R9700 decode measurement.

The historical V620 evidence (`SQ8_0` 0.639007 ms; `SQ8_1` W8A8 0.249762 ms,
2.558x) is retained only as provenance in
`journal/2026/07/26/sq8_1-v620-optimization.md`.  It was neither rerun nor
used as a substitute for the required 15.294955751 tok/s full-model baseline.

The explicit user requirement to inspect actual W8A8 generations was still
performed independently in `sq8_1-w8a8-cpu-fixed-suite/`; its quality result
is a no-go due to a concrete factual explanation error, not due to top-1 or
logit thresholding.

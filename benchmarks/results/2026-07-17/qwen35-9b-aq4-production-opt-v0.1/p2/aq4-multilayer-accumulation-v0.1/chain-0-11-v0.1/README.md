# AQ4 multi-layer accumulation: CPU chain 0--11

This evidence extends the committed Phase 2 CPU-only chain measurement from
layers 0--3 to **0--11**. It reuses the exact 3-context / 9-output-record
fixture and runs the existing '--chain-layer-range 0:11' path. It does not
invoke a GPU, a resident service, a systemd unit, or an active manifest.

The raw AQ4 chain report is [aq4-chain/aq4-report.json](aq4-chain/aq4-report.json)
and the independently calculated BF16 source comparison is
[compare/comparison.json](compare/comparison.json). Both bind the same input
SHA-256, package manifest SHA-256, and probe binary SHA-256 as the prior
four-layer evidence. The topology contains self-attention layers 3, 7, and
11; layer 11 is included because the requested endpoint is inclusive.

The primary result is nonmonotonic: relative L2 increases from 0.042451384 at
layer 0 to a maximum of 0.125535705 at layer 5, then ends at 0.080826993 at
layer 11. The raw comparator classifies the curve as
'nonmonotonic_or_layer_jump' and its deliberately generous zero-origin linear
continuation reaches only 0.215538648 (35.0% of production 0.615).

- [growth-curve-with-deltas.csv](growth-curve-with-deltas.csv) is the complete
  layer-level relative-L2/cosine/max-abs/delta table.
- [extrapolation-analysis.md](extrapolation-analysis.md) records multiple
  continuation models, their assumptions, and the self-attention-boundary
  check.
- [analysis.json](analysis.json) is the machine-readable summary bound to the
  raw comparison report.
- [resource-estimate.md](resource-estimate.md) and
  [time-v.txt](time-v.txt) record the preflight and actual resource usage.

The 0:11 command completed in 2:37.13 with maximum RSS 332008 KiB, no swaps,
and exit status 0, well below the 45-minute safety cap. Full hidden/state
collections were not retained: the raw chain report records the existing
current-layer streaming persistence contract.

H8 is therefore assessed as **partially explains**: depth-wise AQ4 error may
contribute, but the extended CPU chain does not support H8 alone explaining
the production final relative L2 0.615. This statement is limited to the
authorized CPU diagnostic and does not advance to GPU, configuration, or fix
work.

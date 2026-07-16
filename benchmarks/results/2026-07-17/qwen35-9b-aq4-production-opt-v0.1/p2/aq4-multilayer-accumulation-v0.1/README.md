# AQ4 multi-layer accumulation diagnostic v0.1

This CPU-only Phase 2 diagnostic chains the manifest-derived decoder layers 0--3 against the BF16 source fixture.  It does not invoke a GPU, a resident service, or a production manifest, and it does not change the production post-norm epsilon.

The package manifest and the source `config.json` both identify the Qwen3.5-9B topology as repeated `[linear_attention, linear_attention, linear_attention, self_attention]`.  The self-attention layer indices are `3, 7, 11, 15, 19, 23, 27, 31`; therefore 0--3 is the smallest continuous range that includes one self-attention layer.

| layer | kind | relative L2 | cosine | max abs | records |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | linear_attention | 0.042451384 | 0.999106949 | 0.069626808 | 9 |
| 1 | linear_attention | 0.075075875 | 0.997374924 | 0.174329758 | 9 |
| 2 | linear_attention | 0.092594143 | 0.995868575 | 0.253928185 | 9 |
| 3 | self_attention | 0.106253646 | 0.994378165 | 0.202241421 | 9 |

The curve is monotonic with shrinking increments (`+0.032624491`, `+0.017518268`, `+0.013659503`), so it is classified as approximately linear or sublinear rather than superlinear or a self-attention jump.  Applying the deliberately simple linear model anchored at zero before layer 0 gives `0.106253646 * 32 / 4 = 0.850029167` at layer 31.  That is 138.2% of the independently observed production final relative L2 `0.615`; it overshoots by `0.235029167` (38.2%), but is large enough to classify H8 as **explains** rather than only partially explains.

The geometric continuation is recorded in the machine-readable comparison report but intentionally not selected: its early ratio is dominated by the first transition and produces an implausible `556.197559` extrapolation despite decreasing observed increments.

Only the current layer input/output sequence and layer-local state are retained while chaining; each f32 AQ4 output is streamed and compared immediately.  The reports preserve aggregate metrics and fixed-coordinate samples, not an all-layer hidden/state collection.  See [compare/growth-curve.md](compare/growth-curve.md) and [compare/comparison.json](compare/comparison.json) for the identity-bound raw evidence.

The independent post-norm epsilon control is summarized in [epsilon-control-summary.md](epsilon-control-summary.md).  Its layer-output effect is negligible at this scale and does not change the Phase 2 H8 classification.

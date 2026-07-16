# AQ4 chain 0--11: growth and extrapolation analysis

## Observed curve

The raw CPU-only comparison is 'compare/comparison.json'; this file derives its
numbers directly from that report and adds no new fixture or execution path.
'growth-curve-with-deltas.csv' records the full layer table and signed
layer-to-layer deltas.

| layer | kind | relative L2 | delta from prior | cosine | max abs |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | linear_attention | 0.042451 | — | 0.999107 | 0.069627 |
| 1 | linear_attention | 0.075076 | +0.032624 | 0.997375 | 0.174330 |
| 2 | linear_attention | 0.092594 | +0.017518 | 0.995869 | 0.253928 |
| 3 | self_attention | 0.106254 | +0.013660 | 0.994378 | 0.202241 |
| 4 | linear_attention | 0.119419 | +0.013165 | 0.992886 | 0.466560 |
| 5 | linear_attention | 0.125536 | +0.006117 | 0.992172 | 0.557333 |
| 6 | linear_attention | 0.077143 | -0.048393 | 0.997134 | 1.431293 |
| 7 | self_attention | 0.094488 | +0.017345 | 0.995626 | 1.429813 |
| 8 | linear_attention | 0.094775 | +0.000287 | 0.995630 | 1.403173 |
| 9 | linear_attention | 0.092623 | -0.002152 | 0.995813 | 1.345047 |
| 10 | linear_attention | 0.074961 | -0.017662 | 0.997391 | 2.475082 |
| 11 | self_attention | 0.080827 | +0.005866 | 0.996919 | 2.402580 |

The curve rises through layer 5, then becomes nonmonotonic. Its observed
maximum is 0.125535705 at layer 5, and the layer-11 value is 0.080826993.
Therefore the four-point monotonic/sublinear characterization does not survive
the extended range.

## Extrapolations to layer 31

The production final relative L2 used by the prior Phase 2 evidence is 0.615.
There are 20 layer transitions from the observed layer 11 to layer 31. These
models are descriptive CPU-chain continuations, not production-path
measurements.

| model | calculation | layer-31 estimate | fraction of 0.615 | use |
| --- | --- | ---: | ---: | --- |
| zero-origin linear (prior tool model) | 'L11 * 32 / 12' | 0.215539 | 35.0% | deliberately generous, still only partial |
| full-window mean delta | 'L11 + 20 * ((L11 - L0) / 11)' | 0.150601 | 24.5% | uses all 11 observed deltas |
| recent-4 signed mean delta | 'L11 + 20 * mean(d7->8, d8->9, d9->10, d10->11)' | 0.012522 | 2.0% | local continuation; the mean is -0.003415268 |
| recent-4 positive-part mean delta | 'L11 + 20 * mean(max(delta, 0))' over the same four deltas | 0.111591 | 18.1% | upward-biased local continuation that suppresses observed declines |
| early positive-delta geometric limit | from layer 5, 'r=(d4->5/d0->1)^(1/4)=0.658027'; 'L5+d4->5*r/(1-r)' | 0.137306 asymptote | 22.3% | saturation counterfactual; invalidated as a selected model by the negative layer-5->6 delta |
| self-attention block-end geometric level | 'r=sqrt(L11/L3)=0.872180'; 'L11*r^5' at layer 31 | 0.040793 | 6.6% | architecture-aligned supplemental model, based on outputs at layers 3, 7, and 11 |
| observed-max plateau | retain the layer-5 maximum | 0.125536 | 20.4% | simple no-reacceleration upper reference |

The signed full-window and recent-window models use the extended data rather
than fitting only the first four points. The two geometric calculations are
shown explicitly because their assumptions are testable: the initial
positive-delta geometric premise is contradicted by the layer-5->6 decrease,
and the block-end level model has only three points. Neither is selected as a
predictive production model.

To reach 0.615 from layer 11, the remaining 20 transitions would need an
average increase of +0.026708650 per layer. This is 7.66 times the full-window
signed mean +0.003488692, and it is larger than every observed positive
transition after layer 0. No continuation supported by the 12 measured points
reaches the production value.

## Self-attention boundaries

Range 0:11 contains self-attention at layers 3, 7, and 11 because the topology
repeats every four layers. The requested layers 3 and 7 show no consistent
superlinear jump:

| boundary | prior linear transition | into self-attention | following transition | interpretation |
| --- | ---: | ---: | ---: | --- |
| layer 3 | '1->2: +0.017518' | '2->3: +0.013660' | '3->4: +0.013165' | smooth shrinking increments; no distinctive attention jump |
| layer 7 | '5->6: -0.048393' | '6->7: +0.017345' | '7->8: +0.000287' | partial rebound after a linear-layer decrease, not renewed accumulation |
| layer 11 | '9->10: -0.017662' | '10->11: +0.005866' | n/a | another small rebound, still below the layer-5 maximum |

For the fixed-record samples retained by the comparator, 7 of 9 record-level
relative-L2 values decrease at layer 5->6. At layer 6->7, the aggregate rises
but 6 of 9 record-level values decrease, so the aggregate rebound is not a
uniform self-attention signature. This is descriptive only; it does not
identify a kernel or configuration cause.

## H8 assessment

**Verdict: partially_explains; H8 alone does not explain the production relative
L2 of 0.615.** The raw comparator retains its documented
'linear_conservative' classification because its most generous zero-origin line
gives 35.0% of the production value. The extended curve materially weakens the
former 'explains' conclusion: it is nonmonotonic, later-window growth is
nonpositive, and all data-grounded continuations above remain below 25% of
production except the deliberately generous zero-origin line.

This remains a CPU reference-chain result. It neither measures the GPU kernel
path nor audits a production configuration, and it does not authorize a Phase
3 investigation or a fix.

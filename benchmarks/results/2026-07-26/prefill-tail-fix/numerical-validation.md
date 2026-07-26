# Numerical validation

## Method

During the one isolated R9700 window, the same deterministic prompt-token
sequences were run through:

1. the clean AY-base binary, retaining its old M=1 tail behavior; and
2. the tail-fix binary, built from the same clean base plus only the scheduler,
   cursor-rewind, and report-contract changes.

Both use the same SQ8_0 artifact, package, `m128-chunk128` mode, and one
generated token.  The capture contains final hidden state, full logits, top-1,
and the generated token.  [`compare_oracles.py`](compare_oracles.py) compares
F32 little-endian bytes, SHA-256, max absolute error, relative L2, cosine, and
non-finite counts; its result is
[`numerical/comparison.json`](numerical/comparison.json).

## Exact-multiple invariant

The changed planner never rewinds for an exact M=128 multiple.  The oracle
therefore requires exact equality, and observed it:

| prompt | old calls | new calls | final hidden bytes | logits bytes | top-1 | generated token |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 128 | 1 | 1 | exact | exact | exact | exact |
| 512 | 4 | 4 | exact | exact | exact | exact |
| 1024 | 8 | 8 | exact | exact | exact | exact |
| 2048 | 16 | 16 | exact | exact | exact | exact |

All four exact comparisons have zero max-absolute error, zero relative L2,
and matching SHA-256 values for both captured tensors.

## M=1-tail comparison

| prompt | old calls | new calls | hidden relative L2 / max abs | logits relative L2 / max abs | top-1 / generated token |
| ---: | ---: | ---: | ---: | ---: | --- |
| 129 | 2 | 2 | 0.067284 / 1.175808 | 0.053214 / 1.022303 | exact / exact |
| 1000 | 111 | 8 | 0.013616 / 0.869091 | 0.007531 / 0.161585 | exact / exact |
| 4095 | 158 | 32 | 0.012594 / 0.692440 | 0.007181 / 0.259897 | exact / exact |

There are no non-finite values in any capture.  The tail tensors are not
byte-identical, which is expected from the executed path change: baseline
finishes through `execute_m1_stack_token()` and the paged-decode attention
route, whereas the candidate finishes through `execute_stack_chunk()` and
cached-prefix M=128 attention.  Those paths have different attention kernels
and floating-point operation/reduction order.  The change is thus not merely
a host scheduling no-op.  The deterministic next-token/top-1 agreement and
finite capture are recorded, while the full tensor differences remain visible
above rather than being claimed as equality.

## Cache-state / no-padding proof

This implementation does not pad and does not need a padding mask.  The final
fixed-width chunk is made entirely of a contiguous suffix of real token IDs.
Before it runs, all layer cache cursors and the resident serving cursor rewind
to its first real token.  The fixed chunk overwrites every rewound entry;
after it runs, `written_len` equals the logical prompt end, so attention cannot
read any physical data past that boundary.

The CPU test
`paged_decode_state_rewind_replays_a_real_suffix_without_stale_prefix_reads_cpu`
writes a cache prefix, rewinds it, writes a deliberately different suffix,
and checks a causal attention reference against the new logical cache.  It
would fail if stale pre-rewind values were reachable.  Targeted Rust tests and
the Python chunk/deep-boundary/performance validators passed; their command
and result summary are recorded in the repository commit associated with this
result.

An independent CPU full-model oracle was not run.  The requested alternative,
direct old-M=1 output comparison, is the GPU oracle table above.

# Validation status

## Executed non-GPU checks

The focused serving-runtime unit tests exercise the new scheduler contract:

- `serving_wide_chunk_scheduler_preserves_real_token_tail_replay` covers
  M=128/256/512/1024/2048 at N=4095 and checks the exact 1,280/640/320/160/80
  layer-expanded call counts, rewind locations, and real-token tail commits.
- `serving_4096_chunk_never_fabricates_a_4095th_prefill_row` proves that the
  N=4095/M=4096 case remains M=1 rather than padding a nonexistent row.
- `serving_fixed_chunk_selector_rejects_invalid_scheduler_widths` checks the
  bounds and power-of-two requirement.
- the existing M=128 tail replay test remains in the same test module and
  passed in the focused test executable.

The five focused tests passed with the `rocm-ck-gfx1201` feature enabled:

```text
cargo test -p ullm-engine --lib --features rocm-ck-gfx1201 \
  sq8_serving_runtime::tests::serving_wide_chunk_scheduler_preserves_real_token_tail_replay -- --exact
cargo test -p ullm-engine --lib --features rocm-ck-gfx1201 \
  sq8_serving_runtime::tests::serving_4096_chunk_never_fabricates_a_4095th_prefill_row -- --exact
cargo test -p ullm-engine --lib --features rocm-ck-gfx1201 \
  sq8_serving_runtime::tests::serving_fixed_chunk_selector_rejects_invalid_scheduler_widths -- --exact
cargo test -p ullm-engine --lib --features rocm-ck-gfx1201 \
  sq8_serving_runtime::tests::serving_m128_overlap_tail_geometry_and_divisible_geometry_are_explicit -- --exact
cargo test -p ullm-engine --lib --features rocm-ck-gfx1201 \
  sq8_serving_runtime::tests::serving_prefill_modes_bind_fixed_resident_widths_and_implementation_ids -- --exact
```

The source-level runtime gate was also exercised by the scheduler test: M=128
is admitted; M=256/512/1024/2048 is rejected before allocation until the
existing lower measured-M list is extended.

The direct gfx1201 CK helper shape probe was then run under the R9700 lock.
Every M=256/512/1024/2048/4096 × Q/O, K/V, gate/up, down row quantized and
projected successfully. It uses zeroed buffers and bypasses only the public
measured-M whitelist, so it supports extension of that whitelist but is not a
model-level numerical proof. See `wide-m-ck-shape-probe.jsonl`.

## Numerical and text fidelity

There is no M=256+ full-model execution in this evidence set, hence no new
hidden-state/logit comparison, non-finite observation, greedy token sequence,
or generated-text comparison. They are **unconfirmed**, not inferred from the
unit tests.

The relevant M=128 control remains BR's F32-byte-exact generic-versus-grouped
comparison at prompt lengths 128, 512, 1024, 2048, and 4095 (`max_abs=0`, no
non-finite values, same top-1/generated token). That evidence does not prove
the wider reduction partition; the next owner must compare it and then assess
actual generated text according to the lightweight-promotion policy rather
than apply a new numeric cutoff.

## Attention trace and decode

The M=128 control trace records 1,280 cached-prefix Flash2 calls at N=4095.
The wider values in `scheduler-contract.md` are planned unit-test counts only;
no M=256+ full-model GPU trace exists yet because the layer/stack/API
measured-M contract still blocks allocation and dispatch.

No decode path was changed. The closest compatible full-model control is BR's
post-change rerun: 27.411786 tok/s versus the BH reference 27.378731 tok/s.
It is retained as regression context, not falsely labelled as a new decode
measurement for the scheduler-only change.

## Build integration

`cargo check -p ullm-engine --example sq8_ck_serving --features
rocm-ck-gfx1201` passed after this change, compiling both the main serving
example and its `sq8_ck_serving_performance.rs` module. An earlier retry had
seen a transient compile error in BX-owned `runtime/src/ullm_runtime_parts/
part_01.inc` while that independent KV-dtype work was in flight; this task did
not modify that file, and the subsequent check passed without changing it.

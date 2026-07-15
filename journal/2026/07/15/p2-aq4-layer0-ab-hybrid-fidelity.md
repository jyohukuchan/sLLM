# P2 AQ4 layer0 A/B hybrid fidelity

## Scope

- Fixed three-step input and identical production AQ4 package.
- CPU-only baseline, A-only source BF16 raw projection override, B-only override, and A+B combined override.
- Override boundary is after the production A/B matvec and before the production A_log/dt_bias gate/beta transform.
- Each run starts from zero recurrent state. Normal wrapper and worker keep every diagnostic argument at `None`.

## Result

- A-only layer-output relative L2: `2.820126348076646e-05`
- B-only layer-output relative L2: `1.2178440040512472e-04`
- A+B layer-output relative L2: `1.2445526711278346e-04`
- Existing QKV: `7.890094626902091e-04`
- Existing Z: `6.983383610254553e-04`

B contributes about 4.32 times the A-only layer change. A+B is about 15.8% of QKV and 17.8% of Z. The next combined isolation candidate is QKV+Z; adding B afterward is more informative than promoting A or B alone. No threshold or promotion decision was evaluated.

## Verification

- `cargo check -p ullm-engine --bin ullm-engine`: pass
- `cargo test -p ullm-engine --bin ullm-engine -- --test-threads=1`: 26 passed
- default three-step smoke stdout SHA256: `9ac224cc444569bb9e5c4c493eacf4007c06c862c03466da31a058a123e4ad9b` (unchanged)
- artifact is CPU-only; no GPU, service, P3, or Gate action was performed.

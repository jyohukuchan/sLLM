# SQ8 numerical gate v0.2 consumer result

- Status: `test_only_harness_verification`
- Frozen JSON SHA-256: `64a43c032570bed8086e3c441b0774cc470c5ab1e8c67f99e02af2b6307f72bf`
- Complete frozen coverage: `False`
- Actual coverage: `{"decode_blocks_of_64": 2, "hidden_layer_probe_or_mandatory_positions": 24, "mandatory_boundary_positions": 0, "prefill_checkpoints": 5, "primary_decode_positions": 174, "primary_decode_streams": 1}`

All measured metrics passed, but this is explicitly not an admission result because it uses incomplete coverage and/or test-only capture manifests.

## Recorded interpretation details

- P99 uses F64 nearest-rank ceil(0.99*n), because the frozen JSON fixes P99 but not an interpolation convention.
- Bootstrap seed is the first eight big-endian bytes of SHA-256(seed_domain UTF-8); RNG is NumPy PCG64, because the frozen JSON fixes the seed domain but not a PRNG/tie rule.
- Bootstrap pairs each candidate with the scalar-median control repetition; ties choose the lower repetition index.
- A hard candidate-only top-1 regression requires every control repetition to equal reference top-1, which is the conservative reading of 'control top-1 equals reference top-1'.
- max-abs ULP floor uses the maximum reference absolute value inside the evaluated tensor scope.

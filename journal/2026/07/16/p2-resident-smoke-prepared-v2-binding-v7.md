# P2 resident smoke prepared-v2 / binding-v7

## Scope

- Runtime authority: commit `43ba16f2347a45caba8a60cac2189714118db280`, tree `72392a7114f5968d6c2ad05e24762a6790000013`.
- Directional HIP copy input: blob `316d3ae5c13f79678fb8256aa8c66ea7e154660f`, raw SHA-256 `db138bfaf33f59708f24edbec8352a39fe809ff39422d5b742399752c8fa9f5f`.
- GPU commands, model loads, service operations, actual runs, deployment, launcher work, and maintenance/operator work were not performed.

## Reproducible resident driver

Two clean detached builds used `CARGO_BUILD_JOBS=1`, `CARGO_INCREMENTAL=0`, `--locked`, release profile, and independent initially absent target directories:

- `/tmp/ullm-profile-v10-resident-target-a`
- `/tmp/ullm-profile-v10-resident-target-b`

The outputs were byte-identical:

- SHA-256: `d7458fcdf8553871cac00123413676625c61eff2fdee3be9a440e656f05bcc1e`
- bytes: `3505000`
- ELF Build ID: `033ce9b214e2149861a8fcf0381c27bbac5bf1d1`

## Source authorities

- Prepared bootstrap: `tools/run-aq4-p2-resident-prepared-bootstrap.py`, commit `410d6fa1876a6772215604ba765ae1d6a91d67b9`, tree `73fa76d74e042c23c353d6e25172f62cbb364995`, blob `a12032be24ffdabd703d304df3e4ee825bc71634`, raw SHA-256 `62cf9cd77863d18158afad8955b7d09c2c0f8b09046869bacb25c91c789878e0`.
- Binding actual generic runner: `tools/run-aq4-p2-resident-batch.py`, commit `d367b6da07393f55c720ded7250bda8cdc402a79`, tree `8fea6bf90e8ad99c7ed36c719b8b4ad204ce73df`, blob `fed94b749790cdbf6a61e33f3f9e95ebd73502e0`, raw SHA-256 `98e324414d9e2d7e6db5b066209e6f7c6734e391502ae81ecd1809e8ec558e7f`.
- Final validator/generator authority: commit `e36a03ad423a0bb45cc1e4de67d3ca4fddfacdbc`, tree `9189252d996e2eda05761f650960224676867811`, blob `5ee4278a58b18454cf714da6bfe540f5d2ff832c`, raw SHA-256 `15a65fed6d182e706473821f128fbae02214ab0bf988bb7c1f363f69233e9904`.

The prepared bootstrap and binding actual runner are intentionally separate authorities. The bootstrap only validates the in-construction v4 pre-run members and emits a dry-run plan; the generic runner validates the completed immutable bundle through the pinned validator.

## Fresh artifacts

- `resident-one-case-smoke-prepared-v2`: schema `ullm.aq4_p2_resident_smoke_binding_bundle.v4`, run ID `p2-r9700-resident-one-case-smoke-prepared-v4`.
  - `SHA256SUMS`: `27579326b1ba703585d4683ddcabf47676debeca58878e54a2ae2cccee6e99b9`
  - `bundle.json`: `b947e43d6aff609967ea9b0909aa14a689b4f35791a7903134453c42ede8fb38`
- `resident-one-case-smoke-binding-v7`: schema `ullm.aq4_p2_resident_smoke_binding.v7`.
  - `SHA256SUMS`: `e922a38142380bbee3e7e4db18d195f93aad86f4b1123e0034f9149edf1f9918`
  - `binding-manifest.json`: `3b99dcfd11f9c4726a8531f9f828ec62dd84fabe577b6b529636ee0b66918579`

Both roots are mode `0555`. All members have link count one; the resident driver is mode `0555`, and every other member is mode `0444`.

## Validation

- Formal prepared validation: passed.
- Formal binding validation: passed.
- `SHA256SUMS` complete-member checks: passed for both roots.
- Prepared bootstrap archive versus pinned Git object: byte-equal.
- Binding runner and validator archives versus pinned Git objects: byte-equal.
- `tests/test_prepare_aq4_p2_resident_smoke_bundle.py`: `67 passed`.
- Historical prepared-v1 `SHA256SUMS` remained `12e72ded1804ca075fde19f7ceca4d02cde9df2558489288e8ff850caf1a2b2b`.
- Historical binding-v6 `SHA256SUMS` remained `684e3be7a50393b3b8c7b045c3719727b4ea6f1ceaabfd3476c3158215076e50`.

## Next action

Perform independent offline artifact QA, then consume binding-v7 only from the separately authorized immutable-launcher stage.

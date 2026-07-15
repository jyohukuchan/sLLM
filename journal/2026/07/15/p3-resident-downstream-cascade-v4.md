# P3 resident downstream cascade v4

## Scope and source boundaries

- Served-manifest authority source/test commit: `1f5b12803759e6596021dfd8c5e1455f2635f586`.
- Resident driver clean source/build commit: `81ceebb13518f590b5dbf439cd00b35e508c1c3f`.
- Resident binary SHA-256: `458b8603d6823a1c20ea93e7c0d757c8910f3c36c9a2a34ab536853c0c9e7d34`.
- Prepared/binding artifact commit: `bad728000405a711dec4faf10d4a60393bf9d7e8`.
- Downstream launcher/maintenance/test update: `288b165c707413aac01753b8254ea98fe843308f`.
- Downstream QA provenance update: `9f9d5d5d`.
- Complete capture QA command update: `92e7c6ee76ebb0ef44d41f928eb8a93131146817` (tree `928fed24ac4100ad918368498d3e547f4ee3132c`).

The launcher now binds the regenerated prepared and binding roots, the current runner and validator, and the complete resident source/build identity. The resident manifest trust root is compared as an exact document, including source tree/blob/SHA-256, binary bytes/build ID, and all build metadata. Profile execute/evidence, maintenance evidence, capture output, and profile dry-run targets were advanced to fresh v4 paths.

## Official fresh artifacts

All generated roots are mode `0555`, their regular members are mode `0444`, every `SHA256SUMS` check passes, and canonical loader readback passes.

- `resident-one-case-smoke-execute-binding-v1`: `SHA256SUMS` SHA-256 `fa935b6d8ff68c4b2190e4a6bdd706ed209357e8d77a7005d0334a0091cca27c`.
- `resident-one-case-smoke-ready-v1`: `SHA256SUMS` SHA-256 `c1483588d05583d386fc110a4c144959b4694b1d002430a85c84f505c1a5a5cd`.
- `resident-one-case-smoke-profile-ready-v1`: `SHA256SUMS` SHA-256 `ab9938b41eb129971e9aa8c57d6604abb34f61d05b9ac3fdbac04374f8f1ba46`.
- `resident-one-case-smoke-ready-dry-run-v1`: `SHA256SUMS` SHA-256 `483dc0fda13f7dc36ed9ec2c4f5904755036ca48d05514eda51e665b9adcb8db`.
- `resident-one-case-smoke-profile-ready-dry-run-v4`: `SHA256SUMS` SHA-256 `2b6e7cc743230703dd5d85bbbece0e0f7eaf3e036d669724502ced46f29b9b59`.

Both dry-run evidence documents have `status=passed`; all process counters are zero and `service_touched`, `gpu_command_executed`, and `model_load_executed` are false.

## QA

- Exact resident trust-chain Python suite: `346 passed`.
- Diagnostic capture suite with the canonical prepared resident driver: `29 passed` (no skip).
- Remaining attested Python suites passed in the combined run; the exact attestation total is 459 Python plus 22 Rust tests, `481 passed`.
- Resident driver unit suite: `22 passed`.
- `served_model::tests`: `3 passed`.
- SHA256SUMS, artifact modes, canonical execute/ready readback, Git commit/tree/blob provenance, and fresh-output absence checks passed.

## Preserved actual failure and execution boundary

Commit `ec4b0a36e9f10db524cb24ef2b2d5e3bf638249d` and its profile actual v3 failure artifacts remain unchanged. The next v4 profile execute output, execute evidence, maintenance evidence, and capture directory are absent. Only profile dry-run v4 exists. No actual execution, GPU command, model load, sudo, or service operation was performed in this cascade.

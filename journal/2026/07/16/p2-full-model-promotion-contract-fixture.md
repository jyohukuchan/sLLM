# P2 full-model PromotionContract fixture repair

- Updated the full-model negative-test fixture for the current typed `PromotionContract` schema.
- The package-only, non-authorized fixture explicitly carries `None` for authorization audit, authorization lineage, and bridge readiness; production parsing and validation remain strict and unchanged.
- No SQ8 policy, Python fidelity protocol, GPU execution, service operation, or holdout execution was changed or performed.
- `ullm-aq4-p2-full-model` passed 18 tests with `jobs=1`; focused CurrentV2 typed/tamper tests passed 2 tests; the Python fidelity protocol passed 15 tests.
- The full Rust library run reached the final long-running production provenance CPU test with preceding tests green and the isolated HIP test ignored, then was boundedly interrupted after the final test exceeded two minutes.

# P2 engine bridge

- AQ4 session now validates requested prefill M on the bounded grid 1/8/16/32/64/128; default remains 128.
- Added privacy-safe request-terminal execution audit with requested/resolved M, observed token/request widths, phase/internal batch counts, and prepare/commit/discard/error/cancel/reset lifecycle counters. Existing registry operation audit remains nested and unchanged.
- Added CPU unit coverage for grid validation, configurable chunk tails, publish failure, cancellation, and reset failure.
- Validation: `cargo test -p ullm-engine qwen35_aq4_session --lib --no-default-features` (35 passed). No GPU or live worker execution performed.

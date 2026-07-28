# Qwen3.5 AQ4_0 protected-probe recheck

## Scope

This closes the post-`03dcbd32` production-regression gap without changing the
Qwen runtime.  The exact protected 128-token fixture was recovered from the
existing prior-run harness at
`benchmarks/results/2026-07-27/qwen35-moe-loader-wiring-v0.1/run-attempt2-r9700.sh`,
line 34 (`baseline_ids`).  No alternate prompt or token sequence was used.

The current source probe was rebuilt with:

```text
cargo build --release -p ullm-engine --bin ullm-qwen35-aq4-baseline-probe
```

It then ran under `/run/ullm/r9700.lock` on HIP ordinal 1 / gfx1201 with the
same complete AQ4_0 guarded-operation contract as that harness.  The first
attempt omitted four required group8/WMMA guard variables and failed closed at
load time (`Aq4MatvecBatch`); the recorded result is the subsequent complete,
successful guarded run.

## Result

`raw/qwen35-aq4-probe.json` is byte-identical to the retained expected output:

```text
sha256  30865287e7525f4b24449ec24be3aa7619bfbbbbf48522cf2f67f9e58379b588
top-1  220 / 8.529029846191406
```

The rebuilt source probe itself is
`05fee55bfde34ce523aec3cc4dd782c7659b0b71dbfdcac852663676e1155080`.
The untouched frozen production worker remains
`5a274733710d9b80a24d34a31ec6a99ac0b2d1e8fcce45904e906926a0e2e903` at
`/opt/ullm/aq4-overnight-consolidation-v0.1/releases/aq4-consolidated-840a1c7a-5a274733/ullm-aq4-worker`.

Therefore DU's 512-wide Gemma-only change did not alter the protected Qwen
256-wide production path's observable probe bytes.

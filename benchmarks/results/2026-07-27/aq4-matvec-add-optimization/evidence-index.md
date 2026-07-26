# Selected immutable result index

The full R9700 run root is retained locally under
`r9700-window-20260727T061520+0900/`.  This index binds the compact,
content-free acceptance evidence committed alongside the analysis.  Full
profiler CSVs, telemetry streams, and prompt/response bodies remain in that
run root and are intentionally not copied into this index.

| Relative path | SHA-256 | Purpose |
|---|---|---|
| `decode/summary.json` | `64f30538df093ef01ae44fc3f1bcc347faf6015c9236664eab3aedabbcb88d84` | Unprofiled full-model direct/grouped A/B throughput |
| `prefill/summary.json` | `a06e09d26555bdf744af7a9f8e45749991b8a97e1e67e8e3cf3ecca45fea0366` | Cold p=2048/M=128 prefill A/B |
| `launch-invariant/shuffle-reference/assertion.json` | `19a53ee6685dbf81143d20166dbe6e4c6a33796ea4898285052690949bc16447` | Reference 292-module/64-add invariant |
| `launch-invariant/group-specialized/assertion.json` | `fc50541e83ed1873dfbcb3f83b6af5ca028cf47138a93a6621277cf19a0ff091` | Candidate 292-module/64-add invariant |
| `launch-invariant/shuffle-reference/accounting.json` | `4d0f8ebb77748e6d1720af5f34a9e4732f2bccf8d48e935de399621141605ccc` | Reference trace diagnostic |
| `launch-invariant/group-specialized/accounting.json` | `fc6949cce3c88b64c29613d0a26cb9cdd55ad3a5feb78c7a6a18aace23772c48` | Candidate trace diagnostic |
| `promotion-20260727T064100+0900/outcome.json` | `77c38cec112b1fac7ab91c570f380b408478d09c8c46e1be925cdc2cab9b2800` | Lightweight-promotion `activated` outcome |
| `promotion-20260727T064100+0900/comparison.json` | `46f55e0ada3561e292fc497be4bda5fca3a062b9c9f2d660b73cacb1f7de1ce0` | Ten-case content-free quality comparison summary |
| `promotion-20260727T064100+0900/service-events.json` | `d006445a37e9a8a6ce96eaaceec9d30ea1eff48a3f169487a14795f1c10c3882` | Candidate activation restart record |
| `promotion-20260727T064100+0900/candidate-readiness.json` | `307f0e707474b464b60b058fd2cf011cdea90ead96a30d5d5dc5d141f7b0da09` | Candidate readiness probe |

At final verification, the active manifest had subsequently returned to
`3507102fd3015f47204a4f3256b818c58788eadb5573e5d5fe727a076cb1b3e7`.
That later state is documented in `post-window-results.md`; it is not a
promotion-tool rollback because the transaction outcome remains `activated`.

# uLLM measurement-driver identity

The measured executable was a standalone, measurement-only Cargo binary.  It
was built against the clean uLLM runtime checkout below; it is not a change to
the product artifact, the active manifest, the systemd unit, or `/opt/ullm`.

| item | value |
| --- | --- |
| runtime checkout HEAD | `0216b131cf5377d90125abd9c1c49c5a8a210511` |
| runtime checkout status at capture | clean (empty `ullm-clean-status.stdout`) |
| standalone driver Cargo.toml SHA-256 | `753c71f5edf9fce10c850d92e48c4bf44dddb06db6948243500b2854c7e3ba3b` ([source](ullm-prefill-driver-Cargo.toml)) |
| standalone driver main.rs SHA-256 | `0c86a60c358193fc763fb4ccf2eee623d2f2c9a2a8b88624f1ad3393e8b95469` |
| measured binary SHA-256 | `f045abcb2b87bcffab7ca7554c039c43c90e9c5cde6ea88dff36aa8afcf372ee` |
| driver source | [ullm-prefill-driver-main.rs](ullm-prefill-driver-main.rs) |

The driver fixes `Sq8ServingPrefillMode::FixedM128Chunks`, begins its timer
after `session.start`, advances only while the state is `Prefilling`, and
finishes the request/reset outside the timed interval.  It uses an
`Instant`-based synchronized loop for tok/s.  ROCTx ranges are retained only
as trace labels; no profiler range duration is used as throughput.

The command used `--phase prefill --prompt-tokens N --repeats 5`; all required
SQ8 HIP-kernel guard variables and `HIP_VISIBLE_DEVICES=1` are visible in
[commands.md](../commands.md) and each raw condition's `command.txt`.

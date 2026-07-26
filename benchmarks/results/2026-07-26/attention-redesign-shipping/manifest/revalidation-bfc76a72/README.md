# Revalidation at `bfc76a72`

These CPU-only checks rerun the served-model execution-contract coverage while
the R9700 was reserved by another task.  They do not launch a worker, access
the active manifest, or change service state.

- `gateway-tests.txt`: the pinned gateway source and its two manifest/settings
  modules passed 99 tests.
- `tooling-tests-main-root.txt`: the generator, validator, and lightweight
  promotion-wrapper modules passed 54 tests from the repository root.
- `rust-served-model-tests.txt`: seven Rust `served_model` tests passed from
  the clean `bfc76a72` worktree.
- `aq4-full-model-cargo-check.txt`: the AQ4 full-model binary passed
  `cargo check` from that same clean worktree.

`tooling-tests.txt` records one deliberately retained environmental result:
running all 54 tooling tests directly from the detached worktree produces one
failure because an unrelated AQ4 reasoning fixture intentionally hard-codes
the primary worktree's `target/reasoning-v2` path.  The identical suite passes
at that prescribed primary root (`tooling-tests-main-root.txt`), so this is not
an execution-contract regression and is not concealed.

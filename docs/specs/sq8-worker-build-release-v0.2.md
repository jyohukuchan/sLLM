# SQ8_0 Worker Build Release v0.2

Status: ratified production contract

Date: 2026-07-24

## 1. Scope and version boundary

This contract defines the clean-build receipt and the complete sealed worker
release used by the independent Qwen3-14B-FP8 `SQ8_0` serving promotion.
It does not define an AQ4_0 worker, an AQ4_0 partial-FP8 overlay, a campaign
authorization, or an activation authorization.

The v2 schema family is:

- `ullm.sq8_worker_build_receipt.v2`;
- `ullm.sq8_worker_build_provenance.v2`; and
- `ullm.sq8_worker_release_seal.v2`.

The v1 schemas remain frozen historical formats. In particular, a v1 receipt
locates `worker.path` by its recorded absolute path. Moving a v1 receipt does
not relocate its worker. A validator must not reinterpret that field as
relative and must not silently upgrade v1 bytes to v2.

## 2. Publication unit

The only relocatable unit is one complete release directory. Its exact member
set is:

```text
README.md
SHA256SUMS
SEALED.json
build-provenance.json
build-receipt.json
ullm-sq8-worker
```

The directory mode is `0555`. `ullm-sq8-worker` is a regular file with mode
`0555` and link count one. Every other member is a regular file with mode
`0444` and link count one. No symlink, hard link, device, socket, FIFO,
subdirectory, absent member, or extra member is accepted.

`build-receipt.json`, `build-provenance.json`, and `SEALED.json` are strict
canonical ASCII JSON: keys sorted, no insignificant whitespace, no duplicate
keys, no non-finite number, and one trailing LF. Publication is no-replace.

## 3. Build receipt v2

The receipt has exactly these root fields:

```json
{
  "schema_version": "ullm.sq8_worker_build_receipt.v2",
  "source": {},
  "build": {},
  "inputs": [],
  "worker": {}
}
```

`source` has exactly:

- `repository_root`: the canonical absolute source path observed at build
  time;
- `commit`: lowercase 40-hex Git commit;
- `tree`: lowercase 40-hex Git tree;
- `detached`: `true`;
- `worktree_clean`: `true`; and
- `status_sha256`: SHA-256 of the empty byte string.

`repository_root` is an audit fact, not a runtime locator. It remains unchanged
when the release or a matching sealed source checkout is relocated. A v2 live
validation must receive the current source root explicitly; it must never
dereference the recorded historical path as an implicit fallback.

`build` has exactly `argv`, `environment`, and `result`. `result` is
`"success"`. `argv` is an absolute or otherwise named Cargo executable followed
by exactly:

```text
build --locked --release -p ullm-engine --bin ullm-sq8-worker
--features rocm-ck-gfx1201
```

`environment` has exactly:

```text
CARGO_BUILD_JOBS
CARGO_INCREMENTAL
CARGO_TARGET_DIR
CUDA_VISIBLE_DEVICES
GPU_ARCH
HIP_VISIBLE_DEVICES
ROCM_PATH
ROCR_VISIBLE_DEVICES
RUSTC_WRAPPER
SOURCE_DATE_EPOCH
ULLM_HIP_VISIBLE_DEVICES
```

The isolation values are `CARGO_INCREMENTAL=0`, `GPU_ARCH=gfx1201`, all four
recorded GPU visibility variables equal to `-1`, and `RUSTC_WRAPPER=null`.
`CARGO_BUILD_JOBS` is a positive decimal integer. `SOURCE_DATE_EPOCH` is a
decimal epoch derived from the source commit. `CARGO_TARGET_DIR` and
`ROCM_PATH` are absolute build-audit paths, not deployed runtime locators.

`inputs` is the bytewise-sorted, duplicate-free list of exactly these safe
repository-relative paths and their lowercase SHA-256 values:

```text
.cargo/config.toml
Cargo.lock
Cargo.toml
crates/ullm-engine/Cargo.toml
crates/ullm-engine/src/bin/ullm-sq8-worker.rs
crates/ullm-engine/src/reasoning.rs
crates/ullm-engine/src/served_model.rs
crates/ullm-engine/src/sq8_sampling.rs
crates/ullm-engine/src/sq8_serving_runtime.rs
crates/ullm-engine/src/sq8_worker_backend.rs
crates/ullm-engine/src/sq8_worker_protocol.rs
crates/ullm-engine/src/sq8_worker_runtime.rs
crates/ullm-runtime-sys/Cargo.toml
crates/ullm-runtime-sys/build.rs
```

Each input object has exactly `path` and `sha256`. Absolute paths, empty
components, `.`, `..`, backslashes, duplicate entries, and symlink escapes are
rejected.

The builder captures the Git top-level, commit, tree, detached/clean state,
commit timestamp, and every input byte identity before invoking Cargo. After
Cargo, all toolchain queries, and the sealed-worker copy, but before publishing
the receipt or seal, it captures the complete source identity again and
requires exact equality. A source, input, branch, status, commit, tree, or
timestamp change aborts without a receipt or seal.

`worker` has exactly:

```json
{
  "relative_path": "ullm-sq8-worker",
  "bytes": 1,
  "mode": "0555",
  "nlink": 1,
  "sha256": "<lowercase 64-hex>"
}
```

`bytes` above is illustrative and is the positive exact live size. The only
valid locator byte string is `ullm-sq8-worker`. It is resolved from the
directory containing the exactly named `build-receipt.json`. An absolute
locator, alias filename, slash, nested path, dot component, traversal,
backslash, or normalization-equivalent spelling is rejected.

## 4. Build provenance v2

`build-provenance.json` has exactly `schema_version`, `source`, `build`, and
`worker`.

Its `source` repeats the receipt's repository root, commit, tree, detached
state, clean tracked/untracked states, and exact input map. Its `build` repeats
the command and environment and records the build working directory, target
directory, start/finish nanoseconds, toolchain outputs, rejected ambient
compile overrides, hermeticity statement, and successful result.

`toolchain` has exactly `cargo`, `rustc`, `cxx`, and `hipcc`. Each entry has
exactly a canonical absolute audit `path`, the executable `sha256`, and the
first-line `version` output. These recorded paths are not runtime locators.
`ambient_compile_overrides_rejected` is the bytewise-sorted list containing
exactly:

```text
CARGO_ENCODED_RUSTFLAGS
CARGO_PROFILE_RELEASE_CODEGEN_UNITS
CARGO_PROFILE_RELEASE_DEBUG
CARGO_PROFILE_RELEASE_LTO
CARGO_PROFILE_RELEASE_OPT_LEVEL
CARGO_PROFILE_RELEASE_PANIC
CFLAGS
CPPFLAGS
CXXFLAGS
LDFLAGS
RUSTC
RUSTC_BOOTSTRAP
RUSTC_WRAPPER
RUSTDOCFLAGS
RUSTFLAGS
```

Its `worker` repeats every receipt worker field and adds exactly:

```json
{
  "protocol": "ullm.worker.v2",
  "format_id": "SQ8_0",
  "model_id": "ullm-qwen3-14b-sq8"
}
```

The repeated receipt, provenance, and live file identities must be equal.

## 5. Hash inventory and seal

`SHA256SUMS` is exact ASCII and contains these records in this order:

```text
<sha256>  README.md
<sha256>  build-provenance.json
<sha256>  build-receipt.json
<sha256>  ullm-sq8-worker
```

Each digest is recomputed from the member bytes. No additional record,
alternate order, alternate spacing, or missing final LF is accepted.

`SEALED.json` has exactly:

```json
{
  "schema_version": "ullm.sq8_worker_release_seal.v2",
  "source_commit": "<receipt commit>",
  "source_tree": "<receipt tree>",
  "worker_sha256": "<worker sha256>",
  "build_receipt_sha256": "<receipt file sha256>",
  "build_provenance_sha256": "<provenance file sha256>",
  "sha256sums_sha256": "<SHA256SUMS file sha256>",
  "complete": true
}
```

All values are independently recomputed. A receipt alone is not a complete v2
release and cannot enter serving promotion.

## 6. Relocation and source validation

Relocation copies the complete directory byte-for-byte to a new, previously
absent directory. It does not hard-link members and does not rewrite the
receipt, provenance, `SHA256SUMS`, or seal. After the copy, the original may be
unavailable. The relocated directory must still pass the complete-release
validator, and the receipt and seal byte strings and SHA-256 values must remain
identical.

When live-source validation is requested, the caller explicitly supplies the
current sealed source root. The validator requires that it is the Git
top-level, is detached at the receipt commit and tree, is tracked- and
untracked-clean, and that every recorded input has the recorded digest. A
missing, dirty, branch-attached, wrong-commit, wrong-tree, non-top-level, or
ambiguous source root fails closed.

An offline archival check may set live-source verification to false. That mode
still validates the exact release bytes and schemas; it does not authorize a
promotion, campaign, or activation.

## 7. Staging order

The final clean commit is built first into a private release. The complete
release is then copied and validated at its final staged path. Only after that
validation may the operator create, in order:

1. the served-model profile with the staged worker's absolute path;
2. the ephemeral manifest;
3. the CPU-case report;
4. the serving promotion evidence and receipt; and
5. the final candidate manifest.

Those later artifacts intentionally contain absolute paths and are not
relocatable. Generating any of them against the private pre-staging path and
then moving only the worker release is invalid. Every producer and validator
in this sequence passes the same explicit current sealed source root.

This staging sequence prepares evidence only. It does not write production
`active.json`, control `ullm-openai.service`, run a GPU campaign, claim a
campaign authorization, or perform final activation.

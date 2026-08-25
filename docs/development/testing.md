# Host build and test entry points

## Scope

Phase 1 provides a CPU-only repository skeleton and the required H0, H1, and H2 host rows. These commands do not prove HIP compilation, GPU execution, numerical kernel correctness, model execution, compatibility, or performance. H3 and all G/P tiers remain separate later phases.

The host-native library is an intentional C++17 stub. It links through Cargo and the versioned C ABI, reports the HIP backend as unavailable, and never falls back to CPU numerical execution.

## Toolchain and dependencies

- Rust development toolchain: `1.97.1`, selected by `rust-toolchain.toml`.
- Rust MSRV: `1.85.0`.
- Python host CI: `3.12.10`.
- Python packages: Python 3.12/Linux x86_64用のtransitive dependencyを含むexact versionとwheel/sdist SHA-256を`ci/requirements-host.txt`へ固定する。
- CMake: 3.21 or newer.
- C++: C++17.

Install dependencies before running the isolated test commands:

```bash
rustup toolchain install 1.97.1 --profile minimal --component rustfmt --component clippy
rustup toolchain install 1.85.0 --profile minimal
python3 -m pip install --disable-pip-version-check --no-input \
  --require-hashes --only-binary=:all: --no-deps \
  -r ci/requirements-host.txt
```

Dependency installation is not part of a test row. Host test commands use only local manifests, source, fixtures, and the Cargo lockfile.

## Required host rows

Run each row independently. Reports contain the Git identity, manifest hashes, seed, actual collected/selected/outcome counts, per-command timings/resources, and diagnostics. Each required command runs in a tested Linux network namespace with no external route; failure to establish the boundary is an infrastructure failure. H2 additionally verifies a 4 GiB address-space limit inside that namespace.

A dirty developer checkout must opt in explicitly. These reports are labeled `local-development`, `immutable=false`, and cannot be consumed by the strict aggregator:

```bash
python3 ci/tools/run_host_suite.py --row h0 --output-dir .local-artifacts/ci/h0 --allow-dirty-local
python3 ci/tools/run_host_suite.py --row h1 --output-dir .local-artifacts/ci/h1 --allow-dirty-local
python3 ci/tools/run_host_suite.py --row h2 --output-dir .local-artifacts/ci/h2 --allow-dirty-local
```

Immutable evidence requires a clean checkout and explicit equality between reviewed, tested, workflow, and checked-out SHA. The strict runner checks tracked and non-ignored untracked state both before and after the registered commands; a command that mutates the checkout produces a failed result instead of immutable `PASS` evidence:

```bash
candidate_sha=$(git rev-parse HEAD)
python3 ci/tools/run_host_suite.py \
  --row h0 --output-dir .local-artifacts/ci/h0 \
  --strict-ci \
  --expected-reviewed-sha "$candidate_sha" \
  --expected-tested-sha "$candidate_sha" \
  --expected-workflow-sha "$candidate_sha"
```

Exit codes are:

- `0`: all registered cases passed.
- `1`: a test, lint, contract, timeout, or numerical-oracle case failed.
- `2`: required infrastructure or tool execution failed.
- `3`: manifest, schema, identity, collection, or harness contract failed.

Required rows prohibit `SKIP` and `QUARANTINED`. Zero selected tests, unknown tests or markers, dirty or mismatched strict identity, missing reports, network-isolation failure, resource/output breach, and CPU fallback are failures. The row-wide limits are versioned in `ci/matrix/host-v1.json`.

The registry-driven local entry point runs any selected subset without copying suite IDs. It refuses
GPU-, model-, or network-dependent suites. Running all three rows also creates the local aggregate:

```bash
python3 ci/tools/run_local_verification.py --list
python3 ci/tools/run_local_verification.py \
  --rows h0,h1,h2 \
  --output-root .local-artifacts/ci/local-draft \
  --allow-dirty-local
```

Use `--strict` only from a clean checkout. The wrapper binds reviewed, tested, and workflow identity
to `HEAD`; a dirty run remains `local-development, immutable=false`.

The tracked Phase 5 llama.cpp contract uses the pinned source-lock metadata and fixtures, so H0 does
not require the ignored `reference/llama.cpp` checkout. A local source audit remains fail-closed and
explicit:

```bash
python3 ci/tools/run_llama_phase5.py --verify-reference
```

## G0 trusted-local preflight

G0 is not part of CPU host CI and never allocates, copies, dispatches, or runs a kernel. Its private native observer links against the pinned HIP runtime and calls identity APIs only. The runner resolves the canonical device from AMD-SMI's exact BDF, uses that HIP id only as a visibility routing hint, and then requires the observer to see exactly one device with the versioned BDF, UUID, target, product, runtime version, and resolved HIP/ROCr libraries. It also records read-only AMD-SMI/sysfs health and process facts before and after observation. The contract validates `/opt/rocm` 7.14.0, the staged H3 artifact and sidecar hashes, the immutable metadata's declared artifact path versus the staged artifact path, reliable ordered pre/post health/process facts, and zero allocation/copy/kernel/dispatch counts. The temporary observer binary is removed before the report is finalized; only its source/build provenance and hash remain in evidence.

For a trusted local run, use one clean checkout and the same 40-character SHA for all rows. The two rows share a nonblocking host lock and must be run serially; visibility variables are routing hints only. Each row writes `report.json` and `report.json.sha256` below `/tmp/sllm-g0-*`, outside the source tree:

```bash
candidate_sha=$(git rev-parse HEAD)
python3 ci/tools/run_g0_preflight.py \
  --row g0-gfx1030 --trusted-local \
  --output-dir /tmp/sllm-g0-run/g0-gfx1030 \
  --artifact-metadata /tmp/sllm-h3-run/h3-gfx1030/hip-artifact-metadata.json \
  --run-id trusted-g0 --run-attempt 1 \
  --reviewed-sha "$candidate_sha" --tested-sha "$candidate_sha" --workflow-sha "$candidate_sha"
```

Repeat the command for `g0-gfx1201` with that row's exact staged H3 metadata. Run the rows serially and aggregate only the two schema-valid `PASS` reports from the same clean immutable candidate:

```bash
python3 ci/tools/aggregate_g0_results.py \
  --needs-json /tmp/sllm-g0-run/needs.json \
  --artifact-dir /tmp/sllm-g0-run \
  --output-dir /tmp/sllm-g0-aggregate \
  --run-id trusted-g0 --run-attempt 1 \
  --expected-reviewed-sha "$candidate_sha" \
  --expected-tested-sha "$candidate_sha" \
  --expected-workflow-sha "$candidate_sha" \
  --expected-tree-oid "$(git rev-parse HEAD^{tree})"
```

The scripts and host-negative tests do not provide canonical G0 evidence by themselves; that requires successful runs on both canonical GPUs and a successful aggregate bound to the same immutable candidate. G0 must not be described as kernel, allocation, copy, dispatch, numerical, model, compatibility, or performance evidence. Those execution claims begin only at G1.

## Local aggregation

Create `.local-artifacts/ci/needs.json` with the current job conclusions:

```json
{
  "h0": {"result": "success"},
  "h1": {"result": "success"},
  "h2": {"result": "success"}
}
```

Then aggregate the three current reports:

```bash
python3 ci/tools/aggregate_host_results.py \
  --needs-json .local-artifacts/ci/needs.json \
  --artifact-dir .local-artifacts/ci \
  --output-dir .local-artifacts/ci/aggregate \
  --allow-local-development
```

The local path accepts only reports labeled `local-development`; it never upgrades them to immutable evidence. Conversely, `--strict-ci` accepts only `required-ci` reports. The aggregator requires exactly one schema-valid `report.json` and matching `report.json.sha256` per expected row, independently derives each row's ordered command argv, command hash, expected count, and executed command IDs from the current suite registry and host matrix, and rejects self-consistent but unregistered command declarations. It also rejects stale or mismatched identities, missing/duplicate/unknown rows, non-success job conclusions, non-`PASS` states, and content-hash mismatches.

## Direct build checks

The top-level build entry point is Cargo. CMake output produced by `sllm-hip-sys/build.rs` stays below Cargo's `OUT_DIR`.

```bash
cargo +1.97.1 build --workspace --locked --offline
cargo +1.97.1 test --workspace --locked --offline
cargo +1.97.1 clippy --workspace --all-targets --all-features --locked --offline -- -D warnings
cargo +1.85.0 check --workspace --locked --offline
```

## Repository hygiene

H0 checks the tracked candidate tree and the configured base revision. For a local candidate, stage the intended files before using a base commit so new paths are included:

```bash
python3 ci/tools/tracked_tree.py --base HEAD
```

The local command reports untracked and ignored sizes, checkout size, file counts, registered and stale worktrees, branch activity, and ahead/behind state. It never deletes data:

```bash
python3 ci/tools/local_hygiene.py --output .local-artifacts/ci/local-hygiene.json
```

## GitHub Actions

`.github/workflows/host-required.yml` runs H0, H1, and H2 as independent GitHub-hosted CPU jobs with hard timeouts of 15, 15, and 8 minutes. H0 and H1 retain a 10-minute row budget inside that setup-aware job cap; the required workflow as a whole keeps its p95 10-minute target and 15-minute hard ceiling. It checks out `github.sha` without persisted credentials, installs the hash-locked host environment before testing, and invokes every row in strict identity mode. The `host-required` job always runs and is the stable branch-protection check. Official actions are pinned to complete commit SHAs. The workflow does not use a self-hosted or GPU runner and does not run H3.

`.github/workflows/phase7-lifecycle.yml` is a manual control-plane entry point. Its versioned profile
source remains `ci/matrix/phase7-ci-profiles-v1.json`:

- `daily` selects the canonical V620 `gfx1030` and R9700 `gfx1201` short observation profile.
- `weekly` selects the two canonical tuples, H0/H1/H2, and all ten explicit compatibility targets.
- `release` uses the release retention and blocking policy.

Manual profile selection and compatibility compilation run on GitHub-hosted workers. GPU rows require
the trusted labels `self-hosted`, `sllm-semantic-g1`, and `rocm-7.14`; push, pull request, schedule,
and release events cannot enter that path. Local trusted execution is authoritative for daily/weekly
GPU operation. Weekly artifacts are retained for 30 days and release evidence for 90 days. The
performance lane is observational until an independently justified threshold is approved, so a
metric delta alone is not a lifecycle failure. GPU PASS still requires the exact tuple, numerical
result, no fallback, health, and cleanup checks.

Local contract and selection checks do not claim a GPU run:

```bash
python3 ci/tools/phase7_lifecycle.py validate
python3 ci/tools/phase7_lifecycle.py resolve \
  --event workflow_dispatch --requested-profile daily \
  --output /tmp/sllm-phase7-daily-selection.json
python3 -m unittest \
  ci.tests.test_phase7_lifecycle \
  ci.tests.test_phase7_compatibility_compile \
  ci.tests.test_phase7_observation
```

The compile-only runner validates one exact target at a time and deletes its generated ELF after
reporting. A successful compile must not be described as runtime or hardware compatibility. The GPU
observation controller reuses the Phase 5 direct-engine runner and emits only a compact bounded
summary; raw traces and generated binaries are not uploaded.

The canonical public-runtime H3 workflow stays GitHub-hosted and non-required for `gfx1030` and
`gfx1201`. Its final link closure is ordered as `libamdhip64.so`, `libhipblas.so`, then
`libhipblaslt.so`; compile/link/code-object inspection is not GPU execution. The dedicated RMSNorm H3
workflow reuses that link closure but additionally fixes the wave32/wave64 registration and dedicated
artifact schema, so it is retained as a manual regression fixture. `semantic-rmsnorm-g1.yml` is also
manual-only and never runs from a normal push.

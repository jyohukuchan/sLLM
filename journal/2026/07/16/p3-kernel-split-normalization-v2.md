# P3 kernel split normalization v2

## Scope

- Source authority before this change: `eb00cbd83b90d6fd8d519f6662ddea16d5f4438c`.
- The pinned producer remains unchanged at `c8becac66551f216de47d0cd935929afe60b3b96`.
- No GPU command, service operation, or mutation of sealed actual evidence was performed.

## Root cause and correction

The sealed actual-v14 capture contains valid kernel intervals whose raw rocprof row
order is not monotonic. In measured run 2, split CSV lines 82 and 83 have starts
`1402529699792903` and `1402529699782863`; the producer correctly rejects the raw
derived order at line 83.

Capture schema v2 keeps every source trace byte-for-byte unchanged. After marker
identity, marker containment, interval, signed-63-bit numeric, dispatch/correlation
identity, and kernel-family validation, it sorts only each derived kernel split by
numeric `(Start, End, Dispatch_Id, Correlation_Id, original_ordinal)`. HIP API,
memory-copy, and marker row order is not changed. The strict producer order check is
unchanged.

The success artifact now records a recomputable kernel-normalization provenance:
source SHA-256 before/after, raw and per-marker-group adjacent inversion counts,
pre/post order digests, row count, dispatch/correlation ID set and multiset digests,
and duration sum before/after. Capture recomputes the document and compares every
measured split with the expected canonical rows before publishing the artifact.

## Sealed actual-v14 checks

- Kernel rows: 12,263 total; 12 marker groups of 928 rows; 1,127 rows outside markers.
- Raw adjacent numeric-order inversions: 207.
- Per-group inversions: `35,10,8,19,23,16,7,20,17,26,3,15`.
- All 12 groups conserve row count, ID sets/multisets, and duration sum.
- The raw run-2 split still fails the pinned producer at CSV line 83.
- Canonical measured splits 2 through 11 pass the unchanged pinned producer.
- Kernel, HIP API, memory-copy, and marker source hashes are unchanged.

## Verification

- Focused normalization/boundary tests: 4 passed.
- Capture test module excluding two environment-dependent launcher lock tests:
  81 passed, 1 skipped, 2 deselected.
- The full module result was 81 passed, 1 skipped, 2 failed only because
  `/run/ullm/r9700.lock` is absent before the two launcher fixture tests; no lock or
  service state was created for this source-only task.

## Maintenance validator handoff

`tools/run-aq4-p2-resident-smoke-maintenance.py` must move
`PROFILE_CAPTURE_SCHEMA` from v1 to v2, add top-level `kernel_normalization` to its
exact key set, and independently reconstruct the v1 normalization sidecar from the
raw kernel and marker refs. It must compare the exact 12-group provenance, validate
the 10 measured split CSVs against the canonical rows, and require equal source
kernel SHA-256 before/after.

## Final-state-independent launcher fixtures

The three capture-module launcher boundary tests now create a private lock under
their pytest temporary directory and inject it as `LAUNCHER.LOCK_PATH` with
`monkeypatch`. Pytest restores the module constant after each test. The tests no
longer depend on whether `/run/ullm/r9700.lock` exists, while the mocked gate and
profile-executor boundaries continue to prevent service and GPU operations.

The canonical capture module now reports `83 passed, 1 skipped` in 13.37 seconds.
The single intentional skip requires `ULLM_TEST_AQ4_P2_RESIDENT_DRIVER` to point to
a clean release binary; it is unrelated to the kernel normalization or launcher
fixture boundary. Capture source commit `418e507214b2a4c0352ac8867bf9689b81948ca4`
was not modified by this follow-up.

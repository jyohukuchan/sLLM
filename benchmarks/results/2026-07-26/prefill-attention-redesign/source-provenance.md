# Source and executable provenance

## What ran in the full-model prefill window

The final serial GQA numerical and throughput window used a dedicated temporary
build tree rooted at `/tmp/ullm-br-prefill-final`. Its executable hashes are
preserved in [`service/serial-gqa-executable-sha256.txt`](service/serial-gqa-executable-sha256.txt):

| binary | SHA-256 |
| --- | --- |
| `sq8_ck_serving` | `b3e80fab13ca806f27a00139f19876663fa9397188dc112a0ada499416539334` |
| prefill driver | `95599fee9db0b5733b275d4ad167e5ae86560d519b0f5be3cea7029cef6fd05a` |
| `ullm-engine` | `3f3e62d0baff9c37cb99e44997e6043615467f7bb49a253268291f735181fd75` |

`serial-gqa-comparison.json` reports `82a9b5b843a9fdeef2cd6ec692c11baf6599857f`
as the driver's embedded source metadata. That value identifies the copied
driver workspace; it is not a substitute for the runtime HIP source identity.

The temporary runtime tree was based at `1cce350bc1debe1a42f45717f5b9dec7c4c859c6`
with the serial GQA runtime patch present. The complete text of
`cached_prefix_flash2_f32_gqa_grouped_serial_kernel_source()` in that tree and
in this commit has the same SHA-256 when extracted including its wrapper:

```text
fe435c1c649c448181b2bde8c5ab00abd6e6057d34af60128ba2b0fbc335e5b2
```

The launcher/cache implementation in `part_01.inc` was byte-identical between
that final temporary tree and this worktree. `part_00.inc` differs only because
the shared worktree also contains subsequently integrated KV-cache support;
the serial compiler entry itself is the same. Thus the full-model evidence is
for the exact HIPRTC source body and dispatch mapping committed here, not for
the earlier rejected wave32 body.

## Current-worktree compile and decode regression

After moving the serial source into the shared runtime worktree, a release
compile completed without errors:

```text
CARGO_TARGET_DIR=/tmp/ullm-br-main-build \
  cargo build --release -p ullm-engine --example sq8_ck_serving \
  --bin ullm-engine --features rocm-ck-gfx1201
```

Its resulting binary hashes were:

| binary | SHA-256 |
| --- | --- |
| `sq8_ck_serving` | `f792a71891b6645d9ed07d606e7aa7f53da2fa7c681e4b9c9433ce81bb8dc6c8` |
| `ullm-engine` | `591b6dc8542172e65c4c773929b83c727a2405203510fe24caa21c8fe8d2c463` |
| current-worktree prefill/decode driver | `b7936ca7171d5c671a0fc0c8f9862ab5e3a85a0d01cc046a1955ce0c30f7d590` |

The BH decode regression itself used the preceding current-worktree driver
hash `73af355b42e25ee323848e8be61c769693cb0c5019444ed1e39326a1a16b07e4`.
That exact hash, its base HEAD (`c0724b710f745c6d0b3db84fe59b5c3df7febdf4`),
preflight, service window, and 27.411786 tok/s output are retained under
`service/current-head-bh-decode-*` and `raw/current-head-bh-decode-regression/`.
The table above is a subsequent CPU-only rebuild validation, not another GPU
measurement.

No second full-model prefill window was opened merely to re-run an already
byte-identical HIPRTC body after the shared-worktree compile. This avoids
spending another exclusive service restart window. The distinction is recorded
explicitly rather than presenting the decode-only current-worktree run as a
new prefill measurement.

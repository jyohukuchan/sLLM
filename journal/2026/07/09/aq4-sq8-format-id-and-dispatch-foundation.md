# AQ4_0/SQ8_0 format ID and dispatch foundation

Date: 2026-07-09

## 前回の要点

- User direction changed the meaning of the SQ FP8 work:
  - existing AQ4 is now public format `AQ4_0`;
  - FP8 E4M3 is now public format `SQ8_0`;
  - SQ8_0 is adopted as a formal format, not a quality candidate waiting for FP8 acceptance.
- Strict AQ4-vs-FP8 top1/logit/text comparisons should be treated as implementation/regression diagnostics.
- Large files should be split toward roughly 10,000 lines or less.
- Runtime implementation selection should become composable by model architecture, GPU architecture, and concrete GPU name.

## 今回の変更点

Implemented the first low-risk slice in `uLLM-project/`:

- Added `crates/ullm-engine/src/format_id.rs`.
  - Canonical public IDs: `AQ4_0`, `SQ8_0`.
  - Legacy aliases include `aq4`, `aq4-prototype-current-runtime`, `aq4_*`, `sq`, `sq-format-v0.1`, and `sq-fp8*`.
- Added `crates/ullm-engine/src/backend_dispatch.rs`.
  - Minimal registry/scoring model for operation, phase, format, model architecture, GPU architecture, and GPU name.
  - Tests cover arch-level default, concrete GPU override, and format-specific override.
- Updated `crates/ullm-engine/src/sq.rs`.
  - SQ FP8 manifests accept `SQ8_0` and legacy `sq-fp8*` aliases.
  - Optional `candidate.format_id` is validated when present.
- Added Python helper `tools/ullm_format_ids.py`.
- Updated SQ artifact builder.
  - New generated manifests use public `candidate.id = SQ8_0` and `candidate.format_id = SQ8_0`.
  - Legacy `sq-fp8-w8a16-r9700-v0` is preserved as `candidate.implementation_id`.
- Updated SQ runtime row builder.
  - `aq4`/legacy AQ rows are normalized to `AQ4_0`.
  - `sq`/legacy SQ rows are normalized to `SQ8_0`.
  - Legacy input IDs are retained in output metadata.
- Updated external benchmark parser.
  - `model.quantization` is normalized to the public format ID when it is a known alias.
  - stdout-derived `sq_candidate=sq-fp8...` becomes `SQ8_0`, with the old value retained as `sq_candidate_legacy`.
- Added `docs/specs/format-ids-v0.1.md` and updated SQ-related specs/words.
- Added/updated Python and Rust tests.

## 検証

Passed:

```text
cargo fmt --all --check
cargo test -p ullm-engine format_id -- --test-threads=1
cargo test -p ullm-engine backend_dispatch -- --test-threads=1
cargo test -p ullm-engine sq -- --test-threads=1
python3 -m unittest tests.test_ullm_format_ids tests.test_build_sq_fp8_artifact_policy tests.test_sq_candidate_runtime_row
python3 -m unittest tests.test_external_benchmark_batch_parser tests.test_ullm_format_ids tests.test_build_sq_fp8_artifact_policy tests.test_sq_candidate_runtime_row
python3 -m py_compile tools/ullm_format_ids.py tools/build-sq-fp8-w8a16-artifact.py tools/build-sq-candidate-runtime-row.py
python3 -m py_compile tools/ullm_format_ids.py tools/run-external-benchmark.py tools/build-sq-fp8-w8a16-artifact.py tools/build-sq-candidate-runtime-row.py
git diff --check -- ':!README.md'
```

`git diff --check` without exclusions still reports a pre-existing trailing whitespace in
`uLLM-project/README.md`; that file was already dirty before this slice and was not edited here.

## 次の行動

1. Split `crates/ullm-engine/src/main.rs` first.
   - The safest first slice is CLI dispatch and command argument routing.
   - Avoid moving smoke-local algorithm bodies in the same step.
2. Split `crates/ullm-runtime-sys/src/lib.rs` by API family after the public wrapper surface is stable.
3. Split `runtime/src/ullm_runtime.cpp` after introducing C++ implementation descriptors.
   - Keep `ullm_runtime.cpp` as a translation unit initially and move families into `.inc` files if needed.
   - Later move to multiple `.cpp` files once anonymous namespace/helper ownership is clear.
4. Connect the Rust `backend_dispatch` selector to one real runtime path, likely cached-prefix attention executor resolution, before spreading it through all kernels.

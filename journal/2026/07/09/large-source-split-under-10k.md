# Large source split under 10k

2026-07-09

- Goal: split huge source files so ordinary project source files stay under roughly 10,000 lines, excluding generated/build/vendor areas such as `build/`, `target/`, `.git`, and `reference-src/`.
- Split `uLLM-project/crates/ullm-engine/src/main.rs` into a small include entrypoint plus `crates/ullm-engine/src/main_parts/part_00.rs` through `part_06.rs`.
- Split `uLLM-project/crates/ullm-runtime-sys/src/lib.rs` into a small include entrypoint plus `crates/ullm-runtime-sys/src/lib_parts/part_00.rs` and `part_01.rs`.
- Split `uLLM-project/runtime/src/ullm_runtime.cpp` into a small include entrypoint plus `runtime/src/ullm_runtime_parts/part_00.inc` and `part_01.inc`.
- Adjusted nested C++ include paths inside the moved runtime chunks to use `../...`, because quote include lookup is relative to the including chunk file.
- Added the runtime chunk files to `crates/ullm-runtime-sys/build.rs` `rerun-if-changed` tracking.
- Current largest project source file after pruning `.git`, `target`, `reference-src`, and `build` is `crates/ullm-runtime-sys/src/lib_parts/part_01.rs` at 9,237 lines.

Verification:

- `cargo fmt --all --check`
- `cargo check -p ullm-engine`
- `git diff --check -- ':!README.md'`
- `python3 -m unittest tests.test_external_benchmark_batch_parser tests.test_ullm_format_ids tests.test_build_sq_fp8_artifact_policy tests.test_sq_candidate_runtime_row`
- `python3 -m py_compile tools/ullm_format_ids.py tools/run-external-benchmark.py tools/build-sq-fp8-w8a16-artifact.py tools/build-sq-candidate-runtime-row.py`
- `cargo test -p ullm-runtime-sys -- --test-threads=1`
- `cargo test -p ullm-engine format_id -- --test-threads=1`
- `cargo test -p ullm-engine backend_dispatch -- --test-threads=1`
- `cargo test -p ullm-engine sq -- --test-threads=1`

Notes:

- `README.md` was already dirty and still has its pre-existing trailing whitespace issue; this work did not clean or otherwise rely on that file.
- `ullm-runtime-sys` still emits the pre-existing anonymous namespace subobject linkage warnings during C++ compilation.

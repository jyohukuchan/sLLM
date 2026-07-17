# T2 SQ FP8 direct matvec batch

## 前回の要点

- SQ FP8 overlayはF32 materialized fallbackから、payload/scale resident bufferを読むdirect dequant matvecへ移った。
- ただし `SqFp8` storageは単発 `matvec` だけdirect pathで、`matvec_batch` はAQ4専用だった。
- SQ候補をprefill batch評価へ流すには、AQ4と同じbatch matvec API境界でSQ FP8を実行できる必要があった。

## 今回の変更点

- runtimeへ `ullm_runtime_sq_fp8_matvec_batch_f32` を追加した。
- HIPRTC sourceへ `ullm_sq_fp8_matvec_batch_f32_kernel` を追加し、`grid.x=row`、`grid.y=batch` でbatch-major input/outputを処理するようにした。
- Rust FFIへ `sq_fp8_matvec_batch_f32` を追加した。
- `PackageAq4ResidentMatvec::matvec_batch` は、`SqFp8` storageの場合にSQ FP8 batch direct kernelへdispatchする。
- CPU unit testと、`ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_BATCH_KERNEL=1` 付きHIP unit testを追加した。

## 検証

- `cargo test -p ullm-runtime-sys cpu_sq_fp8_matvec_batch_f32_computes_expected_row_block_values -- --test-threads=1`
- `cargo test -p ullm-runtime-sys first_hip_sq_fp8_matvec_batch_f32_computes_expected_values_when_available -- --test-threads=1`
- `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_BATCH_KERNEL=1 cargo test -p ullm-runtime-sys first_hip_sq_fp8_matvec_batch_f32_computes_expected_values_when_available -- --test-threads=1`
- `cargo test -p ullm-runtime-sys cpu_sq_fp8_matvec_f32_computes_expected_row_block_values -- --test-threads=1`
- `cargo check -p ullm-runtime-sys`
- `cargo check -p ullm-engine`
- `cargo fmt --all --check`
- `git diff --check -- ':!README.md'`

## 次の行動

1. `SqFp8` storageをpair/triple matvecへ広げ、self-attention Q/K/V projectionで単発kernel連打にならないようにする。
2. SQ FP8 batch matvecを使うcomponentまたはpackage-level prefill rowを追加し、AQ4 batch matvecとの比較行を保存する。
3. MLP fused境界は、qualityが通るSQ tensorが増えてから優先度を上げる。

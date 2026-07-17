# T2 SQ FP8 direct matvec pair/triple

## 前回の要点

- SQ FP8 direct pathは単発 `matvec` と `matvec_batch` まで進んだ。
- self-attention Q/K/V projectionでは、SQ tensorが複数ある場合でも `matvec_pair_with` / `matvec_triple_with` が単発matvecの連打へ落ちる状態だった。
- AQ4 fused pathと同じ呼び出し粒度に近づけるには、pair/triple projection境界をSQ FP8 direct pathへ広げる必要があった。

## 今回の変更点

- runtimeへ `ullm_runtime_sq_fp8_matvec_pair_f32` と `ullm_runtime_sq_fp8_matvec_triple_f32` を追加した。
- HIPRTC sourceへ `ullm_sq_fp8_matvec_pair_f32_kernel` と `ullm_sq_fp8_matvec_triple_f32_kernel` を追加した。
- Rust FFIへ `sq_fp8_matvec_pair_f32` と `sq_fp8_matvec_triple_f32` を追加した。
- `PackageAq4ResidentMatvec::matvec_pair_with` / `matvec_triple_with` は、対象matrixがすべて `SqFp8` storageの場合にSQ FP8 pair/triple direct kernelへdispatchする。
- mixed AQ4/SQ/F32の場合は従来どおり個別matvec fallbackを使う。

## 検証

- `cargo test -p ullm-runtime-sys cpu_sq_fp8_matvec_pair_f32_computes_expected_mixed_scale_values -- --test-threads=1`
- `cargo test -p ullm-runtime-sys cpu_sq_fp8_matvec_triple_f32_computes_expected_mixed_scale_values -- --test-threads=1`
- `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_PAIR_KERNEL=1 cargo test -p ullm-runtime-sys first_hip_sq_fp8_matvec_pair_f32_computes_expected_values_when_available -- --test-threads=1`
- `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_TRIPLE_KERNEL=1 cargo test -p ullm-runtime-sys first_hip_sq_fp8_matvec_triple_f32_computes_expected_values_when_available -- --test-threads=1`
- `cargo check -p ullm-runtime-sys`
- `cargo check -p ullm-engine`
- `cargo build -p ullm-engine`
- `cargo fmt --all --check`
- `git diff --check -- ':!README.md'`

## 判断

- SQ FP8は単発、batch、pair、triple matvec API境界でF32 materializeなしに実行できるようになった。
- 現在のfull mixed strict-top1保守候補はlayer3 `k_proj` 1 tensorだけなので、実ベンチ上の改善はまだ限定的である。

## 次の行動

1. `q/k/v` または `q/k` が同時にstrict-top1を維持できる小さいSQ候補を探し、pair/triple境界を踏むfull mixed rowを保存する。
2. `up_proj` 系はrow-block幅、scale粒度、W8A8/activation scaleを変えて再探索する。
3. MLP fused境界は、qualityが通る `gate/up` 候補が出た段階でSQ FP8 direct fused pathへ進める。

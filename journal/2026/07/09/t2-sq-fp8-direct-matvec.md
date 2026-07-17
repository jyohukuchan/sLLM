# T2 SQ FP8 direct matvec

## 前回の要点

- `k-layer3-rb16` はfull mixed B=1/4/8でAQ4 final top1と一致したが、実行は `materialized_f32_fallback` だった。

## 今回の変更点

- runtimeに `ullm_runtime_sq_fp8_matvec_f32` とHIPRTC kernel `ullm_sq_fp8_matvec_f32_kernel` を追加した。
- Rust FFIに `sq_fp8_matvec_f32` とscale kind定数を追加した。
- `PackageAq4ResidentMatvec` に `SqFp8` storageを追加し、SQ overlay tensorをpayload/scale resident bufferとして保持するようにした。
- `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_KERNEL=1` でB=1/4/8を通し、`sq_execution_mode=direct_fp8_dequant_matvec` とtop1一致を確認した。

## 次の行動

1. direct pathはまだ単体SGEMVなので、SQ pair/triple/fused matvecへ広げる。
2. `up_proj` 系はscale粒度やrow-block幅を変えてquality boundaryを再探索する。
3. 通るcandidateが増えたら長いprefill/prefix gridへ流す。

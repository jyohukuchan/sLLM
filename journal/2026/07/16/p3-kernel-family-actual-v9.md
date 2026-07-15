# P3 actual-v9 kernel family mapping

## Scope

- `profile-aq4-p2-family-exclusive.py` の保守的な排他分類だけを更新した。
- producer、capture、launcher、maintenance、actual artifact は変更していない。

## Source-grounded mapping

- `__amd_rocclr_fillBufferAligned` と `__amd_rocclr_copyBuffer` は ROCclr の補助kernelなので、明示的な `runtime_support` とした。
- `ullm_bf16_row_f32_kernel` は resident embedding rowをBF16からF32へ展開する実装なので、明示的な `embedding` とした。
- standalone `ullm_add_f32_kernel` は residual addなので、既存の `normalization` とした。
- `ullm_aq4_matvec_silu_mul_f32_kernel` と `ullm_aq4_matvec_add_f32_kernel` はAQ4 projectionへ融合されたkernelなので、normalization patternから排除して `aq4_projection` 専属とした。
- catch-allやunknown rowの無視は追加していない。未知名は引き続き `None`、複数family一致は引き続き `ProfileError` になる。

## Actual-v9 regression

- source trace: `aq4-p3-diagnostic_kernel_trace.csv`
- rows: 12,263
- unknown: 0
- multiple-family matches: 0
- classified rows:
  - runtime_support: 4,071
  - embedding: 1,537
  - paged_validation: 197
  - aq4_projection: 2,986
  - attention: 104
  - recurrent: 1,158
  - normalization: 2,209
  - head: 1
- mapping SHA-256: `d5a159dff6776fc1229d1bacf415715154fb3bb2e3d3051f59bc3dca2ec03b29`

## Verification

- `python3 -m pytest -q tests/test_profile_aq4_p2_family_exclusive.py`: 39 passed
- `python3 -m py_compile tools/profile-aq4-p2-family-exclusive.py`: passed
- `git diff --check`: passed
- producer test collectionは既存のprofiler SHA pinが変更後helperを拒否して停止した。これはfail-closedの期待動作であり、統合レーンが新しいhelper commit/blob/SHAへproducerとcaptureのpinを更新してから回帰する必要がある。

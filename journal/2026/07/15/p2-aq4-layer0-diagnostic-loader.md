# AQ4 layer-0 diagnostic single-matvec loader

## 前回の要点

`ullm-aq4-layer0-qkv-runtime-probe` は単独の `aq4_matvec_f32` を実行する
診断バイナリだが、共通の `PackageAq4ResidentMatvec::load` が production
`Aq4MatvecBatch` の全 phase plan を eager resolve するため、HIP device 1 で
single-matvec guard だけを設定した場合も ColdPrefill admission で停止していた。

## 今回の変更点

- `PackageAq4ResidentMatvec::load_single_diagnostic` を追加した。
- package payload、materialize、storage、shape、identity 検証は既存 loader と共有し、
  診断経路だけは `aq4_batch_plans=None` として `Aq4MatvecBatch` admission を行わない。
- 既存 `load` と `load_with_sq_overlay` は production batch plan を従来どおり解決する。
- HIP の診断 loader も `ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL=1` が無ければ拒否し、
  single `matvec` は fallback 無しのままとした。CPU device 0 の経路は変えていない。
- batch API を診断 instance から呼ぶと production plan 不在で fail-closed する unit test を追加した。

## 検証

- `CARGO_BUILD_JOBS=1 cargo test -p ullm-engine aq4_package_runtime::tests --lib`
  — 2 passed（既存 production plan test を含む）。
- `CARGO_BUILD_JOBS=1 cargo check -p ullm-engine --bin ullm-aq4-layer0-qkv-runtime-probe`
  — 成功。runtime C++ の既存 subobject-linkage warning のみ。
- 変更対象の rustfmt check — 成功。
- CPU device 0 で正式 v7 input sidecar を再実行。`output.f32le` SHA-256 は
  `9683b8c5decd545c35e416da0b0f9568e6f51463ae5395fcd872dc9cbd82b473` で、
  既存 formal runtime-order output と byte-for-byte 一致した。report は
  `status=valid`、`classification=unclassified`、`promotion_eligible=false`、
  `fused=false`、3 rows finite を確認した。

## 残課題

GPU、常駐サービス、holdout、production batch の実行は行っていない。HIP の実行証拠と
数値閾値による昇格判断は別の gated run で扱う。

## 次の行動

親エージェントへ限定 commit を渡し、既存 production registry/default と分離したまま統合する。

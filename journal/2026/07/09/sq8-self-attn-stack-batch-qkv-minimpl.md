# SQ8_0: mixed request-state self-attn stack のqkv batched path 最小実装

## 変更点
- `crates/ullm-engine/src/main_parts/part_06.rs`
  - `PackageSelfAttnResidentStepLayer` を、RMSNorm 前半、q/k/v projection、中間後半へ分割されたまま保持。
  - `PackageSelfAttnResidentStepBatchLayer` に `step_batch_from_device_to_device` を追加。
  - 既存の `step_batch_from_host_to_device` と対になって、各 request を入力RMSNorm後の一時バッファに一括 pack → `matvec_batch` で q/k/v を一括実行 → 各 request へ scatter → `run_device_step_after_qkv_projection` を継承実行する流れを追加。
  - `PackageMixedRequestStateLayer` に `step_batch_from_device_to_device` を追加。

- `crates/ullm-engine/src/main_parts/part_02.rs`
  - `package_mixed_request_state_layers_batch_step` で、同一 layer/timestep の複数 request かつ self-attn の場合だけ batch helper を使う分岐を追加（host→device, device→device）。
  - self-attn でなければ従来どおり request ごと逐次。

## 検証
- `cargo check -p ullm-engine`

## 残課題
- 実機での `sq-fp8-package-self-attn-stack-batch-smoke` 実行による `sq_fp8_batch_matvec_count > 0` の確認は未実施（ユーザー側で実施予定）。

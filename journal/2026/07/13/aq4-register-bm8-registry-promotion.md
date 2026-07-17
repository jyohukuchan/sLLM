# AQ4 register BM8 typed registry promotion

## 前回の要点

AQ4 batch は単一のLegacy descriptorと環境依存の旧batch ABIを使っていた。runtime側ではforced `gfx1201/group16` register BM8 ABIを追加中だった。

## 今回の変更点

- `HipAq4RegisterBm8` featureと`ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_KERNEL` guardを追加した。
- M128 (`rows=32`, `cols=128`, `group_size=16`, `batch=128`) のforced ABI scratch probeを、feature/cache publish前の同期付きprobeへ追加した。
- 適格なHIP gfx1201/group16/rows%32/cols%128形状では、幅2..7をLegacy、幅8..128を`HipAq4GemmRegisterBm8F32` Primaryへ分離した。BM8 descriptorのworkspaceはゼロとした。非適格形状は全幅Legacyとした。
- `StartedOperationPlan`のBM8 wrapperと、`PackageAq4ResidentMatvec`のplan-selected executable dispatchを追加した。start後のdirect ABI bypass/fallbackはない。
- worker canonical guard、deployment profile、AQ4 manifest fixture、生成テスト期待値を更新した。

## 検証

- `cargo fmt --all`
- `cargo check -p ullm-engine`
- `cargo test -p ullm-engine backend_operation_registry -- --test-threads=1`（33 passed, 1 ignored）
- `cargo test -p ullm-engine aq4_package_runtime -- --test-threads=1`（対象テストなし、ビルド成功）
- `cargo test -p ullm-engine --bin ullm-aq4-worker -- --test-threads=1`（11 passed）
- `cargo test -p ullm-runtime-sys cpu_aq4_register_bm8_batch_rejects_without_fallback_or_output_mutation -- --test-threads=1`（passed）
- `cargo test -p ullm-engine aq4_batch_hot_lookup_indexes_pre_resolved_phase_cache_only -- --test-threads=1`（passed）
- `cargo test -p ullm-engine cpu_started_bm8_plan_rejects_before_abi_and_preserves_output -- --test-threads=1`（passed）
- `cargo check --workspace`
- `git diff --check -- ':!README.md'`

## 次の行動

親エージェントがruntime/runtime-sysのforced ABI着地を統合し、実HIP BM8 differentialとfull workspace検証を行う。

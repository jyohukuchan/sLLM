# P2 AQ4 layer-0 QKV/Z/gate/beta fused diagnostic

## 前回の要点

production layer、serving path、既存probe、既定値を変更せず、layer 0 の exact AQ4 QKV/Z/A/B と `A_log`/`dt_bias` を再検証し、fused QKV/Z/gate/beta の finite な f32 sidecar と identity-bound report を出力する診断probeを追加した。CPU device 0 では fused QKV と standalone `matvec` の bit/byte exact を必須にしていた。

## 今回の変更点

- report schemaを `ullm.aq4_layer0_qkv_z_gate_beta_runtime_probe.v2` に更新した。
- standalone QKVの `qkv-standalone.f32le` をfused QKVと同じ入力順・shape `[8192]` でatomic publishし、reportの `outputs.qkv_standalone` と `qkv_component_reference.standalone_output_key`、SHA、行情報に記録する。
- CPUは `reference_backend=cpu` / `reference_kind=formal_standalone_reference` としてbit/byte exactを要求する。HIPは `reference_backend=hip` / `reference_kind=diagnostic_standalone_reference` として有限性・形状・identityをfail-closedで検証し、差分は閾値なしの診断値として記録する。CPU比較失敗時のエラー文はCPU formal referenceに限定した。
- reportのQKV referenceに `operation`、`standalone_rpb_raw`、`standalone_rpb_effective`、`standalone_rpb_source` を追加した。CPUのeffective RPBは `null`/`not_applicable_cpu`、HIP gfx1201は有効な `ULLM_AQ4_MATVEC_RPB` または arch default 32 を記録する。fused RPBは既存の dedicated-valid優先、invalid/unset時のみgeneric fallback、effective 4要求を維持した。
- `AtomicFile::publish` はhard-link前後の所有 `FileStat`を返し、登録側のpost-publish orphan windowを除去した。publication失敗時のcleanupは返却identity一致時だけ行い、登録競合でforeign replacementを削除しない負例テストを追加した。sidecar publish後、report commit前にもidentityを再検証する。

## 検証

実行した検証:

- `rustfmt crates/ullm-engine/src/bin/ullm-aq4-layer0-qkv-z-gate-beta-runtime-probe.rs` — 成功。
- `cargo test --bin ullm-aq4-layer0-qkv-z-gate-beta-runtime-probe` — 15/15 tests passed。CPU/HIP比較契約、standalone backend/RPB記録、AtomicFile identity、registration race、foreign replacement、RPB precedence、manifest exact-name、output symlinkを含む。
- `cargo check --bin ullm-aq4-layer0-qkv-z-gate-beta-runtime-probe` — 成功。既存runtime C++のsubobject-linkage warningのみ。
- CPU device 0実行（GPU/service/holdoutなし） — `status=valid`, `schema=v2`, `classification=unclassified`, `promotion_eligible=false`, `fused=true`, `fallback_allowed=false`。出力keysは `beta`, `gate`, `qkv`, `qkv_standalone`, `z`。QKV formal standalone referenceは `bit_exact=true`, `bit_mismatch_count=0`, `standalone_rpb_effective=null`。
  - `qkv.f32le` / `qkv-standalone.f32le`: `9683b8c5decd545c35e416da0b0f9568e6f51463ae5395fcd872dc9cbd82b473`
  - `z.f32le`: `7ed98f1c7f8988958377b548f44afe3a2ddc5180150d1e3191c7d0e2a408b286`
  - `gate.f32le`: `dbf470352abb0bbe31e23018d5770608a424048f83191f3e063360f6ba857857`
  - `beta.f32le`: `ed4a3a57629fddf561f4f115f5b598a59a10984e579d5a8bff23dbaf0478bf64`
- fused guard未設定 — status 1、出力file 0、CPUで即時拒否。
- 既存出力への再実行 — status 1、`refusing to overwrite an existing output sidecar`。既存6 fileは不変。
- HIP device 1 guard境界（GPU実行なし） — status 1、`HIP_VISIBLE_DEVICES must be exactly 1 for HIP device 1`、出力file 0。

## 次の行動

このbranchの限定commitを親branchへcherry-pickする。GPU実行、service実行、holdout、数値品質閾値判定、promotion判定は未実施であり、HIP referenceは診断契約のまま扱う。

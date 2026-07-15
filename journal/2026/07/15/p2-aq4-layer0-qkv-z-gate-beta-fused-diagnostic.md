# P2 AQ4 layer-0 QKV/Z/gate/beta fused diagnostic

## 前回の要点

既存の単独QKV診断probeと`PackageAq4ResidentMatvec::load_single_diagnostic`を、productionのAQ4 batch-plan admissionから分離したまま再利用する。

## 今回の変更点

- 新規の`ullm-aq4-layer0-qkv-z-gate-beta-runtime-probe`だけを追加した。production layer、serving path、既存probe、既定値は変更していない。
- layer 0のQKV/Z/A/Bをexact tensor nameで選択し、BF16、shape、family、candidate、group、encoding、element/group数、manifest SHA、index/scale/codebook SHA、連結payload SHAを検証・再検証する。
- `A_log`と`dt_bias`をexact passthrough name/shape/dtypeで読み、manifest payload SHAと実ファイルSHAを検証する。
- CPU device 0に固定し、`ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL=1`を必須化した。fused callのエラー時fallbackは許可しない。
- fused QKV/Z/gate/betaを4つのatomic little-endian f32 sidecarにrow streamingで書き、全出力のfinite性を検証する。QKV fused出力は同一inputに対するstandalone `matvec`とbit/byte exactでなければ失敗する。

## 検証

実行した検証:

- `cargo check --bin ullm-aq4-layer0-qkv-z-gate-beta-runtime-probe` — 成功。既存runtime C++のsubobject-linkage warningのみ。
- `cargo test --bin ullm-aq4-layer0-qkv-z-gate-beta-runtime-probe` — 5 tests passed。
- guard未設定の実行はstatus 1で即時拒否し、出力ファイルを作成しなかった。既存出力ディレクトリへの再実行もstatus 1でno-overwrite拒否した。
- CPU device 0実行（GPU/service/holdoutなし） — `status=valid`, `fused=true`, `fallback_allowed=false`。
  - QKV sidecar: 3 rows, shape `[8192]`, SHA-256 `9683b8c5decd545c35e416da0b0f9568e6f51463ae5395fcd872dc9cbd82b473`。
  - Z sidecar: 3 rows, shape `[4096]`, SHA-256 `7ed98f1c7f8988958377b548f44afe3a2ddc5180150d1e3191c7d0e2a408b286`。
  - gate sidecar: 3 rows, shape `[32]`, SHA-256 `dbf470352abb0bbe31e23018d5770608a424048f83191f3e063360f6ba857857`。
  - beta sidecar: 3 rows, shape `[32]`, SHA-256 `ed4a3a57629fddf561f4f115f5b598a59a10984e579d5a8bff23dbaf0478bf64`。
  - fused QKV対standalone CPU: bit mismatch `0`, byte mismatch `0`, max abs `0`, relative L2 `0`。

未実施:

- GPU/service実行、holdout、数値品質閾値判定、promotion判定。

## 次の行動

このbranchの限定commitを親branchへcherry-pickし、oracle仕様確定後にgate/beta/zの解釈や追加referenceを必要に応じて更新する。

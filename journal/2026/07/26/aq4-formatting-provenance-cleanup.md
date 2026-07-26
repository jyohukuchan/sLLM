# AQ4 未コミット Rust 差分の素性確定と作業ツリー清浄化

## 前回の要点

- 2026-07-26 の並行作業後、AQ4 runtime 関連の6ファイルに未コミット差分が残っていた。
  差分量は 1,098 additions / 341 deletions で、主に引数の1行1項目化と import 順序の変更に見えた。
- `RuntimeFeature::HipAq4GemmWmmaGroup8RaggedM` と
  `ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_GROUP8_RAGGED_M_KERNEL` は、調査対象差分の直前の
  HEAD にも既に存在していた。今回の差分による機能追加ではない。
- workspace 全体は既に rustfmt の基準に揃っておらず、`cargo fmt --check` はコミット済みの
  複数ファイルにも差分を報告していた。

## 今回の変更点

- 各対象ファイルについて、差分直前の HEAD と作業版の両方を Rust 2024 / rustfmt 1.9.0-stable の
  同一条件で stdin から `rustfmt --emit stdout` に通し、出力の SHA-256 を比較した。
  repository 内に `rustfmt.toml` はなく、両方とも同じ既定設定を用いた。

  | 対象 | 正規化出力 SHA-256 |
  | --- | --- |
  | `crates/ullm-engine/src/aq4_package_runtime.rs` | `a29a31bf6d83a76668b46fd398b2bf586d698825b455db81ba880d705a047fa9` |
  | `crates/ullm-engine/src/aq4_worker_backend.rs` | `f209322763879107c5bfcd4f0808a1977a4748ac91a906fd4e0ced48be3858d1` |
  | `crates/ullm-engine/src/backend_operation_registry.rs` | `9be021d605def471be71cd917bce777bdc0a701677e4234f22d8303960d42deb` |
  | `crates/ullm-engine/src/loader.rs` | `79c8c229730fbfa0bdfddca65216ccf62cdf723b61c3b930dc88ca3ed8ffa5cf` |
  | `crates/ullm-engine/src/qwen35_aq4_layer_runtime.rs` | `f5948b3da1780ceaf0b4bd2350ceee93161cedd12f51e0ae277b08485bd5b3fd` |
  | `crates/ullm-runtime-sys/src/lib_parts/part_00.rs` | `8f7498d9369e736fe206e7661dd43cc84df6ad2dbf8f139c57de07033f467e89` |

- 6件すべてで正規化出力が一致した。したがって、未コミット差分には実質変更はなく、全件を
  整形のみと確定した。`0a2a67d0`（`style(aq4): normalize runtime source formatting`）に
  6ファイルだけを明示指定してコミットした。既存の staged evidence は含めていない。
- `part_00.rs` は直接の `rustfmt --check` で 4192行目の既存未整形箇所を報告する。
  ただし差分前 HEAD と `0a2a67d0` の正規化 SHA-256 は同一であり、この箇所は今回のコミット前から
  存在していた。範囲外の整形は追加していない。
- GPU を不可視化する `HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1` で以下を実行した。
  - `cargo check --locked -p ullm-runtime-sys -p ullm-engine`: 成功。
  - `cargo test --locked -p ullm-runtime-sys --lib -- --test-threads=1`: 173 passed, 40 ignored。
  - `cargo test --locked -p ullm-engine --lib -- --test-threads=1`: 780 passed, 5 ignored。
- `cargo fmt --all -- --check` は失敗を継続した。`sq8_ck_serving.rs`、複数の AQ4 bin、`lib.rs`、
  `sq8_layer_runtime.rs`、`sq8_model_head_runtime.rs`、および
  `ullm-runtime-sys/examples/sq8_0_paged_decode_split_bench.rs` など、今回の6ファイル以外の
  追跡済みソースが対象である。全体整形は行っていない。

## 次の行動

1. repository 全体の rustfmt 統一は、現在の並行作業を切り分けたうえで、専用の大規模な
   formatting-only change としてレビュー・承認を得てから実施する。`cargo fmt --all` は独断で実行しない。
2. 将来の release build / provenance を作る前に、対象 branch の staged・unstaged・untracked 状態を
   あらためて確認し、build receipt の `worktree_clean` / `untracked_clean` と矛盾しない状態にする。
3. 今回扱わなかった benchmark evidence、既存 staged journal、その他の未追跡項目は所有者の作業として
   そのまま残した。GPU、systemd、`/etc/ullm/served-models/active.json`、`/opt/ullm`、
   activation/campaign には触れていない。

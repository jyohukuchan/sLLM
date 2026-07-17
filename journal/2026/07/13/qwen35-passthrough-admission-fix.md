# Qwen3.5 AQ4 passthrough admission fix

## 前回の要点

`qwen35_retained_activation_bytes` の self-attention geometry admission が、AQ4 行列一覧にない q/k norm を量子化 tensor metadata から探していました。実 package では q/k norm は passthrough tensor です。

## 今回の変更点

- `qwen35_aq4_model_runtime.rs` に exact passthrough metadata lookup helper を追加しました。
- `TensorSelector::Name` 選択結果の exact 名、一意性、rank-1 の非zero shape、shape product と elements の一致を payload 読み込みなしで検証します。
- q_norm/k_norm の head_dim を admission に使用し、両者の shape 不一致を fail-close にしました。
- self-only fixture で passthrough q/k norm の成功と workspace admission、missing/ambiguous/wrong shape/elements の fail-close を追加しました。
- 同ファイルと関連 Qwen3.5 admission の `rows()` passthrough 誤用を確認し、該当箇所はこの修正で解消しました。

## 検証

- `cargo test -p ullm-engine qwen35_aq4_model_runtime -- --test-threads=1`: 11 passed
- `cargo check -p ullm-engine`: passed（既存 C++ subobject-linkage warning のみ）
- `cargo fmt --all` / `cargo fmt --all -- --check`: passed
- `git diff --check -- crates/ullm-engine/src/qwen35_aq4_model_runtime.rs`: passed

コミットは作成していません。

# Causal GQA attention core contract

## 前回の要点

`GraphNodeKind` の RoPE 契約に続き、Grouped Query Attention（GQA）の形状・能力・実行契約をエンジンへ追加する作業を進めた。

## 今回の変更点

- `model_graph.rs` に `CausalGqaAttentionCore` を追加し、Q/K/V/context の幅、GQA ヘッド比、因果 softmax、state/layout/format の契約を検証するテストを追加した。
- `cpu_reference_executor.rs` に F32 RowMajor/TokensHidden 向けの状態なし CPU 実装を追加した。KV キャッシュ状態、PackedRagged、BF16/FP16 は事前検証で拒否する。
- スコア行列を確保せず、最大値減算を含む三走査 softmax と重み付き V 集約を実装した。value head 次元が query/key head 次元と異なる場合にも対応した。

## 検証

- `cargo test -p ullm-engine model_graph -- --test-threads=1`（24 passed）
- `cargo test -p ullm-engine cpu_reference_executor -- --test-threads=1`（42 passed）
- `cargo check --workspace`
- `cargo doc -p ullm-engine --no-deps`
- `cargo fmt --all -- --check`、`git diff --check`

runtime-sys 由来の既存 C++ `-Wsubobject-linkage` 警告は残るが、Rust の検証は成功した。コミットは親エージェント側でまとめる。

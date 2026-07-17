# AQ4 バッチレジストリと能力プローブ

- `OperationKind::Aq4MatvecBatch` と shape-exact geometry、M2..128 の production descriptor、StartedPlan ABI wrapper を追加した。
- `PackageAq4ResidentMatvec` は AQ4 ロード時に幅別 `ResolvedPhasePlans` を一度だけ構築し、hot path は plan lookup と typed wrapper 呼び出しだけを行う。SQ8 の既存 backend dispatch は維持する。
- AQ4 matvec batch と QKV prepare batch の probe 幅を M=128 に変更し、recurrent sequence 専用 feature/guard と M=128 probe を追加した。features は最後の同期成功後にだけ publish する。
- `cargo check -p ullm-engine` と backend registry tests を実行済み。HIP 実機 probe は未実施。

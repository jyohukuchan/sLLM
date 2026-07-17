# P2-2 paged chunk primitives

- M=2..128 の paged KV write と causal GQA chunk read（通常・sigmoid gate）を runtime、Rust wrapper、typed registry に追加した。
- CPU 差分、境界値、非恒等 block table、無効 table の fail-closed、登録・feature・配置 guard を検証した。
- Qwen adapter への接続は次コミットに分離した。HIP 実機が利用できない環境では HIPRTC 実行確認は未実施。

# Paged decode split typed registry

## 前回の要点

既存の単一 paged decode registry、RuntimeFeature probe、三 phase の
`ResolvedPhasePlans` を維持したまま、split-source ABI を load-time に選択できる境界が必要だった。

## 今回の変更点

- `HipPagedDecodeAttentionSplit = 12`、exact-`1` guard、plain/gated split probe、fault checkpoint 16 を追加した。
- `PagedDecodeSourceTile::{Tokens128,Tokens256}`、split executable、caller-owned workspace の ABI wrapper を追加した。
- モデル名や固定 shape に依存しない single/split paged decode registry を追加し、GQA、head/value 上限、overflow、feature、read-only state effect、split workspace 式を load-time に検証する。
- `PagedDecodeDispatchPlans` で single/split を全 phase 事前解決し、`min_cache_len` 境界で split を選択する。feature 不在・workspace 不足は single-only に戻す。
- 非 Qwen shape、tile 別 workspace、invalid geometry、fallback、threshold、CPU ABI、tile mismatch の unit test を追加した。

## 次の行動

`backend_operation_registry.rs` の変更を親側の runtime primitive と統合する。HIP 実測までは split guard を AQ4 worker の production required list に昇格しない。

## 検証

- `cargo fmt --all --check`: 成功
- `cargo check -p ullm-engine`: 成功
- `cargo test -p ullm-engine backend_operation_registry -- --test-threads=1`: 40 passed, 1 ignored
- `cargo test -p ullm-engine aq4_worker_backend -- --test-threads=1`: 4 passed
- `cargo test -p ullm-engine -- --test-threads=1`: 既存 `sq8_ck_serving_performance` example の独立したコンパイル不整合で失敗。registry tests 自体は上記で通過。
- `git diff --check`: 後続で実施

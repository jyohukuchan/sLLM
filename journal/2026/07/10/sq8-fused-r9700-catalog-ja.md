2026-07-10

- 目的: SQ8_0 の fused projection descriptor catalog を、R9700 専用最適化を想定した形に拡張。
- 変更: `backend_dispatch.rs` に R9700 fused 定数を追加し、`SQ8_0_FUSED_PROJECTION_DESCRIPTOR_CATALOG` に
  `generic / rdna4 / r9700` のエントリを追加。`r9700` は `gpu_arch=Some("RDNA4")`,
  `gpu_name=Some("Radeon_AI_PRO_R9700")`, `priority=20` で登録。
- テスト: catalog 件数検証、fused catalog が active 実行リストに含まれないこと、R9700 catalog エントリが
  `select_backend(..., SQ8_0_FUSED_PROJECTION_DESCRIPTOR_CATALOG)` で選択できることを追加。
- 実行: `cargo test -p ullm-engine backend_dispatch --lib` (22 passed)

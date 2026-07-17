# 汎用 paged chunk runtime

- `ullm_runtime_paged_kv_write_chunk_f32` と `ullm_runtime_paged_causal_gqa_chunk_f32`（ゲート版を含む）を追加した。
- Rust ラッパーで M=1..128、算術、キャッシュ、block table、ゲート形状を事前検証し、CPU 実装は非恒等 block table と境界跨ぎを処理する。
- HIPRTC に M 行の bounds-checked writer/reader kernel を追加し、全 attention は M×context ワークスペースを確保しない。
- HIP path は kernel launch 前に block table を D2H 同期検証し、invalid table では launch せず `INVALID_ARGUMENT` を返す。reader は query-head block 単位の 256-thread online reduction で score 再計算を抑え、head/value dim を 256 以下に制限する。
- `cargo test -p ullm-runtime-sys -- --test-threads=1`（144 passed）を実行した。

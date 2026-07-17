# P3候補A 実装準備

- 作業範囲: `/tmp/ullm-p3-candidate-a` の `p3-candidate-a`（base `b1755da`）。main/service/GPUは変更していない。
- 変更: AQ4 native prefillで、`ULLM_AQ4_PREFILL_DIRECT_SEQUENCE_OUTPUT`（既定OFF）を有効にした場合、linear/self-attnのsequence residualをping/pongへ直接出力する経路を追加した。従来のworkspace→ping/pong全行コピーは省略し、最終行の`layer_output_buffer`保持は維持する。出力サイズ検証とcaller disjoint ownershipの契約を追加した。
- コミット: `8b817fab` (`Prototype guarded AQ4 prefill direct sequence output`)
- 検証: `CARGO_BUILD_JOBS=1 cargo check -p ullm-engine --lib` 成功、`CARGO_BUILD_JOBS=1 cargo test -p ullm-engine --lib -- --test-threads=1` 成功（726 passed, 1 ignored）、対象2ファイルの`rustfmt --check`成功、`git diff --check`成功。
- 計測時の期待値: direct経路はnative chunkごとにdecoder layerあたりsequence全行D2Dコピー1回を削減する（最終row保持D2Dとembedding row scatterは残る）。既定OFF経路は従来挙動と同じで、プロファイルではA/Bの数値一致、D2D bytes/launch、GPU interval/p50を比較し、改善がなければ採用しない。

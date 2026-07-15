# P3 candidate-A CPU oracle / safety

## 前回の要点

候補Aは、`ULLM_AQ4_PREFILL_DIRECT_SEQUENCE_OUTPUT` が有効な場合だけ、シーケンス出力を呼び出し側の ping/pong バッファへ直接書き込む実装だった。CPU の既存テストと静的確認は通っていたが、出力バッファの別名（alias）契約、直接経路の事前失敗時の退避、実行失敗後の状態再利用禁止を独立に検証できていなかった。

## 今回の変更点

- `qwen35_aq4_layer_runtime.rs`
  - direct 出力のサイズ不足、residual alias、共有 sequence workspace alias を実行開始前に拒否する純粋な契約検証を追加した。
  - self-attention / linear-attention の両 direct 経路へ契約検証を接続した。
  - CPU テストで alias と短いバッファを fail-closed に拒否することを確認した。
  - `ResidentRequestState` の crate 内可視性を広げ、モデルの CPU テストから実行失敗→同期 reset→再利用の状態機械を検証できるようにした。
- `qwen35_aq4_model_runtime.rs`
  - CPU 専用の routing seam を追加し、M={1,2,8,16,32,64,128} で copy/direct の出力一致と有限値を確認した。
  - direct の admission 失敗は既存 copy 経路へ退避する一方、実行失敗は状態を poison したまま退避しないよう dispatch を更新した。
  - CPU テストで admission fallback、実行失敗後の再利用禁止と reset、alias/長さ不一致を確認した。
  - direct 経路は従来どおり環境変数の明示的 opt-in で、既定値は無効のままとした。

検証結果:

- `rustfmt --edition 2024 --check`（変更2ファイル）: 成功
- `git diff --check`: 成功
- `CARGO_BUILD_JOBS=1 cargo check -p ullm-engine --lib`: 成功
- `CARGO_BUILD_JOBS=1 cargo test -p ullm-engine --lib -- --test-threads=1`: 731 passed, 1 ignored

## 次の行動

実 HIP/R9700 で direct/copy の数値一致、device allocation alias、防御的 reset、M/chunk/位置/KV 状態を確認し、D2D 転送量・起動回数・p50/p95・VRAM の実測を取得する。CPU seam はこれらの GPU 証明や本番 worker/selector 統合を代替しない。

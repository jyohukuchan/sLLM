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
  - 当初の CPU 専用 routing seam は production dispatch と独立していたため削除した。
  - production dispatch 自身が使う direct attempt resolver、ping source/destination 選択、route 適用 transition を共通関数へ抽出した。
  - 同じ transition を CPU fault-injection tests から呼び、M={1,2,8,16,32,64,128} の copy/direct 出力一致と有限値を確認した。
  - per-layer の typed output route（direct / copy / admission copy fallback）と direct 要求フラグを記録するようにし、direct の admission 失敗時は workspace→destination コピーを明示して次層の source を更新するよう dispatch を更新した。
  - direct の admission 失敗だけを既存 copy 経路へ退避し、非再利用状態・実行開始後の失敗・record 欠落はコピー再試行せず poison するようにした。
  - callback 観測により CopyFallback の exactly-one copy と stale destination 上書き、Direct の no-copy、成功時だけの ping switch、copy failure 時の source 保持と poison/reset を確認した。
  - operation record finalizer も production とテストで共通化し、record 欠落時の poison と同期 reset 後の復帰を確認した。
  - M=1 は CPU seam でも indirect copy に分類し、本番 native prefill/direct API の幅契約（M>=2）とは混同しないテストを追加した。
  - direct 経路は従来どおり環境変数の明示的 opt-in で、既定値は無効のままとした。

検証結果:

- `rustfmt --edition 2024 --check`（変更2ファイル）: 成功
- `git diff --check`: 成功
- `CARGO_BUILD_JOBS=1 cargo check -p ullm-engine --lib`: 成功
- `CARGO_BUILD_JOBS=1 cargo test -p ullm-engine --lib -- --test-threads=1`: 733 passed, 1 ignored
- `CARGO_BUILD_JOBS=1 cargo test -p ullm-engine qwen35_aq4_model_runtime::tests --lib -- --test-threads=1`: 17 passed
- `CARGO_BUILD_JOBS=1 cargo test -p ullm-engine qwen35_aq4_layer_runtime::linear_attn_step_state_tests --lib -- --test-threads=1`: 19 passed

## 次の行動

実 HIP/R9700 で direct/copy の数値一致、device allocation alias、防御的 reset、M/chunk/位置/KV 状態を確認し、D2D 転送量・起動回数・p50/p95・VRAM の実測を取得する。CPU seam はこれらの GPU 証明や本番 worker/selector 統合を代替しない。

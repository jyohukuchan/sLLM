# SQ8 worker error terminal hardening

## 前回の要点

- 基点は `2f1c1a1a9c3fdd557958dd0b582c736982fbf6a8`。
- request `17d8` の actual capture は、generate 送信から約 0.058 ms 後に
  worker の `error` event を受信していたが、capture が terminal event として扱わず、
  240秒の request timeout まで待機していた。
- consumed request `17d8` の output、sentinel、既存 authorization lineage は不変とする。

## 今回の変更点

- 実際の Rust worker schema を確認し、fatal を別の event type として扱わず、
  `type: "error"` と `recoverable: false` の組み合わせとして厳密に分類した。
- error event は exact field set、exact request ID、固定 code 集合、non-bool を拒否する
  recoverable、1,024 byte 以下の message を必須にした。message 本文や prefix は保持せず、
  byte count、SHA-256、canonical event SHA-256 のみを failure envelope v5 に残す。
- typed error/fatal の受信後は `worker_error` / `worker_fatal` として即時終了し、
  graceful shutdown を試行してから既存の TERM/KILL/reap fallback へ進む。shutdown の
  結果、lifecycle、stderr、return code/signal を失敗証拠へ残す。
- runner と failure receipt writer は worker-error summary の schema、stage、code、
  recoverable、request binding、hash、message privacy、shutdown 状態を独立に検証する。
- duplicate key、未知 code、数値による bool 別名、request mismatch、架空の
  `type: "fatal"`、secret を含む message、hash/length/shutdown 改ざん、正常 reap、
  maintenance/failure receipt 伝播を fake worker と CPU テストで固定した。
- 実際の Rust request validator との静的比較により、直接の失敗原因は promotion request
  の `eos_token_ids: []` が served-model 固定値 `[248044, 248046]` と一致しないことだと
  確定した。Gate と capture は固定 EOS を送信する。validator、served-model EOS、
  telemetry threshold は緩和していない。最初の token が EOS となり pair projection が
  実行されない場合も、既存の positive pair count 条件により fail closed となる。
- GPU、service、sudo、actual execution、authorization の更新は実行していない。

## 検証

jobs=1 で以下を実行した。

```text
PYTHONPATH=. pytest -q \
  tests/test_capture_aq4_resident_executor_record.py \
  tests/test_capture_aq4_sq8_promotion_telemetry.py \
  tests/test_prepare_qwen35_aq4_sq8_overlay_gpu_promotion.py \
  tests/test_run_qwen35_aq4_sq8_overlay_gpu_promotion.py \
  tests/test_qwen35_aq4_sq8_overlay_promotion_receipt.py
# 279 passed

CARGO_BUILD_JOBS=1 cargo test -p ullm-engine --lib \
  promotion_capture_exact_payload_reports_empty_eos_and_accepts_product_eos \
  -- --test-threads=1
# 1 passed, 749 filtered out

python3 -m py_compile <変更した4 tool>
rustfmt --edition 2024 --check crates/ullm-engine/src/sq8_worker_protocol.rs
git diff --check
```

`--lib` を付けない同一 Rust filter は、変更前から存在する unrelated bin
`ullm-aq4-p2-full-model.rs` の `PromotionContract` 初期化に
`authorization_audit`、`authorization_lineage`、`readiness` が不足しているため、
対象 fixture の実行前に compile error となる。今回の変更では修正していない。

## 次の行動

- request `17d8` は consumed のまま再利用しない。
- future authorization lineage current-v2 は、actual-failure receipt SHA-256
  `42be714d5fb46062900146c726d6ee091739a932aa5aeeaf232797a0b6769565` を exact に追記する。
- この新しい implementation commit 自体は authorization ではない。新 commit を対象に
  current implementation GO、独立監査、candidate、request、sentinel を新規生成し、
  すべての identity が一致するまで actual execution を行わない。

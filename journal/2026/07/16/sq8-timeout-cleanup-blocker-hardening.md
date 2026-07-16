# SQ8 timeout cleanup blocker hardening

## 前回の要点

- 基点は `49d1e6efa245b0477138333ca440587956264e5a`。
- timeout evidence、trusted component の保持済み実行、telemetry の安全整数検証は
  実装済みだった。
- 事後監査で、保持済み generator から live receipt writer を再読込する経路、外側
  capture の無期限 reap、監査件数の数値別名、65件以上の lifecycle 契約不一致、
  `redacted_lines` の安全整数上限不足が残っていることが分かった。

## 今回の変更点

- actual receipt writer から generator へ、保持済み receipt validator と trusted
  component source bytes を明示注入した。pin 後に live writer/generator を置換しても、
  actual writer→generator→validator は保持済みバイト列だけを使う。
- 外側 capture は 1350秒の本体上限後に TERM、KILL、最終 reap をすべて宣言済み
  timeout 付きで行う。stdout/stderr drain と pipe close grace も有界化し、未完了時は
  `cleanup_errors` と `capture_outer_cleanup_timeout` を証拠へ残す。
- `implementation_counts`、prefill histogram、`total_steps`、operator invocation、
  load trace の count/byte 値を exact non-bool int かつ `0..9007199254740991` に統一し、
  `int()` による bool/float/overflow の別名化を除去した。
- lifecycle は先頭64件を保持しつつ `last_event` を最新イベントとして扱う producer
  契約に consumer を合わせた。非省略時は最終保持イベントとの一致を必須にし、
  省略時は最新イベントの時系列整合性を検証する。
- `redacted_lines` は capture normalizer と runner の両方で安全整数上限を必須にした。
- bool、float、`SAFE_INT+1`、65件 lifecycle、reap/pipe hang、pin 後の live path
  差し替えを対象とする回帰テストを追加した。
- GPU、service、sudo、actual execution、authorization の更新は実行していない。

## 検証

jobs=1 で以下を実行した。

```text
python3 -m compileall -q tools tests
PYTHONPATH=. pytest -q \
  tests/test_capture_aq4_resident_executor_record.py \
  tests/test_capture_aq4_sq8_promotion_telemetry.py \
  tests/test_qwen35_aq4_sq8_overlay_promotion_receipt.py \
  tests/test_prepare_qwen35_aq4_sq8_overlay_gpu_promotion.py \
  tests/test_run_qwen35_aq4_sq8_overlay_gpu_promotion.py
# 250 passed

PYTHONPATH=. pytest -q \
  tests/test_capture_aq4_resident_executor_record.py \
  tests/test_run_qwen35_aq4_sq8_overlay_gpu_promotion.py \
  tests/test_qwen35_aq4_sq8_overlay_promotion_receipt.py
# 219 passed

PYTHONPATH=. pytest -q <上記3ファイルと関連するgenerator/prepare/lock/telemetry/lineage/tools> \
  --deselect tests/test_generate_served_model.py::test_aq4_reasoning_candidate_binds_v2_worker_separately_from_active_v1
# 321 passed, 1 deselected

git diff --check
```

除外した1件は fixture が元 checkout の絶対パスを保持しているため、隔離 worktree の
`ROOT` と一致しない既知の環境依存テストである。変更対象の失敗ではない。

## 次の行動

- 新しい commit/tree/source archive identity を確定し、独立監査へ渡す。
- actual execution は、新しい authorization lineage と固定 request/output/sentinel が
  揃うまで実施しない。

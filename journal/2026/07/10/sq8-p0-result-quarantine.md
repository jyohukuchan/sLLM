# SQ8 P0 既存結果の隔離

日付: 2026-07-10

## 前回の要点

Qwen3-14B-FP8の旧v0.1 sidecarはsource `weight_scale_inv`を適用しておらず、同一モデルの品質・性能比較には使えない。一方、loader、40層接続、batch、D2D handoffの接続診断としては履歴を残す価値がある。

## 今回の変更点

- `status=ok`は実行成功として維持し、top-level `result_validity`を追加した。
- 対象selectorに一致するuLLM 21行・8 JSONLを`quarantined/connection_diagnostic`として隔離した。
- implementation、quality comparison、performance comparisonをすべて無効とし、reason code `source_fp8_weight_scale_inv_not_applied`を記録した。
- vLLM行は変更していない。
- normalized throughput gateはperformance validityを暗黙に必須とした。
- `--require-implementation-valid`を追加し、exact boolean trueとartifact manifest SHA-256を必須にした。
- 一般summaryとSQ8 batch-grid summaryは隔離行を既定で除外し、`--include-quarantined`時だけ表示する。
- 影響する8 summaryへ警告を追加した。
- 構造化JSONL移行ツール`tools/quarantine-invalid-sq8-source-scale-results.py`を追加した。

## 検証

- dry-run: 21行・8ファイルを検出。
- apply: 21行を変更。
- 2回目apply: 変更0行で冪等。
- 従来通過していたno-host b2/b4/b8 normalized comparisonは、各uLLM行についてimplementation/performance validityを理由コード付きで拒否し`exit=2`。
- gateなしの既定batch-grid summaryは隔離uLLM行を除外し、vLLM 3行だけを表示。
- `python3 -m unittest tests.test_external_benchmark_batch_parser tests.test_summarize_sq8_vllm_batch_grid tests.test_summarize_benchmark_results tests.test_quarantine_invalid_sq8_source_scale_results`: 85 tests passed。
- `python3 -m py_compile`で変更したPython tool/testを確認。
- `git diff --check`: pass。

## 次の行動

P1として、旧v0.1を暗黙変換せず、新しいcanonical artifact schemaでsource raw F8 payloadとBF16 128x128 block scaleをbyte-exactに保持する。synthetic fixture、実layer0 q_proj golden、Rust canonical readerの順で実装する。

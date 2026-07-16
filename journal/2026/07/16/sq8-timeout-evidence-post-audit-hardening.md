# SQ8 timeout evidence post-audit hardening

## 前回の要点

- 基点は `ea2c5be52241ec596439ad9add1481437874666f`。
- ready/request timeout の分離、failure envelope v4、Gate の trusted_components exact-4
  path/SHA 契約はすでに実装済みだった。
- 事後監査で、telemetry counter の安全整数上限不足と、trusted component の
  検証後パス再参照による TOCTOU が残っていることが分かった。

## 今回の変更点

- capture、runner、receipt writer の成功・失敗 telemetry counter を、
  `type(value) is int` かつ `0..9007199254740991` に統一した。
- diagnostic host staging は同じ安全整数条件に加えて exact zero を必須にした。
- Gate の path/SHA exact-4 契約は維持したまま、trusted component を
  `O_RDONLY|O_CLOEXEC|O_NOFOLLOW` で開き、regular、nlink=1、device/inode、
  サイズ、SHA-256 を開いた fd から検証した。
- 検証済みバイト列を保持し、capture は seal 済み memfd を
  `/proc/self/fd/N` と `pass_fds` で実行する。receipt writer と served-model
  generator は保持したバイト列を compile/exec する。
- evidence には canonical path、SHA-256、device、inode だけを記録し、fd 番号は
  記録しない。保有 fd は context 終了時に決定的に close する。
- 検証後の regular-file 置換、unlink、symlink 置換に対して、元のバイト列を
  実行するか fail closed となり、置換内容を実行しない敵対的テストを追加した。
- GPU、service、sudo、新しい authorization、actual execution は実行していない。

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
git diff --check
```

結果は `208 passed`。

## 次の行動

- 新しい source identity を対象に独立監査と authorization lineage を再生成する。
- actual execution は、その新しい authorization と固定 request/output/sentinel が
  揃うまで実施しない。

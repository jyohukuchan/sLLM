# SQ8 serving candidate chunk validator

## 前回の要点

- `validate-sq8-serving-chunks.py` はM8の `ullm.sq8.serving_chunks.v3`、chunk幅8、all-M1比較だけを固定値で検証していた。
- 既存CLIの既定modeは `m8-chunk8` であり、保存済みM8 evidenceとの互換性を維持する必要がある。

## 今回の変更点

- `--chunk-mode` に `m8-chunk8`、`m32-chunk32`、`m128-chunk128` を追加した。既定値はM8のまま。
- M32/M128はschema `ullm.sq8.serving_chunks.v4`、chunk幅32/128、各fixed cached-prefix implementation IDを組として検証する。
- execution-unit幅は選択modeから計算し、full chunk後の端数は従来どおりM1 unitとして検証する。
- all-M1 resultは既存 `ullm.sq8.serving_smoke.v2`、`prefill_chunk_tokens=8`、`sq8.sequential-m1.v1`を維持する。
- M32/P32とM128/P128のfixtureで、all-M1比較とvLLM source gateまで再計算するtestを追加した。

## 次の行動

- candidate runnerが生成したM32/M128 resultを、対応する `--chunk-mode` で検証する。
- 候補昇格時だけ保存先へvalidation JSONを出力し、M8 evidenceは既存CLIのまま再利用する。

## Verification

- `python3 -m py_compile tools/validate-sq8-serving-chunks.py tests/test_sq8_serving_chunks.py`
- `python3 -m pytest -q tests/test_sq8_serving_chunks.py`: 10 passed
- `python3 -m pytest -q tests/test_sq8_serving*.py`: 124 passed, 14 subtests passed
- 保存済みM8 P32/P128/P512 evidenceを既定CLIと `--require-build-identity` で再検証: passed

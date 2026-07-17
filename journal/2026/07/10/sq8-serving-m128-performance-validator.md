# SQ8 serving M128 performance validator

## 前回の要点

- performance validatorはM8 raw v1、chunk幅8、fixed-M8 implementationを固定値で検証していた。
- 保存済みM8 validationはP3584の既知のTTFT gate failureを含むため、一般化後も同じvalidation JSONを再生成する必要がある。

## 今回の変更点

- `--prefill-mode` に `m128-chunk128` を追加し、raw v2、chunk幅128、`sq8.fixed-m128-cached-prefix.v1` を一組として検証する。
- 既定modeはM8のままとし、raw v1と既存implementation IDを維持した。
- prompt execution call数を `P / chunk + P % chunk`、progressを `calls - 1` としてmode別に再計算する。
- M128のP32 TTFT/decodeは32回のM1 prompt call、31 progress、decode総call 95として検証する。
- TTFT/decodeのhard threshold、timer、VRAM、EOS、build identity契約は変更していない。

## 次の行動

- performance runnerが出力するM128 raw v2を `--prefill-mode m128-chunk128` で検証する。
- M128の正式evidenceが得られた後も、保存済みM8 evidenceは引数追加なしの既定CLIで再現確認する。

## Verification

- `python3 -m pytest -q tests/test_sq8_serving_performance.py`: 29 passed
- `python3 -m pytest -q tests/test_sq8_serving*.py`: 128 passed, 14 subtests passed
- 保存済みM8 resultから再生成したvalidation JSONは保存済みvalidationとbyte exact、SHA-256 `f9c1dc6d91d08490c9c433a76e450ff0e05833ad15af67b56511369702219e88`

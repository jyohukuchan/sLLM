# SQ8 serving M128 deep-boundary validator

## 前回の要点

- deep-boundary validatorはM8 deep raw v1、chunk幅8、448 prefill callsを固定値で検証していた。
- 3584 promptと512 actual generatedによる最終KV 4095、position 4094のtrace/reset/build identity契約は維持する必要がある。

## 今回の変更点

- `--prefill-mode` に `m128-chunk128` を追加し、deep raw v2、chunk幅128、`sq8.fixed-m128-cached-prefix.v1` を一組として検証する。
- 既定modeはM8のままとし、deep raw v1と保存済みvalidation出力を維持した。
- prefill unit幅を選択modeから再構成し、call数を `P / chunk + P % chunk`、progressを `calls - 1` として検証する。
- M128は28 prefill calls、27 progress、511 decode calls、総539 callsとして検証する。
- 512 generated steps、EOS無視、最終KV 4095、position 4094、logical block 255、reset、build identityの判定は変更していない。

## 次の行動

- deep-boundary runnerが出力するM128 raw v2を `--prefill-mode m128-chunk128` で検証する。
- M128昇格結果でも512 generated stepsと全40層cache traceを省略せず保存する。

## Verification

- `python3 -m pytest -q tests/test_sq8_serving_deep_boundary.py`: 18 passed
- `python3 -m pytest -q tests/test_sq8_serving*.py`: 132 passed, 14 subtests passed
- 保存済みM8 deep resultから再生成したvalidation JSONはbyte exact、SHA-256 `8b5833d70375f7e509ebbb3bed566f4c19eac3185f365578e04ea00c5ae4f5b9`

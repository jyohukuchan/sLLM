# T2 SQ FP8 qkv layers3/7/11 prompt bundle

## 前回の要点

- layer3+7 `q16/k16/v16` はfull mixed prompt bundleでstrict top1 `3 / 3` を維持した。
- SQ FP8 direct triple countは `46` だった。

## 今回の変更点

- `benchmarks/results/2026-07-09/sq-fp8-qkv-layers3-7-11-q16-k16-v16-policy-v0.1.json` を追加した。
- artifactを `/tmp/ullm-sq-fp8-qkv-layers3-7-11-q16-k16-v16-policy-v0.1-artifact` に生成した。
- AQ4 baselineはlayer3+7 runから再利用し、SQ側だけを新規測定した。

## 結果

| row | FP8 tensors | prefill tok/s | decode tok/s | end-to-end tok/s | final top1 |
| --- | ---: | ---: | ---: | ---: | --- |
| AQ4 baseline reused | 0 | 66.811351 | 80.883171 | 35.370726 | `24218,4105,329` |
| SQ `layers3/7/11 q16/k16/v16` | 9 | 63.183461 | 76.326909 | 33.173002 | `24218,4105,329` |

- strict top1は `3 / 3` pass。
- SQ FP8 direct triple countは `69`。
- `case_a` のSQ top1 margin over rank2は `0.009701252`。

## 次の行動

1. layer3+7+11 `q16/k16/v16` を現在のQKV triple passing boundaryとして保存する。
2. 次はlayer15 QKVを同じrow-block16で足す。
3. layer15追加で崩れる場合は、layer15のQ/K/Vを単体またはpairで分解する。

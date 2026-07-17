# T2 SQ FP8 qkv layers3/7 prompt bundle

## 前回の要点

- layer3 `q16/k16/v16` はfull mixed prompt bundleとB=1/4/8 short guardでstrict top1を維持した。
- 次は同じrow-block16 QKVをlayer7へ足して、full mixed prompt bundleで崩れるかを見る段階だった。

## 今回の変更点

- `benchmarks/results/2026-07-09/sq-fp8-qkv-layers3-7-q16-k16-v16-policy-v0.1.json` を追加した。
- artifactを `/tmp/ullm-sq-fp8-qkv-layers3-7-q16-k16-v16-policy-v0.1-artifact` に生成した。
- R9700 full mixed `manifest-all` prompt bundleでAQ4 baselineとSQ候補を比較した。
- 途中でgenerated token指定を `5,4105,8201` としてしまい、各request生成数として解釈されてtimeoutした。正しくは `1` を渡す。

## 結果

| row | FP8 tensors | prefill tok/s | decode tok/s | end-to-end tok/s | final top1 |
| --- | ---: | ---: | ---: | ---: | --- |
| AQ4 baseline | 0 | 66.811351 | 80.883171 | 35.370726 | `24218,4105,329` |
| SQ `layers3/7 q16/k16/v16` | 6 | 62.866188 | 77.707480 | 34.095768 | `24218,4105,329` |

- strict top1は `3 / 3` pass。
- SQ FP8 direct triple countは `46`。
- `case_a` のSQ top1 margin over rank2は `0.003049851`。

## 次の行動

1. layer3+7 `q16/k16/v16` を現在のQKV triple passing boundaryとして保存する。
2. 次はlayer11 QKVを同じrow-block16で足す。
3. layer11追加で崩れる場合は、layer11のQ/K/Vを単体またはpairで分解する。

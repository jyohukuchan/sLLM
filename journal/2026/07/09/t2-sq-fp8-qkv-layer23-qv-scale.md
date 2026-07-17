# T2 SQ FP8 qkv layer23 q/v scale

## 前回の要点

- layer23 `k16` はprompt bundleとB=1/4/8 short guardでstrict top1を維持した。
- layer23 `q8` と `v16` は単体でも`case_a`を崩した。
- layer27 `k16` / `k8` も`case_a`を崩したため、現branchのcoverage拡大はlayer23 q/vのscale強化で回復可能かを見る段階だった。

## 今回の変更点

- layer23 `k16` 通過branchに、layer23 `q4`、`v8`、`v4` をそれぞれ追加してR9700 prompt bundle guardを実行した。
- SQ側は `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_TRIPLE_KERNEL=1` と `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_KERNEL=1` でdirect kernelを必須化した。
- 結果は次に保存した。
  - `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-qkv-layer23-q4-v1.md`
  - `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-qkv-layer23-v8-v1.md`
  - `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-qkv-layer23-v4-v1.md`
  - `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-qkv-layer23-qv-scale-v1-comparison.json`

## 結果

| row | FP8 tensors | prefill tok/s | decode tok/s | end-to-end tok/s | final top1 | strict top1 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| SQ `layer23 q4` | 17 | 59.477712 | 72.272506 | 31.749085 | `24218,5582,329` | 2 / 3 |
| SQ `layer23 v8` | 17 | 59.730593 | 73.144428 | 33.132245 | `24218,5582,329` | 2 / 3 |
| SQ `layer23 v4` | 17 | 59.687315 | 73.212004 | 32.586644 | `24218,5582,329` | 2 / 3 |

## 判断

- layer23 `q4` は`case_a`でstrict top1を落とした。
- layer23 `v8` は`case_a`でSQ top1 `5582` とAQ4 top1 `4105` の差が `0.000022412` まで縮まったが、strict top1は回復しなかった。
- layer23 `v4` も`case_a`でstrict top1を落とした。
- row-block幅の単純な縮小だけではlayer23 q/vを回復できない。
- 現branchのpassing boundaryはlayer23 `k16` までとし、q4/v8/v4はfailure guardとして保持する。

## 次の行動

1. q/vをさらに追う場合は、別format/layout、text-level guard、またはlogit近傍を安定させるSQ基準を検討する。
2. SQ候補評価基盤側へ戻る場合も、現時点のpassing branchはlayer23 `k16`までとして扱う。

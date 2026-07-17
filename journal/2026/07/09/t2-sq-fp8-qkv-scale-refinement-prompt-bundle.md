# T2 SQ FP8 qkv scale refinement prompt bundle

## 前回の要点

- `q/k` pair単体と `v` 単体はfull mixed prompt bundleでstrict top1 `3 / 3`。
- `q32/k16/v32` は `case_a` で `5582` が `4105` を上回った。

## 今回の変更点

- `q32/k16/v16` と `q16/k16/v16` のpolicy/artifactを追加した。
- どちらもfull mixed prompt bundleで、triple direct SQ FP8 kernel必須で測った。

## 結果

- `q32/k16/v16`: strict top1 `2 / 3`。`case_a` はSQ `5582` がAQ4 top1 `4105` を `0.000260353` 上回る。
- `q16/k16/v16`: strict top1 `3 / 3`。`case_a` はSQ `4105` がrank2 `5582` を `0.002023697` 上回る。
- どちらも `sq_projection_boundary=triple`、`sq_fp8_triple_matvec_count=23`。

## 次の行動

1. `q16/k16/v16` をlayer3 QKV triple pass boundaryとして扱う。
2. layer7以降へ広げる前に、B=1/4/8 short guardまたは追加promptで再確認する。

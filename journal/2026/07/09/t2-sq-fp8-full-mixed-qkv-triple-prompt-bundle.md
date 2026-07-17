# T2 SQ FP8 full mixed qkv triple prompt bundle

## 前回の要点

- layer3 `q/k/v` SQ FP8候補は短いB=1/4/8 smokeではAQ4 final top1と一致していた。
- telemetryでSQ FP8 direct `triple` projection境界を確認できるようになっていた。

## 今回の変更点

- full mixed `manifest-all` request-state pathで、`len4`、`case_a`、`case_b` のprompt bundleをAQ4/SQで比較した。
- parserを修正し、runtimeが出す `:` 区切りのtop-k matrixをJSONLへ保存できるようにした。
- 比較結果を `comparison.json` と `results.jsonl` に保存した。

## 結果

- AQ4 final top1: `24218,4105,329`
- SQ final top1: `24218,5582,329`
- strict top1: `2 / 3`
- SQ telemetry: `sq_projection_boundary=triple`, `sq_fp8_triple_matvec_count=23`
- `case_a` はtop8 overlap `8 / 8` だが、AQ4 top1 `4105` とSQ top1 `5582` が入れ替わった。

## 次の行動

1. layer3 `q/k/v` tripleは回帰境界として保存し、policy昇格から外す。
2. 次は `q/k` pair、`k`単体、または別scale粒度の `q/k/v` をprompt bundleで検証する。

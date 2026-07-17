# T2 policy JSON builder input

## 前回の要点

- `kup6_gate5_down5` はlen4/case_a/case_bでstrict top1一致だった。
- ただしcase_aのtop8 overlapは `2 / 8` と低く、full SQ policyではなく6層regression subsetとして扱う。
- 選択FP8/fallback方針は `sq-fp8-policy-v0.1` として保存済みだった。

## 今回の変更点

- `tools/build-sq-fp8-w8a16-artifact.py` に `--policy-json` を追加した。
- policyから `candidate_id`、`fp8_selection.include_regex`、scale granularity、row-block widthをbuilder defaultとして解決するようにした。
- 生成manifestに `policy` blockを入れ、policy ID、source policy path、FP8 selection、fallback policy、prompt bundle resultを保存するようにした。
- dry-runで `kup6_gate5_down5` が `22` FP8 tensors、`753` passthrough tensors、row-block32として解決されることを確認した。
- `tests/test_build_sq_fp8_artifact_policy.py` を追加した。

## 次の行動

1. 次のSQ FP8 artifact生成では `--policy-json` を使う。
2. T1 real batch runnerを進め、SQ候補評価で使えるthroughput行を作る。
3. throughput比較ではoverlay load timingを使わず、native FP8またはmaterialization-aware runtime pathを使う。

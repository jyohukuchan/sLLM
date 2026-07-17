# T2 SQ FP8 full mixed v-only prompt bundle

## 前回の要点

- `q/k/v` tripleはfull mixed prompt bundleで `case_a` がtop1 swapした。
- `q/k` pairは同じprompt bundleでstrict top1 `3 / 3` を維持した。

## 今回の変更点

- layer3 `v_proj` 単体のSQ FP8 policyを追加した。
- artifactを `/tmp/ullm-sq-fp8-v-layer3-v32-policy-v0.1-artifact` に生成した。
- `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_KERNEL=1` でsingle direct kernelを必須化してfull mixed prompt bundleを測った。

## 結果

- AQ4 final top1: `24218,4105,329`
- SQ final top1: `24218,4105,329`
- strict top1: `3 / 3`
- SQ telemetry: `sq_projection_boundary=single`, `sq_fp8_single_matvec_count=23`
- `case_a` のSQ top1 marginは `0.016001701`。

## 次の行動

1. `q/k` と `v` は単独では通るため、`q/k/v` failureは累積・相互作用driftとして扱う。
2. 次は `q/k/v` 同時適用を維持し、`v16` または `q16/v16` のscale粒度を試す。

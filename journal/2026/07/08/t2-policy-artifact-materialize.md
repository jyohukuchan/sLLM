# T2 policy artifact materialize

## 前回の要点

- `kup6_gate5_down5` はlen4、case_a、case_bでstrict top1を維持した。
- `tools/build-sq-fp8-w8a16-artifact.py --policy-json` は、policy JSONからinclude regex、candidate ID、row-block scaleをmanifestへ反映できる。
- dry-runでは `22` FP8 tensors、`753` passthrough tensors、row-block32として解決された。

## 今回の変更点

- `benchmarks/results/2026-07-08/sq-fp8-kup6-gate5-down5-policy-v0.1.json` から実FP8 payload artifactを生成した。
- artifactは `/tmp/ullm-sq-fp8-kup6-gate5-down5-policy-v0.1-artifact` に置いた。
- manifestで `policy_id=kup6_gate5_down5`、`fp8_tensor_count=22`、`passthrough_tensor_count=753`、row-block32を確認した。
- R9700 device index `2` で `model.language_model.layers.3.self_attn.k_proj.weight` を `sq-fp8-materialize-smoke` に通し、`roundtrip_max_abs_diff=0`、`verified=true` を確認した。
- 結果を `benchmarks/results/2026-07-08/sq-fp8-kup6-gate5-down5-policy-artifact-v0.1.*` に保存した。
- 計画書、state freeze、SQ FP8 artifact specへ「runtime boundaryは確認済み、throughput/最終SQ policyではない」と追記した。

## 次の行動

1. T1 full package real-batch runnerを実装する。
2. FP8 SQ throughput比較ではhost-side materialize/load timingを使わない。
3. `/tmp` のartifactは必要時に `--policy-json` から再生成する。

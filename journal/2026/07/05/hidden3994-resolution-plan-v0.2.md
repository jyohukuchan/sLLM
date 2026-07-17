# hidden3994 resolution plan v0.2

## 前回の要点

- Qwen3.5-9B p4p46 in-projection packageのprefix driftはhidden `3994` に強く局所化している。
- Layer6 `mlp.down_proj.weight[3994]` row-scale、`p4p65-inproj`、`p4p65+row3456`、既存layer8 manifest packagesはいずれもfive-fixture gateでrejectされた。
- tokens401ではlayer8入力時点ですでにhidden `3994` が低く、layer8単体のrow量子化だけではなく入力ドリフト増幅として扱う必要がある。

## 今回の変更点

- `uLLM-project/docs/plans/qwen-prefix-hidden3994-resolution-plan-v0.2.md` を追加した。
- v0.2では、row3456補償を維持したままhidden3994問題を解くことを目標にした。
- 最初の実行候補を、weak `layer8-upfit` manifest package gridに置いた。
- manifest補正で通らない場合は、activation-aware / row-awareなquantizer policyへ切り替える分岐を明示した。

## 次の行動

1. `tools/build-qwen-row-scale-manifest-package.py` を追加し、manifest row-scale JSONからhardlink package copyを再現可能に作る。
2. `layer8-upfit` の弱倍率gridを5 fixtureで評価する。
3. 通らなければ、tokens401とtokens1のhidden3994 chainを比較し、paired manifest candidateまたはquantizer policy branchへ進む。

## 16:10 JST Progress

- `tools/build-qwen-row-scale-manifest-package.py` を追加した。
  - 初期実装でdestination `manifest.json` のhardlinkを切らずに書いてsource package manifestを一時的に汚染した。
  - 修正後はdestination `manifest.json` を `unlink()` してから書く。
  - source packageはrow3456の4-entry manifestへ復旧し、link count `1` を確認した。
- `tools/generate-qwen-manifest-row-scale-grid.py` を追加した。
- layer8 `mlp.up_proj.weight[6340]` weak gridを作成し、tokens1 prefilterで `1.004` と `1.008` のみ5 fixture評価へ進めた。
- 5 fixture gate結果:
  - `layer8-up6340-s1p004`: `hold`, median improvement `0`, max regression `0.000408172607`
  - `layer8-up6340-s1p008`: `hold`, median improvement `0`, max regression `0.000799179077`
- `1.008` はtokens201を `1.140727997 -> 1.140106201` に改善するが、tokens401は `0.959306717 -> 0.959340096` とわずかに悪化する。
- chain比較では、tokens1のlayer8 hidden3994は正方向、tokens401のlayer8 hidden3994は負方向に増幅しており、weak up6340はtokens401の入力ドリフト増幅を直していない。
- tokens401由来のlayer8 local候補はtokens1を `0.645338058 -> 0.662992477` に悪化させたため、full matrix前にrejectした。

## Updated Next Action

1. weak `layer8-up6340` は単独manifest fixとして採用しない。
2. direct tokens401 layer8 local row-scaleも一般解として追わない。
3. 次はtokens401 layer8入力ドリフトの上流、またはactivation-aware / row-aware quantizer policyへ進む。

## 16:12 JST Quantizer Policy Branch

- `ullm-quant` の現行policy境界を確認した。
  - `resolve_aq_policy`
  - `family_for_tensor`
  - `quant_assignment`
  - `run_one_direct_package_convert`
- 現行実装はtensor family単位のformat assignmentで、row単位またはactivation statistic入力はまだない。
- `p4p46_inproj` baseline dry-run:
  - high tensors: `114`
  - low tensors: `141`
  - estimated output bytes: `9121922016`
- custom `p4p46 + mlp_up` dry-run:
  - high tensors: `147`
  - low tensors: `108`
  - estimated output bytes: `9225731040`
  - estimated output increase: `103809024` bytes
- family-wide `mlp_up` high化はpackage size上は安いが、広すぎてactivation-awareではない。
- 次はbroad family-wide packageより、selected tensor override policyまたはactivation-aware / row-aware policyを優先する。

## 16:15 JST Per-Tensor High Override

- `ullm-quant` に repeatable `--aq-high-tensor <TENSOR_NAME>` を追加した。
- `AqPolicyPlan` に `high_tensors` を追加し、既存plan JSON互換のため `serde(default, skip_serializing_if = "Vec::is_empty")` を付けた。
- `quant_assignment` は `supported_input` がtrueの場合だけ、exact tensor matchをfamily判定と同じhigh assignmentとして扱う。
- `p4p46_inproj + model.language_model.layers.8.mlp.up_proj.weight` dry-run:
  - high tensors: `115`
  - low tensors: `140`
  - estimated output bytes: `9125067744`
  - baselineからのestimated output increase: `3145728` bytes
- layer8 `mlp.up_proj.weight` はhigh、layer9 `mlp.up_proj.weight` はlowのままであることをplan JSONで確認した。
- 検証:
  - `cargo test -p ullm-quant -- --test-threads=1`
  - `cargo check -p ullm-quant`
  - `cargo fmt --all --check`
  - `git diff --check`
- 次はこのplanからfull direct packageを生成し、少数fixtureでprefix smokeを走らせる。

## 16:28 JST Targeted Layer8 MLP Up Package Prefilter

- `p4p46_inproj + model.language_model.layers.8.mlp.up_proj.weight high` のfull direct packageを作成した。
  - package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-layer8-mlp-up-high-reservoir65536-jobs64.ullm.d`
  - summary: `benchmarks/results/2026-07-05/engine/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-layer8-mlp-up-high-reservoir65536-jobs64.json`
  - selected tensors: `255`
  - passthrough tensors: `520`
  - codebooks: `13`
  - failures: `0`
  - build wall time: `1:34.98`
  - max RSS: `3734884` KiB
- 独立verifyも成功した。
  - quantized tensors: `255`
  - passthrough tensors: `520`
  - passthrough payload bytes: `5049777120`
  - wall time: `0:51.78`
  - exit status: `0`
- 既存のlayer6/layer10 row3456 manifest補償をhardlink package copyで付与した。
  - package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-layer8-mlp-up-high-row-scale-layer6-layer10.ullm.d`
  - row_scale entries: `4`
  - source package manifest row_scale entriesは`0`のまま、output manifest row_scale entriesは`4`。
- tokens1/tokens401 prefilterを実行した。
  - tokens1: `0.645338058 -> 0.627647400`、改善 `0.0176906586`
  - tokens401: `0.959306717 -> 0.974622726`、悪化 `0.0153160095`
  - gate decision: `reject`
- 結論:
  - exact layer8 `mlp.up_proj.weight` high化はtokens1には効くが、tokens401の主要hidden3994差分を悪化させる。
  - five-fixture matrixには進めない。
  - 次は別のtargeted tensor、特にlayer8のattention/linear_attn側、または上流入力ドリフト源を試す。

## 16:42 JST Targeted Layer8 QKV Package Prefilter

- `p4p46_inproj + model.language_model.layers.8.linear_attn.in_proj_qkv.weight high` のfull direct packageを作成した。
  - package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-layer8-qkv-high-row-scale-layer6-layer10.ullm.d`
  - summary: `benchmarks/results/2026-07-05/engine/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-layer8-qkv-high-row-scale-layer6-layer10-jobs64.json`
  - selected tensors: `255`
  - passthrough tensors: `520`
  - codebooks: `13`
  - failures: `0`
  - build wall time: `1:40.36`
  - max RSS: `3743064` KiB
- 独立verifyも成功した。
  - quantized tensors: `255`
  - passthrough tensors: `520`
  - passthrough payload bytes: `5049777120`
  - wall time: `0:54.19`
  - exit status: `0`
- tokens1/tokens401 prefilterを実行した。
  - tokens1: `0.645338058 -> 0.651521683`、悪化 `0.00618362427`
  - tokens401: `0.959306717 -> 0.919565201`、改善 `0.0397415161`
  - gate decision: `reject`
- 結論:
  - exact layer8 qkv high化はtokens401には効くが、tokens1の主要hidden3994差分を悪化させる。
  - qkv単独ではfive-fixture matrixには進めない。
  - MLP-up highはtokens1改善・tokens401悪化、qkv highはtokens1悪化・tokens401改善なので、次はcombined qkv+MLP-upを試す価値がある。

## 17:20 JST Combined Layer8 QKV + MLP Up Accepted

- `p4p46_inproj` に次のexact high tensor overrideを追加したpackageを作成した。
  - `model.language_model.layers.8.linear_attn.in_proj_qkv.weight`
  - `model.language_model.layers.8.mlp.up_proj.weight`
- package:
  - `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-layer8-qkv-mlp-up-high-row-scale-layer6-layer10.ullm.d`
- row3456 manifest compensationは維持した。
- Build/verify:
  - selected tensors: `255`
  - passthrough tensors: `520`
  - codebooks: `14`
  - failures: `0`
  - total file bytes: `9127853385`
  - independent verify: exit `0`
- CPU five-fixture gate:
  - decision: `accept`
  - mean improvement: `0.0479898453`
  - median improvement: `0.022603035`
  - max regression: `0`
- CPU per-fixture:
  - tokens1: `0.645338058 -> 0.629640579`
  - tokens101: `1.0805254 -> 1.0805254`
  - tokens201: `1.140728 -> 1.00050735`
  - tokens301: `1.37130928 -> 1.30988121`
  - tokens401: `0.959306717 -> 0.936703682`
- Backend verification:
  - R9700 device index `2`, five-fixture gate: `accept`
  - V620 device index `1`, five-fixture gate: `accept`
  - V620 device index `3`, five-fixture gate: `accept`
- 判断:
  - 単独MLP-up highと単独qkv highはそれぞれ逆方向に失敗したが、combinedではfixture conflictが解けた。
  - hidden3994問題は、現時点では「さらに闇雲にデバッグする」段階ではなく、受入済みcandidateをdurable policyとしてどう表現するかを決める段階。

## 18:30 JST Named Policy Implementation

- `ullm-quant` に `qwen35_9b_p4p46_hidden3994_v1` を追加した。
- policy内容:
  - base: `p4p46_inproj`
  - exact high tensor:
    - `model.language_model.layers.8.linear_attn.in_proj_qkv.weight`
    - `model.language_model.layers.8.mlp.up_proj.weight`
  - generic `p4p46_inproj` は変更しない。
- dry-run:
  - plan: `benchmarks/results/2026-07-05/engine/qwen-hidden3994-policy-qwen35-9b-p4p46-hidden3994-v1-plan.json`
  - high tensors: `116`
  - low tensors: `139`
  - passthrough tensors: `520`
  - estimated output bytes: `9127164896`
- regenerated package:
  - `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-qwen35-hidden3994-v1-row-scale-layer6-layer10.ullm.d`
  - selected tensors: `255`
  - passthrough tensors: `520`
  - codebooks: `14`
  - failures: `0`
  - total file bytes: `9127853385`
  - build wall time: `1:34.67`
  - max RSS: `3739852` KiB
- independent verify:
  - quantized tensors: `255`
  - passthrough tensors: `520`
  - passthrough payload bytes: `5049777120`
- local verification:
  - `cargo fmt --all --check`
  - `cargo check -p ullm-quant`
  - `cargo test -p ullm-quant -- --test-threads=1`
  - `cargo build -p ullm-quant`
- CPU five-fixture rerun:
  - matrix: `benchmarks/results/2026-07-05/engine/qwen-prefix-smoke-matrix-qwen35-hidden3994-policy-cpu-five-fixture/summary.json`
  - summary: `benchmarks/results/2026-07-05/engine/qwen-prefix-qwen35-hidden3994-policy-cpu-five-fixture-summary.json`
  - gate: `benchmarks/results/2026-07-05/engine/qwen-prefix-qwen35-hidden3994-policy-cpu-five-fixture-gates.json`
  - decision: `accept`
  - mean improvement: `0.0479898453`
  - median improvement: `0.022603035`
  - max regression: `0`
- committed in `uLLM-project`: `cf7ed7c Add Qwen hidden3994 AQ policy preset`

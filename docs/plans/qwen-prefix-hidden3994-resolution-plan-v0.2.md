# Qwen Prefix Hidden3994 Resolution Plan v0.2

## 前回の要点

- Qwen3.5-9B p4p46 in-projection packageの`actual_prefix` smokeでは、hidden `3994` が5 fixtureすべてで主要な差分座標になっている。
- Layer6 `mlp.down_proj.weight[3994]` row-scaleは局所的には効くが、five-fixture gateでは不採用になった。
  - 最小の正方向scale `1.004` でもtokens401が `0.959306717 -> 0.964665413` に悪化する。
  - scale `1.026471714` はtokens1/tokens101/tokens301を改善するが、tokens201とtokens401を悪化させる。
- 既存manifestのrow3456補償は必要である。外すとtokens1/tokens101/tokens301/tokens401が大きく悪化する。
- `p4p65-inproj` 単体と `p4p65+row3456` はどちらもfive-fixture gateを通らない。
- tokens401のlayer8 token9 hidden `3994` は、layer8単体のrow量子化だけではなく、layer8入力時点ですでに `-0.464719772` 低い。
- 既存のlayer8 manifest packageでは `layer8-upfit` が最も近いが、tokens1を `+0.00354194641` 悪化させるためrejectである。

## 今回の変更点

この計画はv0.1の「hidden3994を調べる計画」から一段進めて、解決までの作業を次の2本に整理する。

1. まずmanifest metadataで表現できる弱い補正を、5 fixture gateで機械的に探索する。
2. manifest補正で通らない場合は、row個別の後処理ではなくquantizer policy改善へ切り替える。

特に、次の事実を前提にする。

- smoke-only row-scaleは `mlp.up_proj.weight` を直接調整できない。
- `layer8-upfit` はpackage manifestでのみ検証できる候補である。
- 解決候補はrow3456補償を保持したまま、hidden3994の入力ドリフト増幅を弱める必要がある。

## 次の行動

最初に作るべきものは、任意のmanifest row-scale JSONからhardlink package copyを生成し、5 fixture smoke matrixまで流す再現可能な小ループである。

そのループで、`layer8-upfit` の弱倍率gridを評価する。これが通らない場合は、tokens401のlayer8入力ドリフトの上流をさらに分解し、quantizer policy候補へ移る。

## Goal

Qwen3.5-9B p4p46 in-projection packageのprefix driftについて、hidden `3994` が支配的な差分になる問題を、複数fixtureに過適合しない形で解決する。

ここでの「解決」は次のどちらかを指す。

1. package manifest metadataまたはquantizer policyで再現可能に適用できる修正候補が、受入gateを通る。
2. 現行p4p46 direct package policyでは受入gateを満たせないことを、探索範囲、反例、次の量子化方針まで含めて明確にする。

## Success Criteria

修正候補を採用できる条件:

1. CPU `package-golden-prefix-smoke` の `actual_prefix 0..12` で最低5 fixtureを通す。
2. どのfixtureでも最終max absがbaselineより `0.001` を超えて悪化しない。
3. median final max abs improvementが `0.005` 以上である。
4. mean abs、MSE、cosine similarityが悪化方向へ偏らない。
5. 既存row3456 manifest compensationを維持する。
6. 修正はruntime hard-codeではなく、次のどちらかで表現する。
   - direct package manifest metadata
   - quantizer policyまたはpackage生成処理
7. CPU acceptance後にR9700とV620で同じfixture subsetを通し、backend固有差分が既存許容範囲内である。
8. 関連テストとJSON artifact検証が通る。

未解決と判断する条件:

- 5 fixture gateを通るmanifest候補がなく、quantizer policy候補でもhidden3994を改善すると別fixtureが `0.001` 超で悪化する。
- その場合は、p4p46 policyの限界またはfixture集合の再設計が必要な状態として記録する。

## Fixed Benchmark Set

Primary fixture set:

| label | fixture |
| --- | --- |
| `tokens1` | `benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16` |
| `tokens101` | `benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16-tokens101-116` |
| `tokens201` | `benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16-tokens201-216` |
| `tokens301` | `benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16-tokens301-316` |
| `tokens401` | `benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16-tokens401-416` |

Baseline package:

```text
/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer10.ullm.d
```

Baseline worsts:

| fixture | max abs | layer | token | hidden |
| --- | ---: | ---: | ---: | ---: |
| `tokens1` | `0.645338058` | 11 | 7 | 3994 |
| `tokens101` | `1.080525398` | 7 | 12 | 3994 |
| `tokens201` | `1.140727997` | 11 | 13 | 3994 |
| `tokens301` | `1.371309280` | 10 | 12 | 3994 |
| `tokens401` | `0.959306717` | 8 | 9 | 3994 |

## Working Hypotheses

### H1: The row3456 compensation is a separate accepted fix

row3456は現行manifestで保持する。hidden3994探索中にrow3456補償を外すと、別問題が混ざり、評価が不安定になる。

### H2: Layer6 hidden3994 MLP scaling is real but not globally safe

Layer6 MLP row-scaleはtokens1/tokens101/tokens301に効くが、tokens401では入力ドリフト増幅を悪化させる。したがって、単独採用はしない。

### H3: Layer8-upfit is close but too strong

`layer8-upfit` はtokens201を改善し、tokens401をhard gate内に保つが、tokens1を悪化させる。弱い倍率、またはgate/upの片側だけを調整する候補にまだ価値がある。

### H4: Manifest row-scaleで通らなければquantizer policy問題である

hidden3994は複数fixtureにまたがる高感度channelであり、単一row後処理ではfixture間の符号差を吸収できない可能性がある。その場合は、activation-awareまたはrow-awareな量子化方針に移る。

## Execution Plan

### T0: State Freeze and Artifact Index, 0.5h

目的:

- 現在のbaseline、5 fixture、既存候補、reject理由を1つのindexに固定する。

手順:

1. `git status --short --branch` を確認する。
2. 既存gate artifactsを列挙する。
3. baseline package、p4p65 package、layer8 manifest packagesの存在を確認する。
4. `qwen-prefix-row-scale-debug-report-2026-07-05.md` とこの計画の前提が一致するか確認する。

Deliverables:

- `benchmarks/results/2026-07-05/engine/qwen-hidden3994-resolution-v0.2/artifact-index.md`

Exit Criteria:

- この計画の数値表とローカルartifactが一致している。

### T1: Reproducible Manifest Package Builder, 1.5h

目的:

- smoke-onlyではなく、manifest metadata付きpackageを複数作って5 fixture gateへ流せるようにする。

手順:

1. `tools/build-qwen-row-scale-manifest-package.py` を追加する。
2. 入力は次にする。
   - source package path
   - row-scale manifest JSON
   - output package path
3. source packageをhardlink copyし、`manifest.json` の `row_scale_overrides` だけを差し替える。
4. tensor名、shape、row範囲、重複、finite positive scaleを検証する。
5. dry-runで差分対象のmanifest entriesを出す。

Deliverables:

- `tools/build-qwen-row-scale-manifest-package.py`
- package builderのdry-run出力
- builder用の小さいfixtureまたはmanifest unit check

Exit Criteria:

- 既存 `layer6-layer8upfit-layer10` 相当のmanifest packageを再生成できる。
- 再生成packageの5 fixture smoke結果が既存artifactと一致する。

### T2: Weak Layer8-Upfit Manifest Grid, 2.0h

目的:

- `layer8-upfit` の強すぎるtokens1悪化を避けながら、tokens201/tokens401への効果を残せるかを確認する。

候補:

- base row3456 entries:
  - layer6 `linear_attn.out_proj.weight[3456]`
  - layer6 `mlp.down_proj.weight[3456]`
  - layer10 `linear_attn.out_proj.weight[3456]`
  - layer10 `mlp.down_proj.weight[3456]`
- layer8 candidate:
  - `model.language_model.layers.8.mlp.up_proj.weight[6340]`
  - scales: `1.004`, `1.008`, `1.012`, `1.016`, `1.020`, `1.024`, `1.028`, `1.032`, `1.0351020731907503`

追加候補:

- `mlp.gate_proj.weight` 側のfitが既存artifactから再現できる場合、gate-onlyとgate+up weak gridを別条件として作る。

手順:

1. T1のbuilderでgrid packageを作る。
2. `tools/run-qwen-prefix-smoke-matrix.py` で5 fixtureを逐次実行する。
3. `tools/summarize-qwen-prefix-smokes.py` でsummaryを作る。
4. `tools/evaluate-qwen-prefix-candidate-gates.py` でgate判定を作る。
5. tokens1悪化とtokens201改善の線形性を確認し、必要なら `1.000..1.012` を細かく切る。

Deliverables:

- `qwen-prefix-smoke-matrix-layer8-upfit-weak-grid-five-fixture/`
- `qwen-prefix-layer8-upfit-weak-grid-summary.{json,md}`
- `qwen-prefix-layer8-upfit-weak-grid-gates.{json,md}`

Exit Criteria:

- layer8-upfit弱倍率が accepted、rejected、またはpaired candidateへ進むべきか決まる。

### T3: Tokens401 Upstream Drift Chain, 2.0h

目的:

- tokens401 layer8 token9 hidden3994の入力ドリフトが、どの上流layer/moduleで作られているかを確定する。

手順:

1. tokens401でtoken9 hidden3994のcoordinate chainをlayer0..8まで再確認する。
2. token9だけでなく、layer7 token10 hidden3994も同じchainに入れる。
3. 各layerで次を比較する。
   - actual input vs golden before
   - full-reference actual-input replay vs golden after
   - package actual-input replay vs full-reference actual-input replay
4. tokens1の悪化座標 layer11 token7 hidden3994 でも同じ形式のchainを作る。
5. tokens401を改善し、tokens1を悪化させるcandidateの符号差を抽出する。

Deliverables:

- `package-golden-prefix-coordinate-chain-tokens401-v0.2-token9-hidden3994.{json,md}`
- `package-golden-prefix-coordinate-chain-tokens1-v0.2-token7-hidden3994.{json,md}`
- `qwen-hidden3994-upstream-drift-comparison-v0.2.{json,md}`

Exit Criteria:

- tokens401の主因がlayer8直前、layer7以前、または複数layer累積のどれかに分類されている。
- tokens1悪化の原因と同じ方向か逆方向かが分かっている。

### T4: Paired Manifest Candidate Search, 2.5h

目的:

- layer8 weak candidate単独では不十分な場合、tokens1悪化を相殺する小さいpaired candidateを探索する。

候補family:

1. Layer8 weak upfit + layer11 self-attn/down row-scale
2. Layer8 weak upfit + layer6 hidden3994 negative or very weak adjustment
3. Layer7/layer8 upstream row only, tokens401 chainで符号が合うもの

制約:

- 候補数は最初のbatchで最大12 conditionまでにする。
- 各candidateは必ず5 fixture gateで評価する。
- row-dot RMSEだけで採用しない。

手順:

1. T3で符号が合うrowsだけを候補にする。
2. `1.000`, `1.004`, `1.008`, fit scaleのような小さいgridに限定する。
3. layer8 weak gridの上位2条件とpairする。
4. 5 fixture gateでaccepted候補があればmanifest packageとして固定する。

Deliverables:

- paired manifest candidate JSON
- paired package matrix output
- candidate gate ranking

Exit Criteria:

- accepted候補が出る、またはmanifest補正では解けないと判断できる。

### T5: Quantizer Policy Branch, 4.0h

目的:

- Manifest補正が通らない場合に、package生成側の方針でhidden3994高感度channelを扱う。

Policy candidates:

1. row-aware error weighting
   - output projection rowsで、row reconstruction errorとprefix sensitivityを掛けた重みを使う。
2. activation-aware group scale search
   - golden/stat fixtureのpost-norm/MLP activationを使い、dot product errorを下げるscaleを選ぶ。
3. protected hot-row high format
   - tensor全体ではなく、top sensitivity rowsだけ高精度formatまたは別budgetに寄せる。
4. tensor-family policy split
   - `mlp.up_proj` / `mlp.gate_proj` / `mlp.down_proj` のうち、hidden3994 chainに効くfamilyだけを調整する。

手順:

1. まずdry-run planで対象tensor数、row数、推定size impactを出す。
2. `p4p46_inproj` に最小変更を加えたpolicyを1つ作る。
3. packageを生成する。
4. row3456相当の補償が必要ならmanifest row-scaleを維持して比較する。
5. 5 fixture gateを通す。

Deliverables:

- policy diff
- package generation summary
- five-fixture gate summary
- size / speed / reconstruction error comparison

Exit Criteria:

- policy packageがacceptedになる、またはp4p46系ではfixture間の反例を吸収できないと判断する。

### T6: Backend and Regression Verification, 1.5h

目的:

- CPUでacceptedになった候補がbackend固有の偶然ではないことを確認する。

手順:

1. accepted packageでR9700 `device_index=2` の5 fixture smokeを実行する。
2. accepted packageでV620 `device_index=1` の代表3 fixture smokeを実行する。
3. CPU/R9700/V620のmax abs、mean abs、MSE、cosineを比較する。
4. 関連テストを実行する。
   - `cargo fmt --all --check`
   - `cargo test -p ullm-engine row_scale -- --nocapture`
   - `cargo test -p ullm-engine golden -- --test-threads=1`
   - `python3 -m py_compile` for touched tools
   - `git diff --check`

Deliverables:

- backend comparison summary
- final accepted/rejected decision report

Exit Criteria:

- backend差分が既存許容範囲内である。
- docs/researchとjournalに最終判断が残っている。

## Decision Tree

1. If weak layer8-upfit grid passes:
   - promote the smallest accepted manifest package.
   - run backend verification.
2. If weak layer8-upfit is close but tokens1 still regresses:
   - run T3 and T4 paired candidate search.
3. If paired manifest candidates fail:
   - stop manifest override search.
   - move to T5 quantizer policy.
4. If quantizer policy passes:
   - document the policy as the durable fix.
5. If quantizer policy fails:
   - classify current p4p46 direct package policy as insufficient for hidden3994 under the current fixture set.
   - propose either a higher-budget policy or a broader fixture/stat calibration set.

## Risk Controls

- OOM対策:
  - smoke matrixは1 conditionずつ逐次実行する。
  - package生成はhardlink copyを優先し、payloadを不要に複製しない。
  - full-reference traceは対象layer/token/hiddenを絞る。
- 過適合対策:
  - 5 fixture gateを固定し、単一fixture改善では採用しない。
  - median improvementだけでなくmax regressionを必ず見る。
- 候補爆発対策:
  - 各batchのcondition数を最大12に制限する。
  - 次batchは前batchのgate結果を見てから作る。
- 実装修正の安全性:
  - runtime hard-codeは禁止する。
  - manifest metadataまたはquantizer policyとして再現可能にする。

## Progress 2026-07-05 16:10 JST

Implemented:

- `tools/build-qwen-row-scale-manifest-package.py`
  - manifest row-scale JSONを検証し、hardlink package copyへ `row_scale_overrides` を差し込む。
  - destination `manifest.json` はhardlinkを切ってから書く。
- `tools/generate-qwen-manifest-row-scale-grid.py`
  - base row3456 manifestにtarget row-scale entryを追加・置換し、grid JSONとpackage condition listを出す。
- `benchmarks/results/2026-07-05/engine/qwen-hidden3994-resolution-v0.2/artifact-index.md`

Important implementation note:

- 初期builderでは `manifest.json` もhardlinkされたまま書いてしまい、source package manifestを一時的に汚染した。
- 修正後はdestination側の `manifest.json` を `unlink()` してから書く。
- source baseline packageはrow3456の4-entry manifestへ復旧し、link count `1` を確認した。

Weak layer8-upfit grid:

- Generated layer8 `mlp.up_proj.weight[6340]` weak scales:
  - `1.000`, `1.004`, `1.008`, `1.012`, `1.016`, `1.020`, `1.024`, `1.028`, `1.032`, `1.0351020731907503`
- tokens1 prefilter showed:
  - `1.004`: `0.645338058 -> 0.645746231`, hard gate内
  - `1.008`: `0.645338058 -> 0.646137238`, hard gate内
  - `1.012`: `0.645338058 -> 0.646537781`, hard gate超え
- Therefore, only `1.004` and `1.008` were promoted to full five-fixture matrix.

Five-fixture selected gate:

| condition | decision | fixtures | median improvement | max regression | interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| `layer8-up6340-s1p004` | hold | 5 | `0` | `0.000408172607` | hard gate safe but aggregate effect too small |
| `layer8-up6340-s1p008` | hold | 5 | `0` | `0.000799179077` | hard gate safe but aggregate effect too small |

Key per-fixture effects for `layer8-up6340-s1p008`:

- tokens1 worsens `+0.000799179077`.
- tokens101 is neutral.
- tokens201 improves `-0.000621795654`.
- tokens301 improves `-0.0000190734863`.
- tokens401 worsens `+0.000033378601`.

T3 chain comparison:

- tokens1 layer8 hidden3994 baseline output diff is positive at layer8.
- tokens401 layer8 hidden3994 baseline output diff is negative and is the final worst coordinate.
- `layer8-up6340-s1p008` changes tokens401 layer8 hidden3994 by only `+0.000033378601` in the wrong direction.
- This confirms weak layer8-upfit mainly tweaks the tokens201 path and does not address tokens401 input-drift amplification.

T4 local tokens401 layer8 candidate:

- Existing tokens401-derived layer8 row-scale candidate:
  - layer8 `linear_attn.out_proj.weight[3994]` scale `0.9727276122005596`
  - layer8 `mlp.down_proj.weight[3994]` scale `1.012138640209781`
- It was rejected by a tokens1 prefilter:
  - tokens1 baseline `0.645338058`
  - candidate `0.662992477`
  - delta `+0.017654419`

Updated decision:

- Do not promote weak `layer8-up6340` as a standalone manifest fix.
- Do not continue direct tokens401 layer8 local row-scale as a general fix.
- The next useful branch is upstream drift/quantizer policy:
  - either identify an upstream candidate that affects tokens401 before layer8 without the tokens1 sign conflict,
  - or move to activation-aware / row-aware quantizer policy investigation.

Quantizer policy dry-run:

- Baseline `p4p46_inproj` plan:
  - high tensors: `114`
  - low tensors: `141`
  - estimated output bytes: `9121922016`
- Custom `p4p46 + mlp_up` plan:
  - high tensors: `147`
  - low tensors: `108`
  - estimated output bytes: `9225731040`
  - estimated output increase: `103809024` bytes
- Interpretation:
  - family-wide `mlp_up` high assignment is not expensive in size, but it is broad and not activation-aware.
  - A narrower tensor override policy or activation-aware row policy is more aligned with the current evidence than another broad family-wide package.

Per-tensor policy override implementation:

- Added repeatable `--aq-high-tensor <TENSOR_NAME>` to `ullm-quant`.
- Dry-run `p4p46_inproj + model.language_model.layers.8.mlp.up_proj.weight`:
  - high tensors: `115`
  - low tensors: `140`
  - estimated output bytes: `9125067744`
  - estimated output increase over baseline: `3145728` bytes
- The layer8 `mlp.up_proj.weight` tensor is high format while the neighboring layer9 `mlp.up_proj.weight` remains low format.

Targeted package prefilter:

- Built `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-layer8-mlp-up-high-row-scale-layer6-layer10.ullm.d`.
- The package keeps the existing layer6/layer10 row3456 manifest compensation.
- Package build:
  - selected tensors: `255`
  - passthrough tensors: `520`
  - failures: `0`
  - package verify: `255` quantized tensors and `520` passthrough tensors passed.
- Prefilter result:
  - tokens1 improves: `0.645338058 -> 0.627647400`
  - tokens401 regresses: `0.959306717 -> 0.974622726`
  - gate decision: `reject`, max regression `0.0153160095`
- Decision:
  - do not run the five-fixture matrix for layer8 `mlp.up_proj.weight` high-only.
  - the tokens401 sign conflict remains, so the next targeted policy probe should move to another tensor or upstream source.

Targeted layer8 qkv prefilter:

- Built `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-layer8-qkv-high-row-scale-layer6-layer10.ullm.d`.
- The package also keeps the existing layer6/layer10 row3456 manifest compensation.
- Package build:
  - selected tensors: `255`
  - passthrough tensors: `520`
  - failures: `0`
  - package verify: `255` quantized tensors and `520` passthrough tensors passed.
- Prefilter result:
  - tokens1 regresses: `0.645338058 -> 0.651521683`
  - tokens401 improves: `0.959306717 -> 0.919565201`
  - gate decision: `reject`, max regression `0.00618362427`
- Decision:
  - do not run the five-fixture matrix for layer8 `linear_attn.in_proj_qkv.weight` high-only.
  - because qkv improves tokens401 while MLP-up improves tokens1, the next targeted policy probe should test their combination.

## Expected Outcome

`layer8-upfit` の弱倍率はhard gate内には収まるが、aggregate improvementが不足し、tokens401を改善しないことが分かった。
したがって、短期の単独manifest fixとしては弱い。

今後の有力な解は、tokens401のlayer8入力ドリフトを上流で抑えるcandidate、またはactivation-aware / row-awareなquantizer policyである可能性が高い。

この計画では、どちらの場合でも「まだ闇雲にデバッグする」状態を避ける。accepted packageを得るか、manifest補正では足りない根拠を揃えてquantizer policyへ進む。

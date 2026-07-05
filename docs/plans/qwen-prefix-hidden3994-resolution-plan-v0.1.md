# Qwen Prefix Hidden3994 Resolution Plan v0.1

## 前回の要点

- Qwen3.5-9B p4p46 in-projection packageのprefix smokeでは、hidden `3994` が複数fixtureで支配的な差分座標になっている。
- Layer6 `mlp.down_proj.weight[3994]` row-scaleは、layer6/layer7の局所差分を強く下げる。
- ただし、複数fixture gateではまだ解決策ではない。
  - `layer6`: tokens201で `+0.00455665588` 悪化しreject
  - `layer6-mlp-selected`: tokens201で `+0.00403785706` 悪化しreject
  - `layer6-attn-mlp`: tokens201で `+0.0117874146` 悪化しreject
  - `extracted`: tokens1/tokens201が悪化しreject
- したがって、row-dot RMSE改善は候補生成には使えるが、採用判断にはfull-prefix multi-fixture smokeが必要である。

## 今回の変更点

この計画は、既存の `multi-fixture-row-scale-validation-plan-v0.1.md` より一段上の解決計画である。

既存計画は「候補を評価する仕組み」を作る計画だった。今回の計画は、hidden `3994` driftを解決済みと判断できる状態まで進めることを目標にする。

## Goal

Qwen3.5-9B p4p46 in-projection packageの`actual_prefix` smokeで発生しているhidden `3994` driftについて、単一fixtureに過適合しない修正方針を見つけ、package生成またはmanifest metadataとして再現可能に適用できる状態にする。

## Success Criteria

解決済みとみなす条件:

1. CPU `package-golden-prefix-smoke` の `actual_prefix` で、最低5 fixtureを通す。
   - 現在の3 fixture: token ids `1..16`, `101..116`, `201..216`
   - 追加2 fixture以上: 別token windowまたは実テキスト由来fixture
2. どのfixtureでも最終max absがbaselineより `0.001` を超えて悪化しない。
3. median final max abs improvementが `0.005` 以上である。
4. mean abs / mse / cosineが悪化方向に偏らない。
5. 修正はruntime hard-codeではなく、次のどちらかで表現する。
   - package manifest metadata
   - quantizer policy / package生成処理の改善
6. CPU合格後、R9700とV620でbackend差分が既存許容範囲内である。
7. `cargo test -p ullm-engine row_scale -- --nocapture` と関連quantizer検証が通る。

## Non-Goals

- hidden `3994` だけをruntime分岐で特別扱いしない。
- 単一fixtureだけでcandidateをpromoteしない。
- row-dot RMSEだけで採用しない。
- layer8 QKV V845 cell deltaを、複数fixtureの改善なしにpackage policyへ入れない。

## Working Hypotheses

### H1: Layer6 MLP row quantization error is real but incomplete

Layer6 `mlp.down_proj.weight[3994]` のrow-scaleは、tokens1/tokens101で安定した方向を示す。

- tokens1 scale: `1.026471714`
- tokens101 scale: `1.023383096`

しかし、tokens201では後段layer11で最終max absが悪化するため、単独row-scaleは解決策ではなく、後段伝播まで含めた補正が必要である。

### H2: Tokens201 regression is a downstream propagation problem

tokens201ではlayer6/layer7の局所差分は改善している。

- layer6: `0.537414551 -> 0.476898193`
- layer7: `0.966460228 -> 0.497438431`
- layer11 final max: `1.140727997 -> 1.145284653`

つまり、layer6での局所改善がlayer11の別経路を悪化させている可能性が高い。

### H3: Scalar row-scale alone is too weak

`layer6-attn-mlp` と `extracted` bundleは、局所RMSE改善を含むにもかかわらずgateでrejectされた。

したがって、単純なrow-scale bundleではなく、候補ごとの相互作用を探索する必要がある。

### H4: 最終解はquantizer policy改善かもしれない

manifest row-scaleで通る候補が見つからない場合、原因は「特定rowの後処理」ではなく、量子化時のrow/group weighting不足である可能性が高い。

その場合は、manifest overrideではなくquantizerのrow-aware / activation-aware policyへ進む。

## Strategy

1. 固定fixtureとgateを先に決める。
2. v0.10 traceを不足箇所へ揃える。
3. tokens201のlayer11 regressionを、input propagationとlocal layer errorに分解する。
4. row-scale候補を直接promoteせず、condition matrixとして探索する。
5. 合格候補だけをmanifestまたはquantizer policyへ昇格する。
6. CPUで合格してからGPU backendを確認する。

## Phase 1: Freeze the Benchmark Set

### Tasks

1. 現在の3 fixtureを固定する。
   - `qwen35-9b-prefix0-12-seq16`
   - `qwen35-9b-prefix0-12-seq16-tokens101-116`
   - `qwen35-9b-prefix0-12-seq16-tokens201-216`
2. 追加fixtureを2つ以上作る。
   - token ids `301..316`
   - token ids `401..416`
   - 可能なら実テキストprompt由来の16 token fixture
3. すべてのfixtureでbaseline smokeを生成し、`qwen-prefix-smoke-*summary.json` に集約する。

### Deliverables

- 追加golden fixture directories
- baseline smoke JSONL
- consolidated baseline summary

### Exit Criteria

- 最低5 fixtureのbaselineが揃っている。
- 各fixtureの最大差分座標、layer、token、hiddenが表で比較できる。

## Phase 2: Complete v0.10 Traces

### Tasks

1. tokens1の古いtraceをv0.10へ更新する。
   - layer10 token7 hidden3994 under combined
   - layer11 token7 hidden3994 under combined
   - layer7/layer8/layer9の必要箇所
2. tokens201のregression経路をv0.10で揃える。
   - layer6 hidden3994
   - layer7 hidden3994
   - layer11 token13 hidden3994
3. traceから以下を抽出する。
   - `row_dot.<projection>.scale_fit`
   - attention/MLP local delta
   - activation-path contribution
   - top dot-error terms

### Deliverables

- v0.10 trace JSONL/Markdown
- trace coverage table
- row-scale candidate table with observation counts

### Exit Criteria

- 候補ごとに、どのfixtureで観測され、どのfixtureで欠けているかが明示されている。

## Phase 3: Localize Tokens201 Layer11 Regression

### Tasks

1. layer6 candidate適用前後で、tokens201 layer11のactual inputを比較する。
2. layer11 token13 hidden3994について、次を分解する。
   - input drift
   - self-attention local delta
   - MLP local delta
   - residual propagation
3. layer11のcandidateを単独で試すのではなく、layer6 candidateとのpairとして評価する。
4. tokens1/tokens101で同じpairが悪化しないか確認する。

### Candidate Families

- layer6 MLP scale grid:
  - example range: `1.000..1.030`
  - dense around `1.020..1.027`
- layer6 MLP plus layer11 paired correction:
  - layer11 `self_attn.o_proj[3994]`
  - layer11 `mlp.down_proj[3994]`
  - only if local trace suggests paired compensation
- selected top dot-error cell deltas:
  - smoke-only first
  - promote only if multi-fixture gain survives
- quantizer policy alternatives:
  - row-aware scale selection
  - activation-weighted row objective
  - protected-row high precision or lower-error encoding

### Exit Criteria

- tokens201 regression source is classified as one of:
  - layer6 scale magnitude problem
  - layer11 paired local problem
  - broader quantizer policy problem
  - fixture-specific non-generalizable behavior

## Phase 4: Build a Gated Search Loop

### Tasks

1. Add a candidate grid generator.
   - Input: tensor suffix, row index, scale range, step count
   - Output: smoke row-scale JSON files
2. Run candidates through `tools/run-qwen-prefix-smoke-matrix.py`.
3. Aggregate with `tools/summarize-qwen-prefix-smokes.py`.
4. Evaluate with `tools/evaluate-qwen-prefix-candidate-gates.py`.
5. Rank by:
   - no hard reject
   - median improvement
   - worst fixture regression
   - mean abs / mse side effects

### Deliverables

- `tools/generate-qwen-row-scale-grid.py`
- candidate matrix output directory
- candidate gate ranking markdown

### Exit Criteria

- A top candidate or candidate family is clearly accepted, rejected, or moved to quantizer-policy investigation.

## Phase 5: Implement the Durable Fix

### Path A: Manifest Metadata Fix

Use this path only if a row-scale/cell candidate passes all gates.

Tasks:

1. Emit manifest-schema JSON with accepted entries.
2. Build or patch package manifest with accepted entries.
3. Re-run all fixtures without smoke-only CLI overrides.
4. Add package artifact summary and gate table.

Acceptance:

- manifest-only run matches candidate run.
- no fixture hard regression.

### Path B: Quantizer Policy Fix

Use this path if manifest row/cell candidates cannot pass gates.

Tasks:

1. Identify affected tensors and rows.
2. Add a quantizer-side policy that reduces hidden3994 row error without fixture-specific hardcoding.
3. Rebuild package.
4. Re-run the same fixture matrix.
5. Compare against manifest override attempts.

Candidate policy examples:

- row-aware error weighting for selected tensors
- activation-aware weighting using golden/stat fixture inputs
- protected-row encoding for rare high-impact rows
- scale format adjustment for affected tensor family

Acceptance:

- package is generated by policy, not patched manually.
- same fixture gates pass.
- unrelated rows/tensors do not show large regressions.

## Phase 6: Backend and Regression Verification

### CPU Checks

- `package-golden-prefix-smoke` all accepted fixtures
- `cargo test -p ullm-engine row_scale -- --nocapture`
- relevant `ullm-quant` tests
- JSON artifact parse checks

### GPU Checks

After CPU acceptance:

- R9700 actual_prefix smoke on accepted fixture subset
- V620 actual_prefix smoke on accepted fixture subset
- compare CPU/GPU max abs and cosine

### Documentation

- Update `docs/research/qwen-prefix-row-scale-debug-report-2026-07-05.md`
- Add final root-cause report
- Update plan status with accepted/rejected decision table

## Decision Tree

1. If layer6 MLP scale grid finds a no-regression candidate:
   - promote as manifest candidate and verify.
2. If layer6 MLP needs a layer11 paired correction:
   - test pair across all fixtures.
   - promote only if pair passes gate.
3. If no row-scale/cell pair passes:
   - stop manifest override work.
   - move to quantizer policy fix.
4. If quantizer policy still cannot improve without regressions:
   - expand fixture set and re-evaluate whether hidden3994 is a symptom of broader p4p46 format limits.

## Risks

- Candidate search can overfit the current three fixtures.
  - Mitigation: add at least two more fixtures before accepting.
- Row-scale can improve early layers while worsening later layers.
  - Mitigation: final max abs gate across full prefix remains mandatory.
- Bundled candidates can interact destructively.
  - Mitigation: test single, pair, and bundle conditions separately.
- GPU verification can expose backend-specific behavior.
  - Mitigation: CPU acceptance first, then GPU confirmation.
- Memory pressure can grow with fixture/candidate count.
  - Mitigation: sequential smoke matrix runner, no parallel full-prefix smokes by default.

## Next Actions

1. Generate two additional golden fixtures.
2. Regenerate missing v0.10 traces for tokens1 and tokens201.
3. Add a row-scale grid generator.
4. Sweep layer6 MLP selected scale around `1.000..1.027`.
5. If every layer6-only scale fails, test paired layer11 candidates.
6. If pair candidates fail, switch to quantizer policy investigation.

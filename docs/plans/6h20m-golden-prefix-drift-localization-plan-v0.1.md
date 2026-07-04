# 6h20m golden prefix drift localization plan v0.1

## Purpose

前回の`package-golden-prefix-smoke`では、Qwen3.5-9Bの`seq8`、`0..4` prefixはCPU/R9700/V620でほぼ同じ結果になった。一方で`0..8`へ広げると、layer 4以降で`numeric_drift`が出た。

この計画の目的は、layer 4以降のdriftを「prefix連鎖で入力がズレた結果」なのか、「各layerを参照before hiddenから単独で実行しても出るlayer固有の差分」なのかに分解することである。結果として、次に進むべき作業を次のどちらかに絞れる状態にする。

- prefix接続、position、layer kind切替、residual受け渡しの修正
- AQ packageの量子化方針、特にlinear attention layer群の精度方針の見直し

この計画は文章生成品質やfull model generationを測るものではない。既存のgolden prefix fixtureと`.ullm.d` packageを使い、layer境界の数値差分を機械可読に増やす。

## Time Budget

基準は前回の実測完了時間`1139秒`である。20倍は`22780秒`、つまり約`6時間19分40秒`なので、本計画は`6h20m`規模として設計する。

この時間は、単純な再実行だけではなく、Rust側のdrift localization report拡張、window実験、分析スクリプト、journalまとめまで含めた見積もりである。

## 前回の要点

- `tools/export-qwen-golden-tensors.py`は`--layer-range START:END`でprefix fixtureを作れる。
- `GoldenTensorFixture`は連続layer selection、`read_initial_before_f32`、`read_layer_before_f32`、`read_layer_after_f32`を持っている。
- `package-golden-prefix-smoke`はmixed prefixを実行できる。
  - layers `0,1,2,4,5,6,...`は`linear_attn`
  - layers `3,7,11,...`は`self_attn`
- `0..4 seq8`はCPU/R9700/V620でbackend差がほぼない。
- `0..8 seq8`はCPU/R9700ともにlayer 4以降で`numeric_drift`になった。
- CPUとR9700のmetricが近いため、現時点ではGPU backend固有の差分ではなさそうである。

## Current Evidence

`benchmarks/results/2026-07-04/engine/package-golden-prefix-smoke-cpu-prefix0-8-seq8.jsonl`:

| layer | class | MSE | mean abs | max abs | cosine |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | possible_quantization_error | 0.007257800 | 0.052020688 | 5.489426613 | 0.757005733 |
| 1 | possible_quantization_error | 0.012592061 | 0.068731415 | 9.846384048 | 0.726112674 |
| 2 | possible_quantization_error | 0.025978289 | 0.095062335 | 15.920772552 | 0.504583529 |
| 3 | possible_quantization_error | 0.036098765 | 0.122858104 | 14.526961327 | 0.677866091 |
| 4 | numeric_drift | 0.085715877 | 0.149483227 | 18.845571518 | 0.135689776 |
| 5 | numeric_drift | 0.109585372 | 0.165630067 | 24.462188721 | 0.077736798 |
| 6 | numeric_drift | 0.264669272 | 0.228727875 | 44.618324280 | -0.061638164 |
| 7 | numeric_drift | 0.238262119 | 0.227860239 | 31.469402313 | -0.083518201 |

R9700の`0..8 seq8`も同じ傾向で、差分は記録精度上ほぼ一致する。

## 今回の変更点

- 既存のprefix output metricに加えて、各layer実行前の`current_hidden`とfixtureの`layer_N_before`を比較する。
- 1つのCLI実行で次の2モードを扱えるようにする。
  - `actual_prefix`: 現在の挙動。前layerのactual outputを次layer inputにする。
  - `golden_before_each_layer`: 各layerをfixtureの`layer_N_before`から実行し、prefix累積を切り離す。
- JSONLへinput drift metricを追加する。
  - `input_mse`
  - `input_mean_abs_diff`
  - `input_max_abs_diff`
  - `input_cosine_similarity`
  - `input_failure_class`
  - `run_mode`
- CPU/R9700で、`0..8` prefix、`4..8` reset window、`4..5`から`7..8`までの単層windowを比較する。
- JSONL群を読み、prefix起点、layer index、deviceごとの比較表を作る分析スクリプトを追加する。

## 次の行動

最初の実装は、既存`package-golden-prefix-smoke`を壊さず、drift localization用のrun modeとinput metricを足すことである。既存結果との互換性を守るため、既存コマンドのデフォルトは`actual_prefix`のままにする。

## Target Scope

対象:

- model: Qwen3.5-9B
- package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d`
- fixture: `benchmarks/golden/2026-07-04/qwen35-9b-prefix0-8-seq8`
- primary devices:
  - CPU fallback `0`
  - R9700/RDNA4 `2`
- optional device:
  - V620/RDNA2 `1`
- required sequence length:
  - `seq8`
- optional sequence length:
  - `seq16`
- required layer ranges:
  - `0..8` actual prefix
  - `4..8` actual prefix starting from golden layer 4 before
  - `4..8` golden-before-each-layer
  - `4..5`, `5..6`, `6..7`, `7..8` single layer windows

対象外:

- tokenizer統合
- embedding、final RMSNorm、lm_head
- sampling
- full 36 layer generation
- fused AQ kernel
- quantizer再設計
- 新しい`.ullm.d` package生成
- seq256以上の長いloss評価

## Working Hypotheses

1. `0..8`のlayer 4 driftは、layer 0..3のactual outputを入力にしたことによるprefix累積で増幅している。
2. layer 4をfixtureの`layer_4_before`から単独実行してもdriftする場合、linear attention layer 4固有のAQ差分またはRust/HF実装差分が有力になる。
3. `golden_before_each_layer`ではlayer 4..7のoutput driftが小さいが`actual_prefix`では大きい場合、prefix接続または累積入力分布の問題が有力になる。
4. CPU/R9700でinput/output driftが同じ形なら、backend差分ではなくpackage weight、アルゴリズム境界、または量子化方針側の問題である。

## Deliverables

1. Drift localization run mode
   - `package-golden-prefix-smoke`の既存挙動を保つ。
   - 追加run modeとして`golden_before_each_layer`を扱えるようにする。
   - 既存CLI互換を優先する。既存positional tailを壊すなら、新コマンド`package-golden-prefix-drift-smoke`として分離する。

2. Per-layer input drift report
   - 各layerの実行前に`current_hidden`と`fixture.read_layer_before_f32(layer_index)`を比較する。
   - JSONLへinput drift metricとrun modeを保存する。
   - output metricのfailure classとは別に、input側のfailure classも保存する。

3. Window validation matrix
   - `0..8` actual prefixを再実行する。
   - `4..8`をlayer 4のgolden beforeから実行する。
   - `4..5`, `5..6`, `6..7`, `7..8`を単層windowとして実行する。
   - CPU/R9700の差分が小さいかを確認する。

4. Analysis script
   - `tools/analyze-golden-prefix-drift.py`を追加する。
   - 複数JSONLを読み、layerごとに次を出す。
     - input drift
     - output drift
     - input driftからoutput driftへの増幅
     - CPU/R9700 backend delta
     - `actual_prefix`と`golden_before_each_layer`の差
   - summary JSONとMarkdown tableを出力する。

5. Result artifacts
   - `benchmarks/results/2026-07-05/engine/`へ`.txt`と`.jsonl`を保存する。
   - 分析結果を同じ日付配下へ保存する。
   - journalへ実験条件、主要表、判断、残課題を残す。

## Execution Strategy

subagentを使う場合は、次の分担にする。

- Worker A: Rust側のinput drift metricとrun mode実装。
- Worker B: `tools/analyze-golden-prefix-drift.py`とJSONL summary生成。
- Main agent: CLI互換性確認、matrix実行、結果解釈、journal統合。

Workerのwrite scopeは分ける。

- Worker A:
  - `crates/ullm-engine/src/main.rs`
  - 必要なら`crates/ullm-engine/src/golden.rs`
- Worker B:
  - `tools/analyze-golden-prefix-drift.py`
  - 必要なら小さいfixture/report parsing test data

## 6h20m Task Breakdown

### T0: Baseline audit, 0.35h

目的:

- 現在のbranch、既存fixture、既存resultを確認し、計画実行前の前提を固定する。

手順:

1. `git status --short --branch`
2. `cargo fmt --all --check`
3. `cargo check -p ullm-engine`
4. `benchmarks/golden/2026-07-04/qwen35-9b-prefix0-8-seq8/metadata.json`のlayer rangeとpayload shapeを確認する。
5. 既存`0..8` JSONLを読み、layer 4以降のdriftを再確認する。

完了条件:

- 未コミット差分の有無を把握している。
- `prefix0-8` fixtureがlayer 0..7のbefore/afterを持つことを確認している。

### T1: Drift report schema, 0.55h

目的:

- 既存JSONLと互換を保ちながら、input driftを表現できるschemaを決める。

追加field:

- `run_mode`
- `input_mse`
- `input_mean_abs_diff`
- `input_max_abs_diff`
- `input_cosine_similarity`
- `input_failure_class`
- `input_expected_preview`
- `input_actual_preview`
- `input_diff_preview`

完了条件:

- 既存の`package-golden-prefix-smoke` report readerが壊れない。
- 新fieldがない古いJSONLも分析スクリプトで扱える。

### T2: Rust localization mode, 1.10h

目的:

- Rust側で`actual_prefix`と`golden_before_each_layer`を実行できるようにする。

手順:

1. run mode parserを追加する。
2. 既存CLI互換が難しい場合は`package-golden-prefix-drift-smoke`を追加する。
3. 各layer前に`fixture.read_layer_before_f32(layer_index)`を読む。
4. `actual_prefix`では、現在の`current_hidden`と参照beforeを比較してから実行する。
5. `golden_before_each_layer`では、各layerのinputを参照beforeへ差し替えてから実行する。
6. JSONLにinput metricを出す。

完了条件:

- `actual_prefix`の既存結果と同等のoutput metricが出る。
- `golden_before_each_layer`でlayer単体のoutput metricを1回のrange実行で取れる。
- `cargo test -p ullm-engine golden -- --test-threads=1`が通る。

### T3: Analysis script, 0.95h

目的:

- 実験結果を手作業で読まずに、layerごとの原因候補を表にする。

手順:

1. JSONLを複数受け取るCLIを作る。
2. `device_index`, `backend`, `run_mode`, `layer_start`, `layer_end_exclusive`, `layer_index`でgroupingする。
3. `actual_prefix`と`golden_before_each_layer`を同一layerで比較する。
4. CPU/R9700のmetric差を出す。
5. Markdown tableとsummary JSONを出力する。

完了条件:

- 古いJSONL、新しいJSONLの両方を読める。
- first bad layer、largest output MSE layer、largest backend delta layerがsummaryに出る。

### T4: Required validation matrix, 1.25h

目的:

- `seq8`でlayer 4以降のdriftをprefix累積と単層差分へ分ける。

必須実行:

| device | run mode | range |
| --- | --- | --- |
| CPU `0` | `actual_prefix` | `0..8` |
| R9700 `2` | `actual_prefix` | `0..8` |
| CPU `0` | `actual_prefix` | `4..8` |
| R9700 `2` | `actual_prefix` | `4..8` |
| CPU `0` | `golden_before_each_layer` | `4..8` |
| R9700 `2` | `golden_before_each_layer` | `4..8` |

単層window:

| device | range |
| --- | --- |
| CPU `0` | `4..5`, `5..6`, `6..7`, `7..8` |
| R9700 `2` | `4..5`, `5..6`, `6..7`, `7..8` |

完了条件:

- すべての必須実行に`.txt`と`.jsonl`が残る。
- 結果fileだけでrun mode、device、range、layer metricを復元できる。

### T5: Structured linear-attention detail, 0.90h

目的:

- `linear_attn` layerの内部runtime self-checkを文字列ではなく、比較しやすいJSON fieldへ出す。

手順:

1. 現在の`runtime_line`を文字列として残す。
2. 取れる範囲で、linear attention sequence runの主要metricをstructured detailsへ追加する。
3. structured化が大きくなる場合は、layer 4と6だけを対象に限定する。

完了条件:

- layer 4または6で、HF golden outputとの差分とRust内部self-check差分を同じJSON rowから読める。
- 内部self-checkが小さいのにHF差分が大きい場合、量子化または参照実装差分が有力と判断できる。

### T6: Optional seq16 spot check, 0.60h

目的:

- driftのsequence length依存を最小コストで見る。

手順:

1. 時間とVRAMに余裕があれば`seq16`、`layer_range 4:8`のfixtureを作る。
2. CPU/R9700の`golden_before_each_layer 4..8`だけを実行する。
3. seq8と同じlayerで悪化するかを比較する。

完了条件:

- seq16を実行した場合は、結果をoptionalとして明記する。
- 実行できなかった場合も、未実行理由をjournalへ残す。

### T7: Result synthesis and journal, 0.63h

目的:

- 次の作業判断を短く決める。

判断基準:

- `input drift`がlayer 4で大きく、`golden_before_each_layer`のoutput driftが小さい:
  - prefix累積またはlayer 0..3出力分布の問題を優先する。
- `input drift`が小さいのにlayer 4 output driftが大きい:
  - layer 4 linear attentionまたはMLP block単体の問題を優先する。
- CPU/R9700差が大きい:
  - runtime backend差分を優先する。
- CPU/R9700差が小さい:
  - AQ package、参照実装差分、またはlayer kind実装差分を優先する。

完了条件:

- journalに実験表、要約、次の1手が残っている。
- `git diff --check`が通っている。
- 必要な範囲で`cargo fmt --all --check`、`cargo check -p ullm-engine`、対象unit testが通っている。

## Command Template

```bash
PKG=/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d
FIX=benchmarks/golden/2026-07-04/qwen35-9b-prefix0-8-seq8
OUT=benchmarks/results/2026-07-05/engine
mkdir -p "$OUT"

./target/debug/ullm-engine package-golden-prefix-smoke \
  "$PKG" "$FIX" 0 1048576 4 8 64 10000000 0 \
  "$OUT/package-golden-prefix-smoke-cpu-prefix4-8-seq8.jsonl" \
  > "$OUT/package-golden-prefix-smoke-cpu-prefix4-8-seq8.txt"
```

drift modeを新コマンドとして分ける場合の例:

```bash
./target/debug/ullm-engine package-golden-prefix-drift-smoke \
  "$PKG" "$FIX" 0 1048576 4 8 64 10000000 0 golden_before_each_layer \
  "$OUT/package-golden-prefix-drift-cpu-local4-8-seq8.jsonl" \
  > "$OUT/package-golden-prefix-drift-cpu-local4-8-seq8.txt"
```

## Risks

- `numeric_drift`の閾値は粗い。実用上の許容差分かどうかは、この計画だけでは判断しない。
- `linear_attn`内部のstructured metric化が既存helperの戻り値を大きく変える場合、T5はlayer 4/6だけの限定実装に縮小する。
- seq16はVRAMと時間の影響を受けるため、必須ではなくspot check扱いにする。
- 既存CLIのpositional argumentsはすでに長い。互換性を壊しそうなら新コマンドを選ぶ。

## Exit Criteria

この計画は次を満たしたら完了とする。

- `actual_prefix`と`golden_before_each_layer`の比較で、layer 4以降のdriftが入力累積由来かlayer固有由来かを暫定分類できる。
- CPU/R9700の差分がbackend固有かどうかを判定できる。
- `benchmarks/results/2026-07-05/engine/`に主要JSONLとsummaryが残っている。
- `journal/2026/07/05/`に、結果表と次の作業判断が残っている。
- 追加したコードに対して、最低限のfmt/check/testが通っている。

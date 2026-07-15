# P2 AQ4 layer0 QKV/Z combined hybrid fidelity

## 前回の要点

- 既存のlayer0 QKV、Z、A+B isolation artifactは、3行の固定CPU入力で各投影からrecurrent、attention block、layer outputまでを比較している。
- QKVとZを同時に置き換えたcombinedケースは未定義で、単独効果の加算だけでは相互作用を判定できない。

## 今回の変更点

- `tests/test_aq4_layer0_qkv_z_combined_hybrid.py` を追加した。
- baseline、qkv_only、z_only、combinedの4 variantについて、QKV/Z/recurrent/attention block/layer outputのSHA-256、有限値、3 stepのrecurrent state digestを検証する。
- 各variantとbaselineの5境界メトリクス、`combined - baseline` と `(qkv_only - baseline) + (z_only - baseline)` の相互作用残差メトリクスを、per-step/aggregateおよび非有限値なしで検証する。
- 既存QKV、Z、A+B artifactのlayer-output relative L2をnumeric comparisonとして束縛し、CPU、promotion=false、holdout=not_run、policy_not_evaluated、thresholds=null、3 rowsを固定する。
- 予定artifact rootは `benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-layer0-qkv-z-combined-hybrid-fidelity-v0.1/`、report schemaは `ullm.aq4_layer0_qkv_z_combined_hybrid_fidelity.cpu.v1` とした。

## 次の行動

- combined artifact生成後に本テストを実行し、sidecar hash、state digest、相互作用残差、既存QKV/Z/A+B比較値を確認する。artifactは生成済みである。
- GPU、service、promotion、holdoutは実行しない。
- layer-output relative L2はQKV-onlyが `7.890094626902091e-04`、Z-onlyが `6.983383610254553e-04`、QKV+Z combinedが `1.0925492523688434e-03` だった。
- recurrentは完全加算で相互作用残差が0だった。attention blockは加算予測に対して `2.00930022062779e-02`、layer outputは `2.2006302980976514e-02` の相互作用残差で、どちらも小さい相殺として観測された。
- QKV+Z combinedのlayer寄与は既存A+B combinedの約8.78倍である。次はB追加よりもlayer-depth accumulationを優先する。

## 検証

- `python3 -m py_compile tests/test_aq4_layer0_qkv_z_combined_hybrid.py`: 成功。
- `pytest -q tests/test_aq4_layer0_qkv_z_combined_hybrid.py`: 6 passed。
- `git diff --check`: 成功。
- `cargo check -p ullm-engine --bin ullm-engine`: 成功。
- `cargo test -p ullm-engine --bin ullm-engine -- --test-threads=1`: 26 passed。
- 通常3-step smoke stdout SHA256: `9ac224cc444569bb9e5c4c493eacf4007c06c862c03466da31a058a123e4ad9b`（変更なし）。

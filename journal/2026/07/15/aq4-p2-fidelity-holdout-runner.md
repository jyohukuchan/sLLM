# AQ4 P2 holdout runner実装記録

## 前回の要点

`generate-aq4-p2-fidelity-holdout.py` と既存Rust captureはcalibration 24行を固定していた。freeze receipt後にholdout 24行を一度だけ評価する実行境界は未実装だった。

## 今回の変更点

- Rust `ullm-aq4-fidelity-capture`へ既定動作を変えない`--subset holdout`、`--cases-file`、holdout SHA拘束を追加した。
- `prepare-aq4-p2-fidelity-holdout-cases.py`でholdout 24行だけをsource casesへ変換する。
- `run-aq4-p2-fidelity-holdout.py`でCPU preflight、one-shot execute、attempt marker、failure/result receiptを実装した。
- failureはtimeout/OOM/nonzero/partial/validator違反をimmutableに封じ、success/No-Goはremainingを0にする。freeze policyは再導出しない。
- source/active identity、split/policy/cases/freeze/actual receipt SHA、finite/shape/greedy/top-k/relative-L2/stateを検証する。

## 検証

- `pytest -q tests/test_aq4_p2_fidelity_holdout_protocol.py tests/test_aq4_p2_fidelity_holdout_runner.py tests/test_aq4_p2_fidelity_holdout_cases.py tests/test_qwen35_aq4_fidelity_capture.py`（19 passed）
- `cargo check -p ullm-engine --bin ullm-aq4-fidelity-capture`
- `cargo test -p ullm-engine --bin ullm-aq4-fidelity-capture -- --test-threads=1`（7 passed）
- `python3 -m py_compile tools/run-aq4-p2-fidelity-holdout.py tools/prepare-aq4-p2-fidelity-holdout-cases.py`

`cargo fmt --all --check` と対象Rust fileの直接checkは基点由来の既存未整形（file冒頭のserde_json import、既存の長いjson!/一行関数群）で失敗する。変更前commitをrustfmtしてから変更後をrustfmtしたread-only差分では、差分はrecursion limit、subset/cases SHA分岐、runtime identity fields、usageだけであり、既存コードの整形差分をcommitへ混入させていない。

GPU実行は行っていない。実機での残課題は、凍結済みsource artifactをholdout casesで生成し、指定Rust binaryを一回だけ起動してresult receiptを作ることだ。

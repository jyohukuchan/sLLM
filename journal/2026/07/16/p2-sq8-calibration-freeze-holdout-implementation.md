# P2 SQ8 校正・凍結・holdout 実装

## 前回の要点

SQ8 overlay の promotion receipt は、prepared と actual_verified を分離し、実行時の telemetry、artifact、worker、served-model、package を hash-bind する既存境界を利用できる状態だった。実GPU校正と実holdoutは許可しない。

## 今回の変更点

- `tools/qwen35_aq4_sq8_fidelity_protocol.py` を追加した。24行固定のSQ8 plan、metrics、freeze receipt、holdout preflight、不可逆 attempt ledger を filesystem-only で検証する。
- actual_verified receipt が無い場合は prepared receipt から `preflight_only` plan だけを作成し、freeze/holdoutをfail-closedにした。
- receipt SHA、request ID、prepared/maintenance/executor、token SHA、telemetry binding、overlay content/tensor-set、served/worker/package/source-v32、cases/split/policyをidentityへ固定した。
- freezeは24行の各metricを再計算し、relative-L2各行の1超過を拒否する。holdoutはsentinelを先にcreate-new公開し、成功・失敗・クラッシュ後の再試行を拒否する。
- `ullm-aq4-fidelity-capture` にSQ8 overlayの型付きCLI引数とserved identity分岐を追加した。AQ4経路は既存引数のまま維持し、SQ8はartifact/binding/content/source/packageを一括必須化し、ロード後identityを再確認する。

## 検証

- `python3 -m unittest -v tests/test_qwen35_aq4_sq8_fidelity_protocol.py` — 5 tests passed。
- `rustfmt --edition 2021 crates/ullm-engine/src/bin/ullm-aq4-fidelity-capture.rs` — pass。
- `CARGO_BUILD_JOBS=1 cargo check -p ullm-engine --bin ullm-aq4-fidelity-capture` — pass（既存C++ warningのみ）。
- `PYTHONPATH=. pytest ...` の既存対象は42 passed、1 failed。失敗は既存 `tests/test_generate_served_model.py` の reasoning worker絶対パスがisolated worktreeのrootを指さないためで、今回の変更とは無関係。

## 次の行動

実GPU、service、sudo、real holdoutは未実施。親でcommitを確認後、必要ならactual receiptのfixture validatorとproduction generateのhash-bindingを追加監査する。

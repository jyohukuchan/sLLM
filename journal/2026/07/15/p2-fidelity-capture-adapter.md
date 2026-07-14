# P2 24-row fidelity capture adapter

## 前回の要点

P2 fidelity split は `ullm.aq4_p2_fidelity_split.v1` として 24 calibration / 24 holdout に固定されている。measurement lane は GPU 実測前であり、source BF16 と active AQ4 の行を同一 full-context / step=0 で結合する専用境界が必要だった。

## 今回の変更点

- `prepare-qwen35-aq4-fidelity-cases.py` が最新 split の manifest/policy/calibration SHA と fixture hash を再検証し、24件の `ullm.qwen35_aq4_source_calibration_cases.v1` と実行 plan を生成する。
- `ullm-aq4-fidelity-capture` は active package を一度だけロードし、`all_m1` は requested-M をラベルとして M=1 dispatch、`cold_batched` は requested-M dispatch とする。いずれも同じ全 prompt を処理し、final hidden/full logits の step=0 row を sidecar へストリームする。
- `capture-qwen35-aq4-fidelity.py` は source/active full-vector sidecar を同じ row identity で走査し、greedy、順序付きtop10、retention、cosine、relative-L2、max-abs、bounded sufficient statistics を metrics JSON に出力する。`validate-qwen35-aq4-fidelity-capture.py` は 24件、split/policy/cases SHA、重複・欠落・余分、有限値、shape、top10、統計の境界を検証する。
- sidecar は hidden 24×4096×F32、logits 24×248320×F32（合計約24.2 MiB）を上限とし、Rust observer は一回につき最大 chunk 1,048,576 elements の小さいバッファだけを保持する。モデル側の GPU 観測、source exporter の CPU 時間、実測 GPU 時間はこの段階では未実施。

## 次の行動

1. plan に記録された source exporter CLI で BF16 source artifact を一度だけ作る。
2. `ullm-aq4-fidelity-capture` を独立 review 後に一度だけ active AQ4 上で実行する（現時点では GPU/service 実行禁止）。
3. source/active artifact を `capture-qwen35-aq4-fidelity.py` で結合し、metrics validator と既存 freeze validator へ渡す。

検証済み: `CARGO_BUILD_JOBS=1 cargo check --bin ullm-aq4-fidelity-capture`、`cargo test --bin ullm-aq4-fidelity-capture`（2件）、Python fidelity tests（5件）、Python compile、`git diff --check`。GPU/service 実行は未実施。

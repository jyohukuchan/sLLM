# P2 24-row fidelity capture adapter

## 前回の要点

P2 fidelity split は `ullm.aq4_p2_fidelity_split.v1` として 24 calibration / 24 holdout に固定されている。measurement lane は GPU 実測前であり、source BF16 と active AQ4 の行を同一 full-context / step=0 で結合する専用境界が必要だった。

## 今回の変更点

- `prepare-qwen35-aq4-fidelity-cases.py` が最新 split の manifest/policy/calibration SHA と fixture hash を再検証し、24件の `ullm.qwen35_aq4_source_calibration_cases.v1` と実行 plan を生成する。
- `ullm-aq4-fidelity-capture` は active package を一度だけロードし、`all_m1` は requested-M をラベルとして M=1 dispatch、`cold_batched` は requested-M dispatch とする。いずれも同じ全 prompt を処理し、final hidden/full logits の step=0 row を sidecar へストリームする。
- `capture-qwen35-aq4-fidelity.py` は source/active full-vector sidecar を同じ row identity で走査し、greedy、順序付きtop10、retention、cosine、relative-L2、max-abs、bounded sufficient statistics を metrics JSON に出力する。`validate-qwen35-aq4-fidelity-capture.py` は 24件、split/policy/cases SHA、重複・欠落・余分、有限値、shape、top10、統計の境界を検証する。
- sidecar は hidden 24×4096×F32、logits 24×248320×F32（合計約24.2 MiB）を上限とし、Rust observer は一回につき最大 chunk 1,048,576 elements の小さいバッファだけを保持する。モデル側の GPU 観測、source exporter の CPU 時間、実測 GPU 時間はこの段階では未実施。
- `b457ef2` で active manifest の `created_utc` を source manifest から継承し、required HIP guard 集合を完全一致で拘束した。metrics validator は split validator の戻り値と calibration JSONL の実 SHA を再計算し、root binding の一致を要求する。最新 split は manifest `966878f3d9eb13f5b485825208f8072521724f308f5ee3d8a003b0b051198887`、policy `302c3219af286a970ddf39ed090021ef102b51b2d188c0ff337f6b9dd04d1a03`、calibration `20c09f22bb1ca4dfac907de09febddb01ed0228c3f4a17c01efd646491e0983f` である。
- source checkpoint index の既存実体は 19,306,393,663 B（17.98 GiB）で、CPU exporter の 2.0 倍 memory preflight は 38,612,787,326 B（35.96 GiB）以上の available memory を要求する。source/active の新しい実測時間・RSS・VRAM はまだ取得していない。
- `8be3d37` で実行CLIの split/policy/calibration SHA と served/package/worker/guard/device/quantized-revision を必須引数として固定し、plan template と `validate_plan` に伝播した。Rust は package manifest の `source_model_dir` から upstream revision、checkpoint aggregate、tokenizer aggregate を再計算し、source artifact と一致する場合だけ active capture を許可する。upstream revision と quantized artifact revision は別々に記録し、同値を拒否する。policy schema、metric role/aggregation/formula、relative-L2 rejection もRust側で再検証する。
- Rust observer は hidden/logit non-finite を検出した時点で中断する。strict duplicate-key JSON、canonical package tree hash（既存 production runner と同じ `relative\0 + raw file SHA + newline`）、renameat2 `NOREPLACE` 出力公開、dangling symlink/duplicate/nonfinite/provenance のnegative testsを追加した。
- `ae543a9` で metrics lane が source/active の model ID、upstream revision、tokenizer aggregateを再検証する。source exporter の `--threads` は PyTorch intra/inter-op threadsだけを制御し、plan は再現性のため16固定とする。BF16 reduction順序の最下位bit/top-k tieが変わり得るため、個別artifactの全SHAとlegacy cross-checkを再検証する。

## 次の行動

1. plan に記録された source exporter CLI で BF16 source artifact を一度だけ作る。
2. `ullm-aq4-fidelity-capture` を独立 review 後に一度だけ active AQ4 上で実行する（現時点では GPU/service 実行禁止）。
3. source/active artifact を `capture-qwen35-aq4-fidelity.py` で結合し、metrics validator と既存 freeze validator へ渡す。

検証済み: `CARGO_BUILD_JOBS=1 cargo check --bin ullm-aq4-fidelity-capture`、`cargo test --bin ullm-aq4-fidelity-capture`（7件）、Python fidelity tests（6件）、full calibration/path tests（30件+15 subtests）、Python compile、`git diff --check`。実在splitのexpected SHA付きprepare smokeも24行で成功した。GPU/service 実行は未実施。

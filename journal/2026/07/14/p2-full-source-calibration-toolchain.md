# P2 full BF16 source calibration toolchain

## 前回の要点

既存のQwen3.5-9B AQ4 P2 source-oracle v2は、3 rowsのbounded hidden/logit sampleだけを保存しており、full-vector relative L2/max_absを計算できなかった。

## 今回の変更点

- `tools/export-qwen35-aq4-source-calibration.py` を追加した。旧v2とは別schema/rootで、hidden 4096とlogits 248320をf32le sidecarへ64K chunkで保存する。
- `tools/validate-qwen35-aq4-p2-full-calibration.py` を追加した。identity、legacy v2 3-row cross-check、offset/hash/top-k/nonfinite、duplicate/unknown/symlink/TOCTOUを検証する。
- `tools/compare-qwen35-aq4-p2-calibration.py` を追加した。source_gate/path_gateを分離し、chunk streamingとf64累積で相対L2/max_abs/top-k/greedy/nonfiniteを計算する。閾値は生成しない。
- `benchmarks/workloads/qwen35-aq4-p2-source-calibration-cases-v0.1.json` と仕様書を追加した。既存2 prompt/3 rowは互換性canary用で、promotion閾値根拠ではない。
- synthetic full-vector artifactのvalidator/comparator testsを追加した。
- 独立QA後、全nested schemaとexact artifact file setを閉じ、single-link regular fileだけを受理するようにした。
- manifest、rows、sidecar、SHA256SUMSは`O_NOFOLLOW` fdへ固定し、device/inode/size/mtime/ctime/nlinkを読取前後に照合する。rowsはfile/line/recordの3上限を持つ。
- nonfinite rowは順位計算を行わないblocked証跡へ変更した。比較器は異なるchunk幅をglobal element streamとして比較し、短読/余剰を拒否する。
- 独立再QAで、hiddenだけがnonfiniteのblocked rowを比較器が誤って順位検証する不整合を修正した。hidden/logitsの両方がfiniteの場合だけtop-kを再計算し、validatorからblocked comparisonまでの回帰testを追加した。
- exporter/comparatorのpublishを`renameat2(RENAME_NOREPLACE)`へ変更し、競合時にfail-closeする。
- 独立負例を含む20 testsと既存`/tmp/qwen35-aq4-source-calibration-canary-v1`の再検証を通した。モデルは再ロードしていない。

## 次の行動

AQ4 target側のfull-vector sidecarを同じblocked/exact-file/fd-fixed contractで生成し、source gateとpath gateを別run rootで比較する。Rust/GPU/live実行は今回の範囲では行わない。
